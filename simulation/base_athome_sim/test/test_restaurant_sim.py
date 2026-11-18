"""Tests para la simulacion de Restaurant y clientes con mano levantada (HOME-104).
Sin ROS (o con mocks), determinista y ejecutable con pytest.
"""

import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Asegurar que se cargue el paquete local del worktree y no build/ o install/
SIM_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SIM_DIR.parent / "scripts"
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pytest

from base_athome_sim import gz_model, sim_scene
from base_athome_sim.sim_core import (SimArm, restaurant_base_hold_requested,
                                      restaurant_base_hold_angular_command,
                                      restaurant_handoff_confirmed,
                                      sim_arm_enable_requested)
from base_athome_perception.raised_hand import classify_raised_hand, RaisedHandState
from base_athome_perception import pose_store
from sim_domain import validate_lab_domain

RESTAURANT_SDF = SIM_DIR / "worlds" / "restaurant.sdf"
ROBOT_URDF = SIM_DIR.parents[1] / "src/hoverboard_base/urdf/robot.urdf"
RESTAURANT_POSES = SIM_DIR / "config/restaurant_arm_poses.yaml"


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s, d = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[c + x*x*d, x*y*d-z*s, x*z*d+y*s],
                     [y*x*d+z*s, c+y*y*d, y*z*d-x*s],
                     [z*x*d-y*s, z*y*d+x*s, c+z*z*d]])


def _wrist_camera_transform(joints_deg):
    """FK the checked URDF chain so the sim pose test catches camera framing drift."""
    root = ET.parse(ROBOT_URDF).getroot()
    by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    chain, link = [], "arm_camera_optical_frame"
    while link != "base_footprint":
        joint = by_child.get(link)
        assert joint is not None, f"URDF path to {link} does not reach base_footprint"
        chain.append(joint)
        link = joint.find("parent").get("link")
    transform = np.eye(4)
    joint_values = {f"joint{index}": math.radians(value)
                    for index, value in enumerate(joints_deg, 1)}
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        local = np.eye(4)
        local[:3, :3] = _rpy_matrix(rpy)
        local[:3, 3] = xyz
        transform = transform @ local
        kind = joint.get("type")
        if kind in ("revolute", "continuous"):
            motion = np.eye(4)
            motion[:3, :3] = _rotation(np.fromstring(joint.find("axis").get("xyz"), sep=" "),
                                       joint_values.get(joint.get("name"), 0.0))
            transform = transform @ motion
        elif kind == "prismatic" and joint.get("name") == "torso_lift_joint":
            # Restaurant simulation holds the lift at its upper position (zero displacement).
            pass
    return transform


def test_sim_restaurant_look_pose_keeps_staged_can_in_wrist_camera_frame():
    import yaml

    pose = yaml.safe_load(RESTAURANT_POSES.read_text())["named_poses"]["restaurant_look"]
    camera = _wrist_camera_transform(pose["joints_deg"])
    # Debug staging pose 35 cm back (base -0.90, 0, yaw 180) and centered can (-1.78, 0, 1.015).
    yaw = math.radians(180.0)
    dx, dy = -1.78 - (-0.90), 0.0
    can_in_base = np.array([math.cos(yaw) * dx + math.sin(yaw) * dy,
                            -math.sin(yaw) * dx + math.cos(yaw) * dy,
                            1.015, 1.0])
    can_in_camera = np.linalg.inv(camera) @ can_in_base
    assert can_in_camera[2] > 0.3
    focal_px = 640 / (2 * math.tan(1.165 / 2))
    u = 320 + focal_px * can_in_camera[0] / can_in_camera[2]
    v = 240 + focal_px * can_in_camera[1] / can_in_camera[2]
    can_height_px = 0.23 * focal_px / can_in_camera[2]
    can_width_px = 0.075 * focal_px / can_in_camera[2]
    assert u - can_width_px / 2 > 0 and u + can_width_px / 2 < 640
    assert 150 < u < 490, "The counter can should be away from either wrist-image edge"
    assert v - can_height_px / 2 > 0 and v + can_height_px / 2 < 480
    assert can_height_px >= 150
    assert pose["joints_deg"] == [66.45, 146.56, -64.80, 98.79, -70.32, -39.56]


def test_sim_restaurant_pregrasp_matches_verified_front_screen_pose():
    import yaml

    pose = yaml.safe_load(RESTAURANT_POSES.read_text())["named_poses"]["restaurant_pregrasp"]
    assert pose["joints_deg"] == [21.2, 170.93, -72.96, 59.79, -78.0, -89.05]
    assert pose["source"] == "user_request_2026-11-17_planB_barman_present_pose"


def test_gazebo_joint_controllers_start_at_measured_home_pose():
    arm_config = SIM_DIR.parents[1] / "src/base_athome_perception/config/arm_poses_and_limits.yaml"
    initial = gz_model.initial_joint_positions_from_pose_config(str(arm_config), "home")
    assert initial["joint1"] == pytest.approx(math.radians(84.74))
    assert initial["joint2"] == pytest.approx(math.radians(172.79))
    assert initial["joint3"] == pytest.approx(math.radians(-160.31))
    assert initial["joint7"] == pytest.approx(0.04975)
    assert initial["joint8"] == pytest.approx(-0.04975)

    model = ET.fromstring(gz_model.gazebo_urdf(ROBOT_URDF.read_text(), initial))
    controllers = {plugin.findtext("joint_name"): plugin.findtext("initial_position")
                   for plugin in model.findall(".//plugin") if plugin.findtext("joint_name")}
    assert float(controllers["joint1"]) == pytest.approx(initial["joint1"])
    assert float(controllers["joint7"]) == pytest.approx(initial["joint7"])


def test_sim_counter_look_pose_frames_docked_can_at_safe_camera_depth():
    # The base docks to the same Restaurant standoff; this arm-only posture is
    # used after an initial visual confirmation and must keep the can fully framed.
    look_joints = [66.45, 146.56, -64.80, 98.79, -70.32, -39.56]
    limits = SIM_DIR.parents[1] / "src/base_athome_perception/config/arm_poses_and_limits.yaml"
    assert pose_store.validate_joints(look_joints, path=limits).valid
    camera = _wrist_camera_transform(look_joints)
    can_base = np.array([0.497, 0.0, 1.015, 1.0])
    can_camera = np.linalg.inv(camera) @ can_base
    focal_px = 640 / (2 * math.tan(1.165 / 2))
    u = 320 + focal_px * can_camera[0] / can_camera[2]
    v = 240 + focal_px * can_camera[1] / can_camera[2]
    can_height_px = 0.23 * focal_px / can_camera[2]

    assert can_camera[2] == pytest.approx(0.30, abs=0.01)
    assert abs(u - 320) < 10
    assert 0 < v - can_height_px / 2
    assert v + can_height_px / 2 < 480


def test_wrist_camera_info_bridge_reports_the_simulated_camera_intrinsics():
    model = ET.fromstring(gz_model.gazebo_urdf(ROBOT_URDF.read_text()))
    wrist = next(sensor for sensor in model.findall(".//sensor")
                 if sensor.get("name") == "wrist_camera")
    fov = float(wrist.find("camera/horizontal_fov").text)
    width = int(wrist.find("camera/image/width").text)
    expected_fx = width / (2 * math.tan(fov / 2))

    assert "/sim/gz/wrist/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" in gz_model.bridge_args()
    assert math.isclose(expected_fx, 485.0, abs_tol=1.0)
    assert wrist.get("type") == "camera"
    assert "/sim/gz/wrist/depth_image@sensor_msgs/msg/Image[gz.msgs.Image" not in gz_model.bridge_args()


def test_counter_debug_depth_reader_uses_synced_mask_pixels_in_meters():
    from types import SimpleNamespace
    import importlib.util

    source = SIM_DIR.parents[1] / "simulation/scripts/sim_counter_debug.py"
    spec = importlib.util.spec_from_file_location("sim_counter_debug_depth_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pixels = np.linspace(0.5, 2.0, 16, dtype=np.float32).reshape(4, 4)
    pixels[0, 0] = np.nan
    msg = SimpleNamespace(encoding="32FC1", data=pixels.tobytes(), height=4, width=4)
    depth = module.depth_array_from_image(msg)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask[0, 0] = 0

    assert module.median_depth_in_box(depth, (0, 0, 4, 4), mask) == pytest.approx(1.3)

    millimeters = np.array([[500, 900]], dtype=np.uint16)
    msg16 = SimpleNamespace(encoding="16UC1", data=millimeters.tobytes(), height=1, width=2)
    assert module.depth_array_from_image(msg16)[0, 1] == pytest.approx(0.9)


def test_counter_debug_reads_focal_from_numpy_camera_info_matrix():
    from types import SimpleNamespace
    import importlib.util

    source = SIM_DIR.parents[1] / "simulation/scripts/sim_counter_debug.py"
    spec = importlib.util.spec_from_file_location("sim_counter_debug_camera_info_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = SimpleNamespace(k=np.array([612.5, 0, 320, 0, 612.5, 240, 0, 0, 1]))

    assert module.focal_length_from_camera_info(info) == pytest.approx(612.5)
    assert module.focal_length_from_camera_info(SimpleNamespace(k=np.array([]))) is None


def test_barman_stages_counter_can_in_front_of_left_arm_shoulder():
    source = (SIM_DIR.parents[1] / "simulation/scripts/restaurant_actor.py").read_text()
    assert '"base_link", "arm_base_link"' in source
    assert 'counter_can_pose_world(pose, arm_mount.transform.translation.y)' in source


def test_sim_arm_requires_explicit_sim_and_arm_flags_to_start_enabled():
    assert not sim_arm_enable_requested({})
    assert not sim_arm_enable_requested({"SIM": "1"})
    assert not sim_arm_enable_requested({"ARM": "1"})
    assert sim_arm_enable_requested({"SIM": "1", "ARM": "1"})
    assert sim_arm_enable_requested({"ATHOME_SIM": "1", "ARM": "1"})
    assert not SimArm().enabled
    assert SimArm(enabled=sim_arm_enable_requested({"SIM": "1", "ARM": "1"})).enabled


@pytest.mark.parametrize("state", [
    "DETECTING_ITEM", "PICKING_ITEM", "HANDING_OVER", "RELEASING_ITEM", "STOWING_ARM",
])
def test_restaurant_manipulation_states_request_a_base_hold(state):
    assert restaurant_base_hold_requested(state)


@pytest.mark.parametrize("state", ["ORDERING_BARMAN", "WAITING_BARMAN"])
def test_restaurant_barman_service_states_request_a_base_hold(state):
    assert restaurant_base_hold_requested(state)


@pytest.mark.parametrize("state", ["STARTING", "APPROACHING_CUSTOMER", "DELIVERING_TO_CUSTOMER"])
def test_restaurant_navigation_and_idle_states_do_not_request_a_base_hold(state):
    assert not restaurant_base_hold_requested(state)


def test_barman_gripper_handoff_holds_base_until_measured_close():
    assert restaurant_base_hold_requested("WAITING_BARMAN", {"placement": "gripper"})
    assert restaurant_base_hold_requested("WAITING_BARMAN", {"placement": "tray"})
    assert not restaurant_base_hold_requested("UNDOCKING")


def test_restaurant_base_hold_corrects_angular_drift_with_bounded_command():
    assert restaurant_base_hold_angular_command(0.0, 0.1) == pytest.approx(0.04)
    assert restaurant_base_hold_angular_command(0.1, 0.0) == pytest.approx(-0.04)
    assert restaurant_base_hold_angular_command(-math.pi + 0.01, math.pi - 0.01) == pytest.approx(-0.016)
    assert restaurant_base_hold_angular_command(float("nan"), 0.0) == 0.0


@pytest.mark.parametrize("text", ["robot yes", "yes", "Robot Yes."])
def test_barman_gripper_handoff_requires_explicit_yes(text):
    assert restaurant_handoff_confirmed(text)


@pytest.mark.parametrize("text", ["robot no", "maybe", "water", ""])
def test_barman_gripper_handoff_rejects_nonaffirmative_voice(text):
    assert not restaurant_handoff_confirmed(text)


def test_barman_yes_places_can_at_open_gripper_tcp():
    from types import SimpleNamespace
    from base_athome_sim.sim_robot import SimRobot

    robot = SimRobot.__new__(SimRobot)
    robot.gz = True
    robot.restaurant_handoff_model = "restaurant_counter_can_1"
    robot.restaurant_handoff_at = time.monotonic()
    robot.arm = SimpleNamespace(gripper_cmd_mm=99.5)
    robot.base = SimpleNamespace(x=1.0, y=2.0, yaw=0.0)
    robot._lookup = lambda *_: (None, (0.4, 0.1, 0.8))
    commands = []
    robot._gz_run = lambda command: commands.append(command) or "data: true"
    robot.get_logger = lambda: SimpleNamespace(info=lambda *_: None, warn=lambda *_: None)

    assert robot._stage_restaurant_handoff_can()
    assert robot.restaurant_item_model == "restaurant_counter_can_1"
    assert robot.bag_map == pytest.approx([1.4, 2.1, 0.8])
    assert robot.rigid_object and len(commands) == 1


def test_gazebo_arm_feedback_clock_uses_ros_sim_time():
    from types import SimpleNamespace
    from base_athome_sim.sim_robot import SimRobot

    robot = SimRobot.__new__(SimRobot)
    robot.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=12_500_000_000))

    assert robot._gz_feedback_clock() == pytest.approx(12.5)


def test_gazebo_restaurant_state_clears_goals_and_publishes_zero_during_pick():
    from types import SimpleNamespace
    from base_athome_sim.sim_robot import SimRobot, Twist

    published = []
    robot = SimpleNamespace(
        restaurant_base_hold=False,
        restaurant_base_hold_yaw=None,
        base=SimpleNamespace(goal=(1.0, 2.0, 0.5), x=0.0, y=0.0, yaw=0.4),
        cmd_vel=(0.4, 0.2, 123.0),
        gz_cmd=SimpleNamespace(publish=published.append),
        receptionist_follow_active=False,
        get_logger=lambda: SimpleNamespace(info=lambda *_: None),
    )
    SimRobot._restaurant_base_state(robot, SimpleNamespace(data='{"state":"PICKING_ITEM"}'))
    motion = Twist()
    motion.linear.x, motion.angular.z = 0.5, 0.2
    SimRobot._cmd_vel(robot, motion)
    robot.base.yaw = 0.3  # simulate a small angular drift after the hold was anchored
    SimRobot._gz_tick(robot, 124.0)

    assert robot.restaurant_base_hold
    assert robot.base.goal is None
    assert robot.cmd_vel == (0.0, 0.0, 0.0)
    assert len(published) == 1
    assert published[0].linear.x == 0.0
    assert published[0].angular.z == pytest.approx(0.04)


def test_restaurant_base_hold_logs_start_and_end_odom_pose():
    from types import SimpleNamespace
    from base_athome_sim.sim_robot import SimRobot

    messages = []
    robot = SimpleNamespace(
        restaurant_base_hold=False, restaurant_base_hold_yaw=None,
        restaurant_base_hold_pose=None, restaurant_base_hold_started_state=None,
        base=SimpleNamespace(x=-0.9, y=0.0, yaw=math.pi, goal=None), cmd_vel=(0.0, 0.0, 0.0),
        get_logger=lambda: SimpleNamespace(info=messages.append),
    )
    SimRobot._restaurant_base_state(robot, SimpleNamespace(data='{"state":"ORDERING_BARMAN"}'))
    robot.base.x += 0.01
    SimRobot._restaurant_base_state(robot, SimpleNamespace(data='{"state":"HANDING_OVER"}'))
    assert robot.restaurant_base_hold
    SimRobot._restaurant_base_state(robot, SimpleNamespace(data='{"state":"UNDOCKING"}'))
    assert not robot.restaurant_base_hold
    assert len(messages) == 2
    assert "SIM_RESTAURANT_BASE_HOLD_START" in messages[0]
    assert "odom_xy_yaw=-0.9000,0.0000,3.1416" in messages[0]
    assert "SIM_RESTAURANT_BASE_HOLD_END" in messages[1]
    assert "from=ORDERING_BARMAN to=UNDOCKING" in messages[1]
    assert "drift_m=0.0100" in messages[1]


def test_restaurant_pose_evidence_logs_map_and_odom_during_manipulation():
    from types import SimpleNamespace
    from base_athome_sim.sim_robot import SimRobot

    messages = []
    robot = SimpleNamespace(
        restaurant_base_hold=True,
        restaurant_pose_log_at=0.0,
        restaurant_map_pose=(1.1, 2.2, 0.3),
        base=SimpleNamespace(x=1.0, y=2.0, yaw=0.25),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        get_logger=lambda: SimpleNamespace(info=messages.append),
    )
    SimRobot._restaurant_pose_log(robot)
    assert len(messages) == 1
    assert 'map_xy_yaw=1.1000,2.2000,0.3000' in messages[0]
    assert 'odom_xy_yaw=1.0000,2.0000,0.2500' in messages[0]


def test_restaurant_pose_evidence_uses_map_tf_when_amcl_pose_is_missing():
    from types import SimpleNamespace
    import numpy as np
    from base_athome_sim.sim_robot import SimRobot

    messages = []
    angle = 0.4
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    robot = SimpleNamespace(
        restaurant_base_hold=True,
        restaurant_pose_log_at=0.0,
        restaurant_map_pose=None,
        real_nav=True,
        base=SimpleNamespace(x=1.0, y=2.0, yaw=0.25),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        _lookup=lambda target, source: (rotation, np.array([1.1, 2.2, 0.0])),
        get_logger=lambda: SimpleNamespace(info=messages.append),
    )
    SimRobot._restaurant_pose_log(robot)
    assert 'map_xy_yaw=1.1000,2.2000,0.4000' in messages[0]
    assert 'odom_xy_yaw=1.0000,2.0000,0.2500' in messages[0]


def test_manipulation_lab_monitors_every_collision_without_changing_geometry():
    import importlib.util
    spec = importlib.util.spec_from_file_location('manipulation_dock_lab',
                                                SIM_DIR.parent / 'scripts/manipulation_dock_lab.py')
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    source = '<sdf><model name="robot"><link name="a"><collision name="a1"><geometry><box><size>1 2 3</size></box></geometry></collision><collision name="a2"/></link><link name="b"><collision name="b1"/></link></model></sdf>'
    root = ET.fromstring(lab.with_contact_sensors(source))
    for link in root.findall('./model/link'):
        collisions = {c.get('name') for c in link.findall('collision')}
        assert {s.findtext('contact/collision') for s in link.findall('sensor')} == collisions
        assert all(s.findtext('topic') == '/restaurant/contacts' for s in link.findall('sensor'))
        assert all(s.findtext('contact/topic') == '/restaurant/contacts' for s in link.findall('sensor'))
        for sensor in link.findall('sensor'):
            link.remove(sensor)
    assert ET.tostring(root) == ET.tostring(ET.fromstring(source))


def test_make_customer_raised_hand_keypoints():
    """make_customer debe situar la muneca por encima de la cabeza y los hombros."""
    # Mano derecha levantada (de pie)
    cust_r = sim_scene.make_customer([2.0, 1.0], facing_xy=[0.0, 0.0], hand="right", seated=False)
    kp_r = cust_r["keypoints"]
    nose_z = kp_r[0][2]
    r_sh_z = kp_r[6][2]
    r_wr_z = kp_r[10][2]
    l_wr_z = kp_r[9][2]

    assert r_wr_z > nose_z, f"Muneca derecha ({r_wr_z}) debe estar sobre la cabeza ({nose_z})"
    assert r_wr_z > r_sh_z, f"Muneca derecha ({r_wr_z}) debe estar sobre el hombro ({r_sh_z})"
    assert l_wr_z < r_sh_z, "Brazo izquierdo debe estar abajo"

    # Mano izquierda levantada (sentado)
    cust_l = sim_scene.make_customer([2.5, -1.0], facing_xy=[0.0, 0.0], hand="left", seated=True)
    kp_l = cust_l["keypoints"]
    assert kp_l[9][2] > kp_l[0][2], "Muneca izquierda debe estar sobre la cabeza"
    assert kp_l[10][2] < kp_l[6][2], "Brazo derecho debe estar abajo"

    # Ambas manos levantadas
    cust_both = sim_scene.make_customer([3.0, 0.0], facing_xy=[0.0, 0.0], hand="both")
    kp_b = cust_both["keypoints"]
    assert kp_b[9][2] > kp_b[0][2]
    assert kp_b[10][2] > kp_b[0][2]


def test_customer_projection_classifies_correctly():
    """Al proyectar un cliente visto por la camara frontal, classify_raised_hand lo reconoce."""
    from base_athome_perception import bag_vision as bv

    cust = sim_scene.make_customer([2.5, 0.0], facing_xy=[0.0, 0.0], hand="right")
    # Camara en (0, 0, 1.0) mirando hacia +X
    T = np.array([0.0, 0.0, 1.0])
    # Rotacion estandar camera optical frame hacia +X mundo:
    # cam_x -> -Y mundo, cam_y -> -Z mundo, cam_z -> +X mundo
    R = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    K = (615.0, 615.0, 320.0, 240.0)

    projected = bv.project_people([cust], T, R, K, width=640, height=480)
    assert len(projected) == 1, "El cliente debe ser visible en el campo de vision"
    box, kp, conf = projected[0]

    cls = classify_raised_hand(kp, conf)
    assert cls.is_raised is True
    assert cls.state == RaisedHandState.RIGHT


def test_restaurant_visible_gesture_sequence_tracks_dialogue_and_service():
    from base_athome_sim.restaurant_humans import ParticipantGestures, gesture_model_sdf
    gestures = ParticipantGestures()
    assert gestures.observe({'state': 'FINDING_BARMAN'}) == frozenset({2})
    assert 'standing_right.dae' in gesture_model_sdf(2, 2 in gestures.observe({'state': 'FINDING_BARMAN'}))
    assert 'seated_down.dae' in gesture_model_sdf(0, False)

    # Robbie starts searching; only the barman signals. Both diners wave only
    # after the mission announces customer detection.
    raised = gestures.observe({'state': 'DETECTING_CUSTOMERS'})
    assert raised == frozenset({0, 1})
    assert 'seated_right.dae' in gesture_model_sdf(0, 0 in raised)
    assert 'seated_left.dae' in gesture_model_sdf(1, 1 in raised)

    # The first table is mapped to customer 0 and lowered when served advances.
    gestures.observe({'state': 'APPROACHING_CUSTOMER', 'customer_idx': 0,
                      'info': {'customer_xy': [2.0, 2.8]}})
    raised = gestures.observe({'state': 'APPROACHING_CUSTOMER', 'customer_idx': 0,
                              'info': {'customer_xy': [2.0, 2.8]}})
    assert raised == frozenset({0, 1})
    raised = gestures.observe({'state': 'ENROLLING_FACE', 'customer_idx': 0, 'served': 0})
    assert raised == frozenset({0, 1})
    raised = gestures.observe({'state': 'DETECTING_CUSTOMERS', 'customer_idx': 0, 'served': 0})
    assert raised == frozenset({0, 1})
    gestures.observe({'state': 'APPROACHING_CUSTOMER', 'customer_idx': 0,
                      'info': {'customer_xy': [2.0, 2.8]}})
    raised = gestures.observe({'state': 'RETURNING_TO_BAR', 'customer_idx': 0, 'served': 1})
    assert raised == frozenset({1})
    raised = gestures.observe({'state': 'DETECTING_CUSTOMERS', 'customer_idx': 0, 'served': 1})
    assert raised == frozenset({1})
    # The remaining seated customer is visible and still classified by the
    # existing geometric detector; the missed scan was gesture state only.
    from base_athome_perception import bag_vision as bv
    remaining = sim_scene.make_customer([2.5, 0.0], facing_xy=[0.0, 0.0], hand='right', seated=True)
    T = np.array([0.0, 0.0, 1.0])
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    projected = bv.project_people([remaining], T, R, (615.0, 615.0, 320.0, 240.0), 640, 480)
    assert len(projected) == 1
    assert classify_raised_hand(projected[0][1], projected[0][2]).is_raised
    assert 'seated_down.dae' in gesture_model_sdf(0, 0 in raised)
    assert 'seated_left.dae' in gesture_model_sdf(1, 1 in raised)


def test_person_faces_with_customer():
    """sim_scene.person_faces genera caras validas para clientes de pie y sentados."""
    cust_stand = sim_scene.make_customer([2.0, 1.0], hand="right", seated=False)
    faces_stand = sim_scene.person_faces(cust_stand)
    assert len(faces_stand) > 10, "Debe generar caras de cuerpo, cabeza y brazos"

    cust_seat = sim_scene.make_customer([2.0, 1.0], hand="left", seated=True)
    faces_seat = sim_scene.person_faces(cust_seat)
    assert len(faces_seat) > 10


def test_customer_sdf_valid_with_gz():
    """gz_model.customer_sdf genera un SDF valido con cuerpo y brazo levantado."""
    sdf_r = gz_model.customer_sdf("cliente_mesa1", [2.5, 1.5], hand="right", seated=False)
    assert "cliente_mesa1" in sdf_r
    assert "v_cuerpo" in sdf_r
    assert "v_brazo_d" in sdf_r


@pytest.mark.parametrize("env_person, bag, customers, expected", [
    (False, [-1.8, 0.0, 1.015], [{"xy": [2.0, 2.8]}], False),
    (False, [1.0, 1.0, 0.3], [], True),
    (True, [-1.8, 0.0, 1.015], [{"xy": [2.0, 2.8]}], True),
    (False, None, [], False),
])
def test_scene_person_only_when_requested_or_bag_mission(monkeypatch, env_person, bag, customers, expected):
    from types import SimpleNamespace
    from base_athome_sim import sim_robot
    monkeypatch.delenv("ATHOME_SIM_PERSON", raising=False)
    if env_person:
        monkeypatch.setenv("ATHOME_SIM_PERSON", "0.5,1.8")
    fake = SimpleNamespace(bag_map=bag, customers_xy=customers)
    assert sim_robot.SimRobot._scene_person_enabled(fake) is expected


def test_restaurant_counter_can_has_requested_mass_inertia_contact_and_friction():
    root = ET.fromstring(gz_model.restaurant_can_sdf(
        (-1.8, 0.0, 1.015), name='restaurant_counter_can_1'))
    model = root.find('./model')
    assert model.get('name') == 'restaurant_counter_can_1'
    assert float(model.findtext('./link/inertial/mass')) == pytest.approx(0.20)
    inertia = model.find('./link/inertial/inertia')
    radius, height, mass = 0.0375, 0.23, 0.20
    assert float(inertia.findtext('ixx')) == pytest.approx(mass * (3 * radius**2 + height**2) / 12)
    assert float(inertia.findtext('iyy')) == pytest.approx(float(inertia.findtext('ixx')))
    assert float(inertia.findtext('izz')) == pytest.approx(mass * radius**2 / 2)
    collision = model.find('./link/collision')
    assert float(collision.findtext('./geometry/cylinder/radius')) == pytest.approx(radius)
    assert float(collision.findtext('./geometry/cylinder/length')) == pytest.approx(height)
    assert float(collision.findtext('./surface/friction/ode/mu')) == pytest.approx(1.0)
    assert float(collision.findtext('./surface/friction/ode/mu2')) == pytest.approx(1.0)
    diffuse = model.findtext('./link/visual/material/diffuse')
    assert diffuse == '0.18 0.48 0.32 1'
    assert collision.find('./surface/friction/torsional/ode') is not None
    assert float(collision.findtext('./surface/contact/ode/kp')) > 0
    pose_topic = gz_model.restaurant_item_odometry_topic('restaurant_counter_can_1')
    assert model.findtext('./plugin/odom_topic') == pose_topic
    assert f'{pose_topic}@nav_msgs/msg/Odometry[gz.msgs.Odometry' in gz_model.bridge_args()


def test_restaurant_startup_has_no_bar_object_and_sim_grasp_tracks_actor_pose():
    script = (Path(__file__).resolve().parents[3] / 'src/base_athome_skills/scripts/restaurant.sh').read_text()
    restaurant_block = script.split('export ATHOME_SIM_WORLD=restaurant.sdf', 1)[1].split(
        'export ATHOME_SIM_CUSTOMERS=', 1)[0]
    assert 'unset ATHOME_SIM_TABLE_OBJECT' in restaurant_block
    assert 'export ATHOME_SIM_TABLE_OBJECT=' not in script

    from types import SimpleNamespace as NS
    from base_athome_sim.sim_robot import SimRobot
    robot = NS(gz=True, restaurant_item_poses={
        'restaurant_counter_can_1': (time.monotonic(), (-1.72, 0.05, 1.015)),
    }, lock=threading.Lock(), base=NS(x=-1.1, y=0.0, yaw=math.pi),
       restaurant_item_model=None, bag_map=None, rigid_object=False, restaurant_item_contact=False)
    SimRobot._sync_restaurant_item_target(robot, (0.6, 0.05, 1.015))
    assert robot.restaurant_item_model == 'restaurant_counter_can_1'
    assert robot.bag_map == [-1.72, 0.05, 1.015]
    assert robot.rigid_object is True
    robot.restaurant_item_poses['restaurant_counter_can_1'] = (
        time.monotonic() - 3.0, (-1.72, 0.05, 1.015))
    SimRobot._sync_restaurant_item_target(robot)
    assert robot.restaurant_item_model is None and robot.bag_map is None


def test_late_counter_can_is_detached_until_simulated_grasp():
    from types import SimpleNamespace as NS
    from base_athome_sim.sim_robot import SimRobot

    detached = []
    robot = NS(
        obj_attached=False,
        gz_ready_t=time.time() - 3600.0,
        pub_restaurant_can_detach={"restaurant_counter_can_1": NS(
            publish=lambda msg: detached.append(msg))},
        lock=threading.Lock(),
        restaurant_item_poses={},
        get_logger=lambda: NS(info=lambda _msg: None),
    )
    msg = NS(pose=NS(pose=NS(position=NS(x=-1.78, y=0.0, z=1.015))))

    # The actor spawns the can long after gz_ready_t (startup + 15 s).
    SimRobot._restaurant_item_odom(robot, "restaurant_counter_can_1", msg)
    assert len(detached) == 1
    assert robot.restaurant_item_poses["restaurant_counter_can_1"][1] == (-1.78, 0.0, 1.015)

    # Once the grasp logic owns it, odometry must not detach the grasp joint.
    robot.obj_attached = True
    SimRobot._restaurant_item_odom(robot, "restaurant_counter_can_1", msg)
    assert len(detached) == 1


def test_restaurant_grasp_does_not_attach_when_gripper_open_or_at_pregrasp():
    from types import SimpleNamespace as NS
    from base_athome_sim.sim_robot import SimRobot
    import numpy as np

    published_attach = []
    robot = NS(
        gz=True,
        lock=threading.Lock(),
        base=NS(x=-1.15, y=0.0, yaw=math.pi),
        obj=[0.65, 0.0, 1.018],
        obj_attached=False,
        gripper_was_closed=False,
        grasp_slips=False,
        bag_in_hand=False,
        bag_map=[-1.80, 0.0, 1.018],
        rigid_object=True,
        restaurant_item_model='restaurant_counter_can_1',
        restaurant_item_contact=False,
        restaurant_detached_pending_release=False,
        pub_restaurant_can_attach={'restaurant_counter_can_1': NS(publish=lambda msg: published_attach.append(True))},
        pub_world=NS(publish=lambda msg: None),
        get_parameter=lambda name: NS(value=0.075),
        get_logger=lambda: NS(info=lambda msg: None),
        _physical_grasp=lambda: False,
        _sync_restaurant_item_target=lambda tcp: None,
    )
    # Escenario 1: Brazo en pregrasp (d = 0.098 m > 0.045 m) con pinza abierta (105 mm) -> NO acopla
    robot.arm = NS(gripper_mm=105.0, gripper_cmd_mm=105.0, gripper_target_mm=105.0,
                   grip_block_mm=None)
    robot._lookup = lambda f1, f2: (None, np.array([0.552, 0.0, 1.015]))
    SimRobot._update_object(robot, None)
    assert not robot.obj_attached
    assert len(published_attach) == 0

    # Escenario 2: Brazo avanza sobre la lata (d = 0.015 m) pero pinza sigue abierta -> NO acopla
    robot._lookup = lambda f1, f2: (None, np.array([0.640, 0.0, 1.018]))
    SimRobot._update_object(robot, None)
    assert not robot.obj_attached
    assert len(published_attach) == 0

    # Escenario 3: Pinza recibe orden de cierre alrededor de la lata (cmd = 50.0 mm <= 77 mm) -> ACOPLA
    robot.arm.gripper_cmd_mm = 50.0
    robot.arm.gripper_target_mm = 50.0
    robot.arm.gripper_mm = 75.0
    SimRobot._update_object(robot, None)
    assert robot.obj_attached is True
    assert len(published_attach) == 1


def test_attached_restaurant_can_blocks_sim_feedback_at_object_diameter():
    from base_athome_sim.measured_arm_feedback import apply_attached_object_gripper_width
    measured = {"feedback_ok": True, "gripper_mm": 1.7, "joints_deg": [0.0] * 6}
    attached = apply_attached_object_gripper_width(measured, attached=True, diameter_mm=75.0)
    assert attached["gripper_mm"] == pytest.approx(75.0)
    assert attached["joints_deg"] == measured["joints_deg"]
    assert measured["gripper_mm"] == pytest.approx(1.7)
    assert apply_attached_object_gripper_width(measured, attached=False, diameter_mm=75.0) is measured
    stale = {**measured, "feedback_ok": False}
    assert apply_attached_object_gripper_width(stale, attached=True, diameter_mm=75.0) is stale


def test_restaurant_attach_and_detach_callbacks_track_simulated_can(monkeypatch):
    from types import SimpleNamespace as NS
    from base_athome_sim import sim_robot
    monkeypatch.setattr(sim_robot, "GZ_WORLD", "restaurant")
    robot = NS(gz=True, restaurant_item_model=None, obj_attached=False,
               lock=threading.Lock(),
               bag_map=[-1.8, 0.0, 1.015], rigid_object=False, restaurant_item_contact=False,
               restaurant_gripper_attach_reported=False,
               restaurant_item_poses={}, gripper_stall=sim_robot.GripperStall(),
               arm=NS(grip_block_mm=70.0))
    sim_robot.SimRobot._restaurant_can_attached(robot, "restaurant_counter_can_1")
    assert robot.obj_attached and robot.rigid_object and robot.restaurant_item_contact
    assert robot.bag_map == [-1.8, 0.0, 1.015]
    sim_robot.SimRobot._restaurant_can_detached(robot, "restaurant_counter_can_2")
    assert robot.obj_attached  # unrelated detach cannot release this can
    sim_robot.SimRobot._restaurant_can_detached(robot, "restaurant_counter_can_1")
    assert not robot.obj_attached and not robot.restaurant_item_contact
    assert robot.restaurant_item_model is None and robot.bag_map is None and not robot.rigid_object
    assert robot.arm.grip_block_mm is None  # detach releases the simulated finger stop


def test_restaurant_tray_item_is_dynamic_and_has_friction_for_nav2_acceleration():
    root = ET.fromstring(gz_model.restaurant_tray_item_sdf((0.155, 0.0, 1.4845)))
    model = root.find('./model')
    assert model.get('name') == gz_model.RESTAURANT_TRAY_ITEM
    assert float(model.findtext('./link/inertial/mass')) == pytest.approx(0.15)
    collision = model.find('./link/collision')
    assert float(collision.findtext('./surface/friction/ode/mu')) == pytest.approx(1.0)
    assert float(collision.findtext('./surface/friction/ode/mu2')) == pytest.approx(1.0)
    assert float(collision.findtext('./surface/friction/torsional/ode/patch_radius')) > 0
    assert float(collision.findtext('./surface/contact/ode/kp')) > 0


def test_restaurant_tray_is_parametrized_30cm_with_physical_floor_and_rim():
    urdf = gz_model.gazebo_urdf('<robot name="test"><link name="mounting_cube_link"/></robot>')
    root = ET.fromstring(urdf)
    tray = root.find(f'./link[@name="{gz_model.RESTAURANT_TRAY_LINK}"]')
    assert tray is not None
    assert float(tray.find('./inertial/mass').get('value')) == pytest.approx(0.15)
    floor = tray.find('./collision[@name="floor"]/geometry/box').get('size').split()
    assert [float(v) for v in floor[:2]] == pytest.approx([gz_model.RESTAURANT_TRAY_SIDE_M] * 2)
    assert {c.get('name') for c in tray.findall('./collision') if c.get('name', '').startswith('rim_')} == {
        'rim_front', 'rim_back', 'rim_left', 'rim_right'}
    assert all(c.find('./geometry/box').get('size') for c in tray.findall('./collision'))
    joint = root.find('./joint[@name="restaurant_tray_fixed_joint"]')
    assert joint.find('parent').get('link') == 'mounting_cube_link'
    assert joint.find('child').get('link') == gz_model.RESTAURANT_TRAY_LINK


def test_gazebo_gripper_fingers_have_contact_friction_for_dynamic_can():
    urdf = gz_model.gazebo_urdf('''<robot name="test">
      <link name="link7"><collision name="mesh"><geometry><mesh filename="finger.stl"/></geometry></collision></link>
      <link name="link8"><collision name="mesh"><geometry><mesh filename="finger.stl"/></geometry></collision></link>
    </robot>''')
    root = ET.fromstring(urdf)
    for finger in ('link7', 'link8'):
        gazebo = root.find(f'./gazebo[@reference="{finger}"]')
        assert float(gazebo.findtext('mu1')) >= 1.0
        assert float(gazebo.findtext('mu2')) >= 1.0
        assert float(gazebo.findtext('kp')) > 0
        collision = root.find(f'./link[@name="{finger}"]/collision')
        assert collision.get('name') == 'finger_pad'
        assert collision.find('./geometry/box').get('size') == '0.056 0.0765 0.0265'
        assert '<maxVel>' in ET.tostring(gazebo, encoding='unicode')
        assert '<minDepth>' in ET.tostring(gazebo, encoding='unicode')
    assert any(arg.startswith('/sim/gz/joint_states@') for arg in gz_model.bridge_args())


def test_gazebo_gripper_opening_matches_urdf_and_can_diameter():
    assert gz_model.GRIPPER_HALF_MAX_M == pytest.approx(0.0525)
    assert 2 * gz_model.GRIPPER_HALF_MAX_M > (2 * 0.0375 + 0.02)


def test_restaurant_scene_publisher_skips_disabled_bag_person(monkeypatch):
    from types import SimpleNamespace
    from base_athome_sim.sim_robot import SimRobot
    monkeypatch.delenv("ATHOME_SIM_PERSON", raising=False)
    robot = SimpleNamespace(bag_map=[-1.8, 0.0, 1.015], customers_xy=[{"xy": [2.0, 2.8]}],
                            _scene_person_enabled=lambda: False,
                            pub_world=SimpleNamespace(publish=lambda _: pytest.fail("no scene markers expected")))
    SimRobot._publish_scene(robot, None)


def test_customer_sdf_supports_both_hands_and_validates_with_gz():
    sdf_both = gz_model.customer_sdf("cliente_mesa2", [3.5, 1.5], hand="both", seated=True)
    assert "v_brazo_d" in sdf_both
    assert "v_brazo_i" in sdf_both
    with tempfile.NamedTemporaryFile("w", suffix=".sdf") as f:
        f.write(sdf_both)
        f.flush()
        res = subprocess.run(["gz", "sdf", "-k", f.name], capture_output=True, text=True)
        assert res.returncode == 0, f"Error validando SDF: {res.stderr}"


def test_restaurant_sdf_valid_with_gz():
    """El mundo restaurant.sdf debe existir y ser valido para Gazebo Harmonic."""
    assert RESTAURANT_SDF.exists(), f"No existe {RESTAURANT_SDF}"
    res = subprocess.run(["gz", "sdf", "-k", str(RESTAURANT_SDF)], capture_output=True, text=True)
    assert res.returncode == 0, f"gz sdf -k fallo en restaurant.sdf: {res.stderr}\n{res.stdout}"
    assert "Valid" in res.stdout or res.returncode == 0


def test_restaurant_sdf_contains_required_furniture():
    """El salon debe incluir paredes, barra de cocina y 3 mesas."""
    root = ET.parse(RESTAURANT_SDF).getroot()
    models = {m.attrib.get("name"): m for m in root.iter("model")}

    required_models = ["piso", "paredes", "barra", "mesa_1", "mesa_2", "mesa_3"]
    for name in required_models:
        assert name in models, f"Falta el modelo '{name}' en restaurant.sdf"
        m = models[name]
        assert m.findtext("static") == "true", f"El modelo '{name}' debe ser static=true"


def test_restaurant_sdf_header_documentation():
    """El encabezado del archivo debe documentar donde arranca el robot y clientes sugeridos."""
    with open(RESTAURANT_SDF, encoding="utf-8") as f:
        header = f.read(2048)

    assert "x = 0.0, y = 0.0" in header, "Debe documentar pose de arranque del robot (start_pose de sim_robot)"
    assert "barra" in header.lower(), "Debe mencionar la barra"
    assert "ATHOME_SIM_CUSTOMERS" in header, "Debe documentar la variable ATHOME_SIM_CUSTOMERS"
    assert "mesa" in header.lower(), "Debe documentar mesas o clientes sugeridos"


def test_athome_sim_customers_env_parsing(monkeypatch):
    """Verifica que sim_robot parsee correctamente ATHOME_SIM_CUSTOMERS con ATHOME_SIM_PERSON."""
    monkeypatch.setenv("ATHOME_SIM_CUSTOMERS", "1.5,2.3;3.5,1.5,left;2.5,-1.5,right,seated")
    monkeypatch.setenv("ATHOME_SIM_PERSON", "0.5,1.8")

    raw_cust = os.environ.get("ATHOME_SIM_CUSTOMERS", "").strip()
    customers = []
    for item in raw_cust.split(";"):
        if item.strip():
            parts = [v.strip() for v in item.split(",") if v.strip()]
            cx, cy = float(parts[0]), float(parts[1])
            hand = parts[2] if len(parts) >= 3 else "right"
            seated = (parts[3].lower() == "seated") if len(parts) >= 4 else False
            customers.append({"xy": [cx, cy], "hand": hand, "seated": seated})

    assert len(customers) == 3
    assert customers[0] == {"xy": [1.5, 2.3], "hand": "right", "seated": False}
    assert customers[1] == {"xy": [3.5, 1.5], "hand": "left", "seated": False}
    assert customers[2] == {"xy": [2.5, -1.5], "hand": "right", "seated": True}

    # Ambos conviven
    assert os.environ.get("ATHOME_SIM_PERSON") == "0.5,1.8"


def test_gazebo_publishes_customers_without_a_bag():
    """Restaurant no tiene bolsa: igual hay que publicar /sim/people y crear los clientes en Gazebo."""
    import inspect
    from base_athome_sim import sim_robot
    src = inspect.getsource(sim_robot.SimRobot._setup_gazebo) if hasattr(sim_robot, "SimRobot") else inspect.getsource(sim_robot)
    assert "elif self.customers_xy" in src


def test_simulated_table_retreat_accepts_lab_domains_but_never_real_or_default_domain():
    """Restaurant's sim-only retreat must work on parallel lab domains."""
    import ast

    adapter = Path(__file__).resolve().parents[3] / "src" / "base_athome_skills" / "base_athome_skills" / "restaurant_node.py"
    module = ast.parse(adapter.read_text(encoding="utf-8"))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_sim_table_retreat_allowed")
    isolated = ast.Module(body=[function], type_ignores=[])
    exec(compile(isolated, str(adapter), "exec"), globals())

    assert _sim_table_retreat_allowed({"ATHOME_SIM": "1", "ROS_DOMAIN_ID": "66"})
    assert _sim_table_retreat_allowed({"ATHOME_SIM": "1", "ROS_DOMAIN_ID": "42"})
    assert not _sim_table_retreat_allowed({"ATHOME_SIM": "1", "ROS_DOMAIN_ID": "10"})
    assert not _sim_table_retreat_allowed({"ATHOME_SIM": "1", "ROS_DOMAIN_ID": "0"})
    assert not _sim_table_retreat_allowed({"ROS_DOMAIN_ID": "66"})


@pytest.mark.parametrize("domain", ["0", "10", "233", "not-a-domain", None])
def test_restaurant_lab_domain_rejects_unavailable_domain_ids(domain):
    with pytest.raises(ValueError):
        validate_lab_domain(domain)


@pytest.mark.parametrize("domain", [1, 42, 66, 77, 232])
def test_restaurant_lab_domain_accepts_isolated_lab_ids(domain):
    assert validate_lab_domain(str(domain)) == domain


@pytest.mark.parametrize("env_person, bag, customers, expected", [
    (False, [-1.8, 0.0, 1.015], [{"xy": [2.0, 2.8]}], False),   # Restaurant: objeto en la barra + clientes -> sin prisma
    (False, [1.0, 1.0, 0.3], [], True),                          # mision de la bolsa: la persona apunta a la bolsa
    (True, [-1.8, 0.0, 1.015], [{"xy": [2.0, 2.8]}], True),      # pedida explicitamente
    (False, None, [], False),
])
def test_scene_person_only_when_requested_or_bag_mission(monkeypatch, env_person, bag, customers, expected):
    from types import SimpleNamespace
    from base_athome_sim import sim_robot
    monkeypatch.delenv("ATHOME_SIM_PERSON", raising=False)
    if env_person:
        monkeypatch.setenv("ATHOME_SIM_PERSON", "0.5,1.8")
    fake = SimpleNamespace(bag_map=bag, customers_xy=customers)
    assert sim_robot.SimRobot._scene_person_enabled(fake) is expected


def test_sim_arm_motion_can_be_held_still_while_arm_stays_enabled_for_docking():
    from base_athome_sim.sim_core import sim_arm_enable_requested
    base = {'SIM': '1', 'ARM': '1'}
    assert sim_arm_enable_requested(base)
    assert not sim_arm_enable_requested({**base, 'ATHOME_SIM_ARM_MOTION': '0'})
    assert not sim_arm_enable_requested({'ARM': '1'})
