"""Modelos de la simulacion (sin ROS)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_athome_sim.sim_core import (SimArm, SimBase, SimMicrophone, SimPanTilt,
                                      sim_arm_start_state,
                                      sim_arm_start_state_matches,
                                      gripper_stall_enabled_after_detach,
                                      map_odom_relay_stamp,
                                      restaurant_gripper_contact_block)  # noqa: E402


def test_arm_follows_joint_cmd_only_when_enabled():
    a = SimArm([0] * 6)
    a.joint_cmd([10] * 6, 1.0, now=0.0)
    a.update(now=1.0)
    assert a.q == [0] * 6  # deshabilitado: no se mueve
    a.set_enable(True)
    a.joint_cmd([10] * 6, 1.0, now=0.0)
    a.update(now=0.5)
    assert 0 < a.q[0] < 10
    a.update(now=1.1)
    assert a.q == pytest.approx([10] * 6) and not a.moving


def test_sim_arm_initial_command_matches_gazebo_home_joint_state():
    import math

    initial = {f"joint{i}": math.radians(value) for i, value in enumerate(
        (84.74, 172.79, -160.31, -20.99, -10.47, 1.49), 1)}
    initial.update(joint7=0.04975, joint8=-0.04975)
    joints, width = sim_arm_start_state(initial)
    arm = SimArm(enabled=True)
    arm.q = joints
    arm.gripper_mm = arm.gripper_cmd_mm = arm.gripper_target_mm = width
    assert arm.q == pytest.approx([84.74, 172.79, -160.31, -20.99, -10.47, 1.49])
    assert (arm.gripper_mm, arm.gripper_cmd_mm, arm.gripper_target_mm) == pytest.approx((99.5,)*3)


def test_sim_arm_start_feedback_must_reach_configured_home_before_adoption():
    expected = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.05, -0.05]
    stale = [0.06, 0.06, -0.06, -0.06, -0.06, 0.026, 0.019, -0.020]
    assert not sim_arm_start_state_matches(stale, expected)
    settled = [value + (0.01 if index % 2 else -0.01)
               for index, value in enumerate(expected)]
    assert sim_arm_start_state_matches(settled, expected)
    assert not sim_arm_start_state_matches(stale[:-1], expected)


def test_arm_trajectory_linear():
    a = SimArm([0] * 6)
    a.set_enable(True)
    a.trajectory([1.0, 2.0], [[10] * 6, [20] * 6], now=0.0)
    a.update(now=1.5)
    assert a.q[0] == pytest.approx(15.0)


def test_pan_tilt_speed_limited_and_needs_torque():
    p = SimPanTilt(speed_dps=10)
    p.target = (30, -10)
    p.update(1.0)
    assert p.pan == 0
    p.torque = True
    p.update(1.0)
    assert p.pan == 10 and p.tilt == -10


def test_base_reaches_goal():
    b = SimBase(speed=1.0)
    b.set_goal(1.0, 0.0, 0.5)
    assert not b.update(0.5)
    assert b.update(0.6) and b.x == 1.0 and b.yaw == 0.5


def test_microphone_listens_by_request_with_result_always():
    m = SimMicrophone(start_timeout=8, end_silence=1.0, asr_s=1.0)
    assert not m.hear("lost before request", now=0.0)          # sin pedido: se pierde
    m.request("a", now=0.0)
    m.muted = True
    assert not m.hear("robot talking", now=0.5)               # con mute: se pierde
    m.muted = False
    assert m.hear("My name is Robin", now=2.0)
    assert m.due(now=3.0) == []
    rid, st, text = m.due(now=6.0)[0]
    assert (rid, st, text) == ("a", "ok", "My name is Robin")
    m.request("b", now=10.0)
    assert m.due(now=17.0) == []
    assert m.due(now=18.1) == [("b", "no_speech", "")]           # nadie hablo: resultado explicito
    m.request("c", now=20.0)
    assert m.request("d", now=21.0) == "c"                       # el nuevo reemplaza al anterior

def test_gz_urdf_has_plugins_and_bridges():
    from base_athome_sim import gz_model
    urdf = '<robot name="r"><link name="base_link"/></robot>'
    out = gz_model.gazebo_urdf(urdf)
    assert out.endswith("</robot>")
    assert "VelocityControl" in out and "gpu_lidar" in out and "rgbd_camera" in out
    assert 'name="restaurant_rear_laser"' in out
    assert "/sim/gz/rear_scan" in out and "rpy=\"0 0 3.14159265359\"" in out
    assert out.count("JointPositionController") == len(gz_model.POSITION_JOINTS)
    assert any(a.startswith("/sim/gz/joint1/cmd_pos@") for a in gz_model.bridge_args())
    assert "/sim/gz/rear_scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan" in gz_model.bridge_args()


def test_restaurant_pick_prop_is_a_self_authored_bottle_shape():
    import xml.etree.ElementTree as ET
    from base_athome_sim import gz_model
    sdf = ET.fromstring(gz_model.table_object_sdf((-1.8, 0.0, 1.015)))
    assert sdf.find('.//collision[@name="body_collision"]/geometry/cylinder') is not None
    assert sdf.find('.//collision[@name="neck_collision"]/geometry/cylinder') is not None
    assert 'OpenRobotics' not in ET.tostring(sdf, encoding='unicode')


def test_gripper_stops_at_object_width_when_closing():
    from base_athome_sim.sim_core import SimArm
    arm = SimArm()
    arm.set_enable(True)
    arm.gripper_mm = 100.0
    arm.grip_block_mm = 65.0
    arm.gripper(10.0, now=0.0)
    arm.update(now=5.0)
    assert arm.gripper_mm == 65.0          # el objeto impide cerrar mas
    arm.gripper(100.0, now=5.0)
    arm.update(now=10.0)
    assert arm.gripper_mm == 100.0         # abrir no esta bloqueado


def test_gripper_command_keeps_closing_while_feedback_stops_at_contact():
    from base_athome_sim.sim_core import SimArm
    arm = SimArm()
    arm.set_enable(True)
    arm.gripper_mm = arm.gripper_cmd_mm = 105.0
    arm.grip_block_mm = 75.0
    arm.gripper(50.0, now=0.0)
    arm.update(now=2.0)
    assert arm.gripper_mm == 75.0
    assert arm.gripper_cmd_mm == 50.0
    assert arm.gripper_target_mm == 50.0


def test_gripper_stall_uses_only_fresh_measured_contact():
    from base_athome_sim.sim_core import GripperStall
    stall = GripperStall()
    assert stall.block_mm(50.0, now=0.0) is None
    for i in range(10):
        stall.measure(105.0 - 5.0 * i, now=0.1 * i)
        assert stall.block_mm(50.0, now=0.1 * i) is None
    for i in range(10):
        stall.measure(74.6 + 0.1 * (i % 2), now=1.0 + 0.1 * i)
    assert stall.block_mm(50.0, now=1.9) == pytest.approx(74.7, abs=0.2)
    assert stall.block_mm(74.0, now=1.9) is None
    assert stall.block_mm(50.0, now=3.0) is None


def test_detaching_restaurant_can_clears_stale_stall_until_open_command():
    from base_athome_sim.sim_core import GripperStall

    stall = GripperStall()
    stall.measure(75.0, now=1.0)
    assert stall.block_mm(50.0, now=1.5) == pytest.approx(75.0)
    stall.clear()
    assert stall.block_mm(50.0, now=1.5) is None
    assert not gripper_stall_enabled_after_detach(True, 50.0)
    assert gripper_stall_enabled_after_detach(True, 95.0)
    assert gripper_stall_enabled_after_detach(False, 50.0)


def test_restaurant_carry_can_handover_take_then_release_clears_contact_block():
    # Carrying the attached can blocks closing; actor detach at handover must
    # suppress that same contact while the gripper remains closed and on release.
    block, pending = restaurant_gripper_contact_block(
        50.0, 65.0, item_present=True, detach_pending_release=False)
    assert block == pytest.approx(65.0) and not pending

    block, pending = restaurant_gripper_contact_block(
        50.0, 65.0, item_present=False, detach_pending_release=True)
    assert block is None and pending

    block, pending = restaurant_gripper_contact_block(
        105.0, 65.0, item_present=False, detach_pending_release=pending)
    assert block is None and not pending


def test_simulated_map_odom_relay_is_fresh_and_future_stamp_is_bounded():
    assert map_odom_relay_stamp(10.0, 9.95) == pytest.approx(10.15)
    assert map_odom_relay_stamp(10.0, 10.20) == pytest.approx(10.15)
    assert map_odom_relay_stamp(10.0, 10.26) is None
    assert map_odom_relay_stamp(10.0, 9.4) is None
    assert map_odom_relay_stamp(10.0, 9.95, lead_s=1.0, max_lead_s=0.15) == pytest.approx(10.15)

    source = (Path(__file__).resolve().parents[1] / 'base_athome_sim/sim_robot.py').read_text()
    assert 'self.map_odom_relay_enabled = bool(self.real_nav and self.gz and GZ_WORLD == "restaurant")' in source
    assert 'self._relay_slam_map_odom()' in source
    assert 'SIM_MAP_ODOM_RELAY enabled rate_hz=50 lead_s=0.15' in source
    assert 'age < -0.25 or age > 0.5' in source
    assert 'SIM_MAP_ODOM_RELAY published count=' in source


def test_gazebo_scoped_finger_joint_names_produce_measured_gripper_width():
    from base_athome_sim.sim_core import gripper_width_from_joint_states

    assert gripper_width_from_joint_states(
        ["robot::joint7", "robot::joint8"], [0.0375, -0.0375]
    ) == pytest.approx(75.0)
    assert gripper_width_from_joint_states(
        ["robot::joint7", "other::joint7", "robot::joint8"], [0.04, 0.03, -0.04]
    ) is None
    assert gripper_width_from_joint_states(["robot::joint7"], [0.04]) is None
