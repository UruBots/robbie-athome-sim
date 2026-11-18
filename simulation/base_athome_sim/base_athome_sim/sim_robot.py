#!/usr/bin/env python3
"""Robot simulado Base@Home: mismos topicos/servicios que la Jetson (Piper, pan-tilt, whisper),
la RealSense de la NUC, la camara de muneca y Nav2 (navigate_to_pose, /amcl_pose, /scan, TF).

Las apps reales (receptionist, manipulacion, mision) corren SIN CAMBIOS contra este nodo con
ATHOME_SIM=1 y un ROS_DOMAIN_ID distinto del robot (por defecto 42, ver alias `sim_env`).

Control del "mundo" (alias en scripts/aliases/nuc_aliases.sh):
  /sim/speech  (String)  lo que dice el invitado; solo se escucha con la ventana de escucha
                         abierta, igual que whisper_node en modo bajo demanda    -> sim_decir "..."
  /sim/guest   (String)  "door" (invitado en la puerta, mirando al robot) | "none" | "next"
  /sim/auto_guest (Bool) responde solo cada vez que el robot abre la ventana de escucha
"""
import glob
import copy
import json
import math
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped, Twist, Vector3
from geometry_msgs.msg import Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
try:
    from control_msgs.action import FollowJointTrajectory
except ImportError:  # pragma: no cover
    FollowJointTrajectory = None
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from std_msgs.msg import Bool, Empty, Float32, Float64, String
from tf2_msgs.msg import TFMessage
from std_srvs.srv import SetBool, Trigger
from base_athome_sim import sim_scene
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformListener
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

from base_athome_sim import gz_model, real_gap
from base_athome_sim.measured_arm_feedback import (
    arm_command_state_from_measured_sample,
    arm_status_from_measured_sample,
    apply_attached_object_gripper_width,
    measured_goal_deviation_deg,
    measured_goal_is_settled,
    measured_sample_stamp_advances,
    MAX_ARM_FEEDBACK_AGE_S,
    GZ_ARM_SETTLE_TIMEOUT_S,
)
from base_athome_sim.sim_core import (
    GripperStall, REST_DEG, SimArm, SimBase, SimMicrophone, SimPanTilt, sim_arm_start_state,
    sim_arm_start_state_matches, restaurant_handoff_confirmed,
    restaurant_base_hold_requested, restaurant_base_hold_angular_command, sim_arm_enable_requested,
    gripper_stall_enabled_after_detach,
    restaurant_gripper_contact_block,
    map_odom_relay_stamp,
)

D2R = math.pi / 180.0
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ARM_CONFIG = os.path.join(REPO_ROOT, "src/base_athome_perception/config/arm_poses_and_limits.yaml")
DEFAULT_FACES = ["/opt/ros/jazzy/share/rqt_image_view/resource/lena.png", "/usr/share/pixmaps/faces/surfer.jpg"]
AUTO_LINES = ["My name is John and my favorite drink is cider",
              "My name is Mary and my favorite drink is red wine"]


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def to_image(arr, encoding, stamp, frame):
    m = Image()
    m.header.stamp, m.header.frame_id = stamp, frame
    m.height, m.width = arr.shape[:2]
    m.encoding = encoding
    m.step = arr.strides[0]
    m.data = arr.tobytes()
    return m


GZ_WORLD = os.path.splitext(os.path.basename(os.environ.get("ATHOME_SIM_WORLD") or "casa_basura.sdf"))[0]   # donde se crean bolsa y persona


class SimRobot(Node):
    BAG_KNOT_BELOW_TOP_M, BAG_KNOT_MM = 0.04, 10.0
    PERSON_WALK_MPS = 0.35

    def __init__(self):
        super().__init__("sim_robot")
        p = self.declare_parameter
        p("map_yaml", os.path.expanduser("~/mapa_base_athome.yaml"))
        p("start_pose", [0.0, 0.0, 0.0])          # x, y, yaw (si hay waypoint 'home' se usa ese)
        p("object_xyz", [0.39, -0.11, 0.92])      # base_footprint: ~12 cm de avance desde la PRE-GRASP actual (el limite de IK esta en ~14-15 cm)
        p("object_diameter_m", 0.075)              # = object_diameter_m de la calibracion de muneca; la pinza se detiene aqui
        p("guest_distance_m", 1.2)
        p("faces", [""])                           # imagenes de rostros (vacio = publicas del sistema)
        p("auto_guest", False)
        p("camera_rate_hz", 15.0)
        p("nav_speed", 0.5)
        p("gazebo", False)                         # True: base, lidar y camaras vienen de gz sim
        gp = lambda n: self.get_parameter(n).value

        self.cb = ReentrantCallbackGroup()
        self.arm_status_cb = MutuallyExclusiveCallbackGroup()
        self.arm_status_period_last = None
        self.arm_status_periods = []
        self.arm_status_period_report_at = time.monotonic()
        self.lock = threading.Lock()
        sim_arm_enabled = sim_arm_enable_requested(os.environ)
        # ATHOME_SIM_REAL_GAP=1: como el Piper real, el brazo arranca sin torque (enabled=false)
        # y la capa de manipulacion debe habilitarlo por /arm/set_enable.
        gap_arm_disabled = sim_arm_enabled and real_gap.arm_starts_disabled(os.environ)
        self.arm, self.pt = SimArm(enabled=sim_arm_enabled and not gap_arm_disabled), SimPanTilt()
        if gap_arm_disabled:
            self.get_logger().info("ATHOME_SIM_REAL_GAP=1: brazo simulado arranca deshabilitado (enabled=false)")
        self.gz_start_joint_positions = None
        arm_start_pose = os.environ.get("ATHOME_SIM_ARM_START_POSE", "").strip()
        if arm_start_pose and sim_arm_enabled:
            initial = gz_model.initial_joint_positions_from_pose_config(ARM_CONFIG, arm_start_pose)
            self.gz_start_joint_positions = [initial[f"joint{index}"] for index in range(1, 9)]
            self.arm.q, start_width = sim_arm_start_state(initial)
            self.arm.gripper_mm = self.arm.gripper_cmd_mm = self.arm.gripper_target_mm = start_width
        if sim_arm_enabled:
            self.get_logger().info("Simulation arm enabled by explicit SIM=1/ATHOME_SIM=1 and ARM=1")
        self.mic = SimMicrophone()
        self.auto_guest = bool(gp("auto_guest"))
        self.auto_idx = 0
        self.listen_count = 0
        self.prompt = ""
        wp = self._waypoints().get("home")
        sx, sy, syaw = (wp["x"], wp["y"], math.radians(wp.get("yaw_deg", 0))) if wp else gp("start_pose")
        self.base = SimBase(sx, sy, syaw, speed=float(gp("nav_speed")))
        self.cmd_vel = (0.0, 0.0, 0.0)
        self.restaurant_base_hold = False
        self.restaurant_base_hold_yaw = None
        self.restaurant_base_hold_pose = None
        self.restaurant_base_hold_started_state = None
        self.restaurant_map_pose = None
        self.restaurant_pose_log_at = 0.0
        self.restaurant_handoff_model = None
        self.restaurant_handoff_at = 0.0
        self.restaurant_gripper_attach_reported = False
        self.obj = list(gp("object_xyz"))
        if os.environ.get("ATHOME_SIM_OBJECT_XYZ"):  # pruebas de alcance: "x,y,z" en base_footprint
            self.obj = [float(v) for v in os.environ["ATHOME_SIM_OBJECT_XYZ"].split(",")]
        elif GZ_WORLD == "restaurant":
            self.obj = [0.50, 0.16, 1.015]
        self.obj_attached = False
        self.gripper_was_closed = True
        # Bolsa fija en el mapa (ATHOME_SIM_BAG="x,y,z", z = la manija): sin esto el objeto sigue al robot
        self.bag_map = [float(v) for v in os.environ["ATHOME_SIM_BAG"].split(",")] if os.environ.get("ATHOME_SIM_BAG") else None
        self.person_xy = [float(v) for v in os.environ.get("ATHOME_SIM_PERSON", "0.5,1.8").split(",")]
        self.person_pointing = False
        self.person_points = os.environ.get("ATHOME_SIM_PERSON_POINTS", "1") == "1"   # 0: nunca hace el gesto
        self.point_err = float(os.environ.get("ATHOME_SIM_POINT_ERR_M", "0.15"))     # la gente no apunta con precision
        self.grasp_slips = os.environ.get("ATHOME_SIM_GRASP_SLIPS", "0") == "1"       # ningun agarre del piso sujeta la bolsa
        self.person_hands_over = os.environ.get("ATHOME_SIM_PERSON_HANDS", "1") == "1"  # 0: no se la da
        self.bag_in_hand = False
        # ATHOME_SIM_TABLE_OBJECT="x,y,z": objeto rigido, z es su centro; se agarra por el centro, con su diametro
        table_obj = os.environ.get("ATHOME_SIM_TABLE_OBJECT")
        self.rigid_object = bool(table_obj)
        if table_obj:
            self.bag_map = [float(v) for v in table_obj.split(",")]
        self.restaurant_item_poses = {}
        self.restaurant_item_model = "restaurant_counter_can_1" if GZ_WORLD == "restaurant" else None
        self.restaurant_item_contact = False
        self.restaurant_detached_pending_release = False
        walk = os.environ.get("ATHOME_SIM_PERSON_WALK", "")   # "x,y;x,y": camina al oir que el robot la va a seguir
        self.person_walk = [[float(v) for v in p.split(",")] for p in walk.split(";") if p]
        self.walk_start = None
        # ATHOME_SIM_PERSON_WALK_AFTER_S: empieza a caminar sola a los N s (sin esperar "follow you"), para probar a Nav2
        self.walk_after_s = float(os.environ.get("ATHOME_SIM_PERSON_WALK_AFTER_S", "0") or 0)
        self.t_start = time.time()
        self.person_sway = float(os.environ.get("ATHOME_SIM_PERSON_SWAY_M", "0.0"))
        # ATHOME_SIM_CUSTOMERS="x,y;x,y" (clientes sentados o de pie con mano levantada para Restaurant)
        self.customers_xy = []
        raw_cust = os.environ.get("ATHOME_SIM_CUSTOMERS", "").strip()
        if raw_cust:
            for item in raw_cust.split(";"):
                item = item.strip()
                if item:
                    parts = [v.strip() for v in item.split(",") if v.strip()]
                    if len(parts) >= 2:
                        try:
                            cx, cy = float(parts[0]), float(parts[1])
                            hand = parts[2] if len(parts) >= 3 else "right"
                            seated = (parts[3].lower() == "seated") if len(parts) >= 4 else False
                            self.customers_xy.append({"xy": [cx, cy], "hand": hand, "seated": seated})
                        except ValueError:
                            pass
        self.bag2 = [float(v) for v in os.environ["ATHOME_SIM_BAG2"].split(",")] if os.environ.get("ATHOME_SIM_BAG2") else None
        self.wrist_tilt_err = math.radians(float(os.environ.get("ATHOME_SIM_WRIST_TILT_ERR_DEG", "0")))   # montaje real distinto del modelo
        self.head_pix_err = float(os.environ.get("ATHOME_SIM_DEPTH_NOISE_MM", "5"))
        self.wrist_ghost = os.environ.get("ATHOME_SIM_WRIST_GHOST", "1") == "1"       # falso positivo sobre la garra
        self.rng = np.random.default_rng(0)
        self.torso_pos, self.torso_goal = 0.0, 0.0
        self.bag_delivered = False
        self.pub_torso_js = self.create_publisher(JointState, "/torso/joint_states", 10)
        self.pub_torso_st = self.create_publisher(String, "/torso/status", 10)
        self.pub_people = self.create_publisher(String, "/sim/people", 5)
        self.pub_receptionist_state = self.create_publisher(String, "/sim/receptionist/guest_state", 5)
        self.guest_present = False
        self.guest_idx = 0
        self.receptionist_seated_guests = []
        self.receptionist_seat_xy = None
        self.receptionist_bag_state = "idle"
        self.receptionist_bag_held = False
        self.receptionist_bag_model = "recepcionista_bolsa"
        self.receptionist_host_xy = [2.0, 1.9]
        self.receptionist_host_yaw = 0.0
        self.receptionist_follow_active = False
        self.receptionist_host_path_done = False
        self.receptionist_drop_xy = [4.15, 0.55]
        self.receptionist_bag_pose_update = 0.0
        self.guest_d = float(gp("guest_distance_m"))
        faces = [f for f in gp("faces") if f] or sorted(glob.glob(os.path.expanduser("~/sim_faces/*.jpg"))) or DEFAULT_FACES
        self.faces = ([] if (os.environ.get("ATHOME_SIM_REAL_PERCEPTION") == "1" or
                                   os.environ.get("ATHOME_SIM_RENDERED_HUMANS") == "1") else
                      [im for im in (cv2.imread(f) for f in faces) if im is not None])
        self.get_logger().info(f"Rostros simulados: {len(self.faces)}")
        calib = {}
        try:
            with open(ARM_CONFIG, encoding="utf-8") as f:
                calib = (yaml.safe_load(f) or {}).get("wrist_camera_calibration") or {}
        except OSError:
            pass
        self.wfx = float(calib.get("focal_px", 485.0))
        self.wcx, self.wcy = float(calib.get("cx_px", 320.0)), float(calib.get("cy_px", 240.0))
        self.gz = bool(gp("gazebo"))
        self.receptionist_humans = None
        if self.gz and self.auto_guest and (os.environ.get("ATHOME_SIM_REAL_PERCEPTION") == "1"
                                           or os.environ.get("ATHOME_SIM_RENDERED_HUMANS") == "1"):
            from .receptionist_humans import ReceptionistHumans
            self.receptionist_humans = ReceptionistHumans(self, GZ_WORLD)
        self.gz_arm_ready = False
        self.gz_arm_feedback = None
        self.gz_arm_feedback_at = 0.0
        self.gz_joint_state_periods = []
        self.gz_joint_state_last_received_at = None
        self.gz_joint_state_last_period_report = time.monotonic()
        self.gz_arm_state_initialized = False
        self.gz_initial_arm_logged = False
        self.gz_initial_pose_logged = False
        self.gz_initial_pose_started = time.monotonic()
        self.gz_arm_moving_until = 0.0
        self.gz_shown_xy = None
        self.map = None if self.gz else self._load_map(gp("map_yaml"))

        # --- Publicadores (mismos nombres que el robot real) ---
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_arm_js = self.create_publisher(JointState, "/arm/joint_states", 10)
        self.pub_pt_js = self.create_publisher(JointState, "/pan_tilt/joint_states", 10)
        self.pub_arm_status = self.create_publisher(String, "/arm/status", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_pt_conn = self.create_publisher(Bool, "/pan_tilt/connected", latched)
        self.pub_voice = self.create_publisher(String, "/voice/transcription", 10)
        self.pub_vresult = self.create_publisher(String, "/voice/result", 20)
        self.pub_vstatus = self.create_publisher(String, "/voice/status", 10)
        self.pub_level = self.create_publisher(Float32, "/voice/mic_level", 10)
        self.pub_rgb = self.create_publisher(Image, "/camera/camera/color/image_raw", qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, "/camera/camera/aligned_depth_to_color/image_raw", qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera/color/camera_info", qos_profile_sensor_data)
        self.pub_wrist = self.create_publisher(Image, "/arm_camera/image_raw", qos_profile_sensor_data)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_amcl = self.create_publisher(PoseWithCovarianceStamped, "/amcl_pose", latched)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.pub_restaurant_rear_scan = self.create_publisher(
            LaserScan, "/restaurant/rear_scan", qos_profile_sensor_data)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", latched)
        self.pub_world = self.create_publisher(MarkerArray, "/sim/world", 5)
        self.pub_receptionist_task = self.create_publisher(String, "/sim/receptionist/task_state", 5)
        self.pub_bag_attach = self.create_publisher(Empty, gz_model.BAG_ATTACH_TOPIC, 10)
        self.pub_bag_detach = self.create_publisher(Empty, gz_model.BAG_DETACH_TOPIC, 10)
        self.pub_recep_bag_attach = self.create_publisher(Empty, "/sim/gz/recepcionista_bolsa/attach", 10)
        self.pub_recep_bag_detach = self.create_publisher(Empty, "/sim/gz/recepcionista_bolsa/detach", 10)
        self.pub_restaurant_can_attach = {
            f"restaurant_counter_can_{idx}": self.create_publisher(Empty, f"/sim/gz/restaurant_counter_can_{idx}/attach", 10)
            for idx in (1, 2)
        }
        self.pub_restaurant_can_detach = {
            f"restaurant_counter_can_{idx}": self.create_publisher(Empty, f"/sim/gz/restaurant_counter_can_{idx}/detach", 10)
            for idx in (1, 2)
        }
        self.gz_ready_t = time.time() + 15.0
        self.pub_pt_conn.publish(Bool(data=True))
        self.create_timer(0.5, self._publish_receptionist_state)

        # --- Suscripciones / servicios ---
        sub = lambda t, n, f, q=10: self.create_subscription(t, n, f, q, callback_group=self.cb)
        for model_name in self.pub_restaurant_can_attach:
            sub(Empty, f"/sim/gz/{model_name}/attach",
                lambda _msg, name=model_name: self._restaurant_can_attached(name), 5)
            sub(Empty, f"/sim/gz/{model_name}/detach",
                lambda _msg, name=model_name: self._restaurant_can_detached(name), 5)
        sub(JointState, "/arm/joint_cmd", self._joint_cmd, 5)
        sub(JointTrajectory, "/arm/joint_trajectory", self._trajectory, 2)
        sub(Float32, "/arm/gripper_cmd", lambda m: self.arm.gripper(m.data), 5)
        sub(Bool, "/arm/enable", lambda m: self.arm.set_enable(m.data), 5)
        sub(Vector3, "/pan_tilt/target_degrees", lambda m: setattr(self.pt, "target", (m.x, m.y)))
        sub(JointState, "/pan_tilt/command", self._pt_cmd)
        sub(Bool, "/voice/mute", lambda m: setattr(self.mic, "muted", bool(m.data)))
        sub(String, "/voice/initial_prompt", lambda m: setattr(self, "prompt", m.data))
        sub(String, "/sim/speech", self._speech)
        sub(String, "/sim/guest", self._guest)
        sub(Bool, "/sim/auto_guest", lambda m: setattr(self, "auto_guest", bool(m.data)))
        sub(PoseStamped, "/goal_pose", self._goal_pose)  # "2D Goal Pose" de RViz, como bt_navigator
        sub(Twist, "/cmd_vel", self._cmd_vel)
        sub(Float64, "/torso/goal", self._torso_goal)
        sub(String, "/hri/say_something", self._heard_robot)
        sub(String, "/sim/receptionist/task_command", self._receptionist_task_cb, 5)
        srv = lambda t, n, f: self.create_service(t, n, f, callback_group=self.cb)
        srv(SetBool, "/arm/set_enable", self._srv_enable)
        srv(SetBool, "/pan_tilt/set_torque", self._srv_torque)
        srv(Trigger, "/voice/listen_trigger", self._srv_listen)
        sub(String, "/voice/listen_request", self._listen_request)
        sub(String, "/voice/cancel", lambda m: self._voice_cancelled(self.mic.cancel(m.data.strip()), "cancel"))
        srv(Trigger, "/arm/home", self._srv_home)
        if FollowJointTrajectory is not None:  # misma interfaz que piper_arm_controller (HOME-45)
            ActionServer(self, FollowJointTrajectory, "/arm/follow_joint_trajectory", execute_callback=self._fjt_execute,
                         goal_callback=lambda g: GoalResponse.ACCEPT if self.arm.enabled else GoalResponse.REJECT,
                         cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=self.cb)
        # ATHOME_SIM_NAV2=1: navega Nav2 de verdad; este nodo no ofrece la accion ni dice donde esta el robot en el mapa
        self.real_nav = os.environ.get("ATHOME_SIM_NAV2") == "1"
        if not self.real_nav:
            ActionServer(self, NavigateToPose, "navigate_to_pose", execute_callback=self._nav_execute,
                         goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda g: CancelResponse.ACCEPT,
                         callback_group=self.cb)

        self.tf = TransformBroadcaster(self)
        st = TransformStamped()
        st.header.frame_id, st.child_frame_id = "map", "odom"
        st.transform.rotation.w = 1.0
        self.static_tf = StaticTransformBroadcaster(self)
        if not self.real_nav:     # con Nav2, map -> odom y /amcl_pose los publica AMCL
            self.static_tf.sendTransform(st)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.map_odom_relay_enabled = bool(self.real_nav and self.gz and GZ_WORLD == "restaurant")
        self.map_odom_lock = threading.Lock()
        self.latest_slam_map_odom = None
        self.latest_slam_map_odom_received_s = 0.0
        self.map_odom_relay_count = 0
        if self.map_odom_relay_enabled:
            self.create_subscription(TFMessage, "/tf", self._cache_slam_map_odom, 10,
                                     callback_group=self.cb)
            self.get_logger().info(
                "SIM_MAP_ODOM_RELAY enabled rate_hz=50 lead_s=0.15 max_source_age_s=0.5")

        self.last = time.time()
        self.create_timer(0.02, self._tick)
        if self.gz:
            self._setup_gazebo()
        else:
            self.create_timer(1.0 / (min(float(gp("camera_rate_hz")), 5.0) if self.bag_map is not None else float(gp("camera_rate_hz"))),
                              self._cameras)       # dibujar la escena es caro: no dejar sin tiempo a la base y al elevador
            self.create_timer(0.1, self._scan)
        self.create_timer(0.5, self._slow)
        self.create_timer(0.05, self._fast)
        self.create_timer(0.05, self._arm_status, callback_group=self.arm_status_cb)
        if self.map is not None:
            self.pub_map.publish(self.map["msg"])
        self.get_logger().info("Robot simulado listo (Piper, pan-tilt, voz, camaras, Nav2 falso).")

    # ------------------------------------------------------------------ mundo
    @staticmethod
    def _waypoints():
        wps = {}
        for f in ("semantic_map.yaml", "waypoints.yaml"):
            try:
                with open(os.path.join(REPO_ROOT, "src/base_athome_navigation/config", f), encoding="utf-8") as fh:
                    for k, v in ((yaml.safe_load(fh) or {}).get("waypoints") or {}).items():
                        if isinstance(v, dict) and "x" in v:
                            wps.setdefault(k, v)
            except OSError:
                pass
        return wps

    def _load_map(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                info = yaml.safe_load(f)
            img_path = info["image"] if os.path.isabs(info["image"]) else os.path.join(os.path.dirname(path), info["image"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            res, origin = float(info["resolution"]), info["origin"]
            occ_th = float(info.get("occupied_thresh", 0.65))
            occ = (255 - img.astype(np.float32)) / 255.0 > occ_th   # negate=0
            occ = np.flipud(occ)                                     # fila 0 = y minima
            msg = OccupancyGrid()
            msg.header.frame_id = "map"
            msg.info.resolution = res
            msg.info.width, msg.info.height = img.shape[1], img.shape[0]
            msg.info.origin.position.x, msg.info.origin.position.y = float(origin[0]), float(origin[1])
            msg.info.origin.orientation.w = 1.0
            free = (255 - np.flipud(img).astype(np.float32)) / 255.0 < float(info.get("free_thresh", 0.25))
            grid = np.full(occ.shape, -1, np.int8)
            grid[free], grid[occ] = 0, 100
            msg.data = grid.flatten().tolist()
            self.get_logger().info(f"Mapa cargado: {path}")
            return {"occ": occ, "res": res, "ox": float(origin[0]), "oy": float(origin[1]), "msg": msg}
        except Exception as e:  # sin mapa: /scan vacio, todo lo demas funciona
            self.get_logger().warn(f"Sin mapa para /scan ({e})")
            return None

    def _guest(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ("door", "present", "si", "yes"):
            self.guest_present = True
        elif cmd in ("none", "absent", "no", "seated", "sentado"):
            self.guest_present = False
        elif cmd == "next":
            self.guest_idx += 1
            self.guest_present = True
            if self.gz:
                threading.Thread(target=self._gz_reset_reception_guest, daemon=True).start()
        self.get_logger().info(f"Invitado #{self.guest_idx + 1}: {'presente' if self.guest_present else 'ausente'}")

    def _goal_pose(self, msg):
        q = msg.pose.orientation
        self.base.set_goal(msg.pose.position.x, msg.pose.position.y, 2 * math.atan2(q.z, q.w))

    def _cmd_vel(self, msg):
        if self.restaurant_base_hold:
            self.cmd_vel = (0.0, 0.0, 0.0)
        else:
            self.cmd_vel = (msg.linear.x, msg.angular.z, time.time())

    def _speech(self, msg):
        ok = self.mic.hear(msg.data)
        self.get_logger().info(f"Invitado dice: '{msg.data}' -> {'ESCUCHADO' if ok else 'PERDIDO (el robot no estaba escuchando)'}")

    # ------------------------------------------------------------------ callbacks reales
    def _joint_cmd(self, msg):
        if len(msg.position) < 6:
            return
        is_rad = all(abs(v) <= 3.5 for v in msg.position[:6])
        tgt = [v / D2R if is_rad else v for v in msg.position[:6]]
        dur = max(0.6, min(10.0, msg.velocity[0])) if len(msg.velocity) and msg.velocity[0] > 0 else 2.5
        with self.lock:
            self.arm.joint_cmd(tgt, dur)
            if len(msg.position) >= 7:
                g = msg.position[6]
                self.arm.gripper(g * 1000.0 if g < 1.0 else g)

    def _trajectory(self, msg):
        try:
            idx = [msg.joint_names.index(f"joint{i}") for i in range(1, 7)]
        except ValueError:
            return
        times, pts = [], []
        for pt in msg.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if times and t <= times[-1]:
                continue
            times.append(t)
            pts.append([pt.positions[i] / D2R for i in idx])
        if pts:
            with self.lock:
                self.arm.trajectory(times, pts)

    def _fjt_execute(self, goal_handle):
        R = FollowJointTrajectory.Result
        res = R()
        started_wall = time.monotonic()
        msg = goal_handle.request.trajectory
        try:
            idx = [list(msg.joint_names).index(f"joint{i}") for i in range(1, 7)]
        except ValueError:
            res.error_code, res.error_string = R.INVALID_JOINTS, "faltan joint1..joint6"
            goal_handle.abort()
            return res
        times, pts = [], []
        for pt in msg.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if times and t <= times[-1]:
                continue
            times.append(t)
            pts.append([pt.positions[i] / D2R for i in idx])
        if not pts:
            res.error_code, res.error_string = R.INVALID_GOAL, "sin puntos"
            goal_handle.abort()
            return res
        with self.lock:
            self.arm.trajectory(times, pts)
        settle_timeout = GZ_ARM_SETTLE_TIMEOUT_S if self.gz else 2.0
        t_end = time.time() + times[-1] + settle_timeout
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                with self.lock:
                    feedback = self.gz_arm_feedback
                    age = self._gz_feedback_clock() - self.gz_arm_feedback_at
                    moving = self._gz_feedback_clock() < self.gz_arm_moving_until
                deviation = (measured_goal_deviation_deg(feedback, age, pts[-1]) if self.gz
                             else max(abs(a - b) for a, b in zip(self.arm.q, pts[-1])))
                self.get_logger().warn(
                    f"SIM_FJT_CANCEL trajectory_s={times[-1]:.2f} elapsed_s={time.monotonic() - started_wall:.2f} "
                    f"moving={moving} feedback_age_s={age:.3f} final_deviation_deg={deviation}")
                with self.lock:
                    self.arm.joint_cmd(list(self.arm.q), 0.6)
                goal_handle.canceled()
                return res
            if not self.arm.enabled:
                res.error_code, res.error_string = R.PATH_TOLERANCE_VIOLATED, "sim: brazo deshabilitado"
                goal_handle.abort()
                return res
            with self.lock:
                moving = self.arm.moving
                measured_feedback = self.gz_arm_feedback
                feedback_now = self._gz_feedback_clock()
                measured_age = feedback_now - self.gz_arm_feedback_at
                measured_moving = feedback_now < self.gz_arm_moving_until
            if not moving:
                goal_deg = pts[-1]
                if self.gz:
                    dev = measured_goal_deviation_deg(measured_feedback, measured_age, goal_deg)
                    if dev is None:
                        if time.time() <= t_end:
                            time.sleep(0.05)
                            continue
                        res.error_code, res.error_string = R.GOAL_TOLERANCE_VIOLATED, "sim: feedback medido obsoleto"
                        goal_handle.abort()
                        return res
                    settled = measured_goal_is_settled(
                        measured_feedback, measured_age, goal_deg, is_moving=measured_moving)
                    if not settled:
                        if time.time() <= t_end:
                            time.sleep(0.05)
                            continue
                        res.error_code, res.error_string = R.GOAL_TOLERANCE_VIOLATED, (
                            f"sim: Gazebo no se asentó al final de trayectoria (desvio {dev:.2f} grados)")
                        goal_handle.abort()
                        return res
                else:
                    dev = max(abs(a - b) for a, b in zip(self.arm.q, goal_deg))
                if dev <= 2.0:
                    res.error_code, res.error_string = R.SUCCESSFUL, f"sim: llegada verificada ({dev:.2f} grados)"
                    goal_handle.succeed()
                else:
                    res.error_code, res.error_string = R.GOAL_TOLERANCE_VIOLATED, f"sim: desvio {dev:.2f} grados"
                    goal_handle.abort()
                return res
            if time.time() > t_end:
                res.error_code, res.error_string = R.GOAL_TOLERANCE_VIOLATED, "sim: timeout"
                goal_handle.abort()
                return res
            time.sleep(0.05)
        return res

    def _pt_cmd(self, msg):
        pan, tilt = self.pt.target
        for n, v in zip(msg.name, msg.position):
            if "pan" in n:
                pan = v / D2R
            elif "tilt" in n:
                tilt = v / D2R
        self.pt.target = (pan, tilt)

    def _srv_enable(self, req, res):
        self.arm.set_enable(req.data)
        res.success, res.message = True, f"sim: brazo {'habilitado' if req.data else 'libre'}"
        return res

    def _srv_torque(self, req, res):
        self.pt.torque = bool(req.data)
        res.success, res.message = True, f"sim: torque pan-tilt {req.data}"
        return res

    def _srv_home(self, req, res):
        self.arm.joint_cmd(REST_DEG, 3.0)
        res.success, res.message = True, "sim: home"
        return res

    def _voice_event(self, rid, event, **kw):
        d = {"id": rid, "event": event, "t": round(time.time(), 3), "sim": True}
        d.update(kw)
        self.pub_vresult.publish(String(data=json.dumps(d)))

    def _voice_cancelled(self, rid, why):
        if rid:
            self._voice_event(rid, "result", status="cancelled", text="", reason=why)

    def _open_listen(self, rid, prompt="", start_timeout=None, end_silence=None):
        self._voice_cancelled(self.mic.request(rid, start_timeout, end_silence), "replaced")
        self._voice_event(rid, "listening")
        self.get_logger().info(f"Escucha {rid} abierta (espera {start_timeout or self.mic.start_timeout:.0f} s; "
                               f"prompt: '{(prompt or self.prompt)[:50]}')")
        if self.auto_guest:
            if "answer yes or no" in (prompt or self.prompt).lower():
                line = "yes"
            else:
                line = AUTO_LINES[self.auto_idx % len(AUTO_LINES)]
                self.auto_idx += 1
            threading.Timer(1.0, lambda: self._speech(String(data=line))).start()

    def _listen_request(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            d = {}
        self.listen_count += 1
        self._open_listen(str(d.get("id") or f"r{self.listen_count}"), d.get("prompt", ""),
                          d.get("start_timeout"), d.get("end_silence"))

    def _srv_listen(self, req, res):
        self.listen_count += 1
        rid = f"t{self.listen_count}"
        self._open_listen(rid)
        res.success, res.message = True, json.dumps({"id": rid})
        return res

    async def _nav_execute(self, goal_handle):
        p = goal_handle.request.pose.pose
        yaw = 2 * math.atan2(p.orientation.z, p.orientation.w)
        self.base.set_goal(p.position.x, p.position.y, yaw)
        mine = self.base.goal  # identidad del objetivo de ESTE goal (un goal nuevo lo reemplaza)
        self.get_logger().info(f"Nav2 (sim): ir a ({p.position.x:.2f}, {p.position.y:.2f})")
        fb = NavigateToPose.Feedback()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                if self.base.goal is mine:  # no pisar el objetivo de un goal posterior
                    self.base.goal = None
                goal_handle.canceled()
                return NavigateToPose.Result()
            if self.base.goal is not mine and self.base.goal is not None:
                goal_handle.abort()  # reemplazado por otro goal (como el preempt de Nav2)
                return NavigateToPose.Result()
            if self.base.goal is None:
                break
            fb.distance_remaining = float(math.hypot(p.position.x - self.base.x, p.position.y - self.base.y))
            fb.current_pose.header.frame_id = "map"
            fb.current_pose.pose.position.x, fb.current_pose.pose.position.y = self.base.x, self.base.y
            goal_handle.publish_feedback(fb)
            time.sleep(0.1)
        goal_handle.succeed()
        return NavigateToPose.Result()

    # ------------------------------------------------------------------ bucles
    def _tick(self):
        now = time.time()
        dt, self.last = now - self.last, now
        with self.lock:
            self.arm.update(now)
        self.pt.update(dt)
        step = 0.10 * dt       # elevador simulado, 10 cm/s
        self.torso_pos += max(-step, min(step, self.torso_goal - self.torso_pos))
        if self.gz:
            self._gz_tick(now)
        elif self.base.goal is not None:
            self.base.update(dt)
        elif now - self.cmd_vel[2] < 0.5:  # teleop directo (sin objetivo de Nav2)
            v, w = self.cmd_vel[0], self.cmd_vel[1]
            self.base.yaw += w * dt
            self.base.x += v * dt * math.cos(self.base.yaw)
            self.base.y += v * dt * math.sin(self.base.yaw)
        stamp = self.get_clock().now().to_msg()
        q_command = [v * D2R for v in self.arm.q]
        self._gz_gripper_feedback(now)
        half = min(gz_model.GRIPPER_HALF_MAX_M, max(0.0, self.arm.gripper_mm / 2000.0))
        command_arm_positions = q_command + [half, -half]
        if self.gz:
            with self.lock:
                feedback = self.gz_arm_feedback
                feedback_age = self._gz_feedback_clock() - self.gz_arm_feedback_at
            if feedback is not None and 0.0 <= feedback_age <= MAX_ARM_FEEDBACK_AGE_S:
                arm_names = [f"joint{i}" for i in range(1, 9)]
                arm_positions = list(feedback["positions_rad"])
                arm_stamp = feedback["stamp"]
            else:
                arm_names, arm_positions, arm_stamp = [], [], stamp
        else:
            arm_names = [f"joint{i}" for i in range(1, 9)]
            arm_positions, arm_stamp = command_arm_positions, stamp
        arm = JointState(name=arm_names, position=arm_positions)
        arm.header.stamp = arm_stamp
        pt = JointState(name=["pan_joint", "tilt_joint"], position=[self.pt.pan * D2R, self.pt.tilt * D2R])
        pt.header.stamp = stamp
        full = JointState(name=arm.name + pt.name + ["torso_lift_joint"],
                          position=list(arm.position) + list(pt.position) + [self.torso_pos])
        full.header.stamp = arm_stamp if self.gz and arm.name else stamp
        joint_commands = list(zip([f"joint{i}" for i in range(1, 7)] + pt.name + ["torso_lift_joint"],
                                  q_command + list(pt.position) + [self.torso_pos]))
        if self.gz:
            cmd_half = min(gz_model.GRIPPER_HALF_MAX_M, max(0.0, self.arm.gripper_cmd_mm / 2000.0))
            joint_commands += [("joint7", cmd_half), ("joint8", -cmd_half)]
        if self.gz and os.environ.get("ATHOME_SIM_REAL_PERCEPTION") == "1":
            from base_athome_sim.restaurant_humans import apply_measured_head
            apply_measured_head(self, full, pt)
        self.pub_js.publish(full)
        self.pub_arm_js.publish(arm)
        if pt.name:
            self.pub_pt_js.publish(pt)
        if self.gz:
            for j, v in joint_commands:
                self.gz_joint_pubs[j].publish(Float64(data=float(v)))
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, "odom", "base_footprint"
        t.transform.translation.x, t.transform.translation.y = self.base.x, self.base.y
        _, _, t.transform.rotation.z, t.transform.rotation.w = yaw_quat(self.base.yaw)
        self.tf.sendTransform(t)
        self._relay_slam_map_odom()
        for rid, status, text in self.mic.due(now):
            if status == "ok" and restaurant_handoff_confirmed(text):
                self._stage_restaurant_handoff_can()
            self._voice_event(rid, "result", status=status, text=text,
                              reason="" if status == "ok" else "start_timeout")
            if status == "ok":
                self.pub_voice.publish(String(data=text))
            self.get_logger().info(f"/voice/result [{rid}] {status} '{text}'")

    def _cache_slam_map_odom(self, msg):
        """Remember fresh SLAM map->odom corrections, ignoring this node's future-dated relay."""
        now_s = self.get_clock().now().nanoseconds * 1e-9
        for transform in msg.transforms:
            if (transform.header.frame_id.lstrip("/"), transform.child_frame_id.lstrip("/")) != ("map", "odom"):
                continue
            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            age = now_s - stamp
            # SLAM Toolbox can future-date its estimate slightly; receipt time
            # still bounds how long the cached transform may be relayed.
            if not math.isfinite(age) or age < -0.25 or age > 0.5:
                continue
            with self.map_odom_lock:
                self.latest_slam_map_odom = copy.deepcopy(transform)
                self.latest_slam_map_odom_received_s = now_s

    def _relay_slam_map_odom(self):
        """Publish the latest Restaurant SLAM correction at the robot's 50 Hz sim tick."""
        if not self.map_odom_relay_enabled:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        with self.map_odom_lock:
            source = copy.deepcopy(self.latest_slam_map_odom)
            received_s = self.latest_slam_map_odom_received_s
        stamp_s = map_odom_relay_stamp(now_s, received_s)
        if source is None or stamp_s is None:
            return
        stamp_sec = math.floor(stamp_s)
        stamp_nsec = round((stamp_s - stamp_sec) * 1e9)
        if stamp_nsec >= 1_000_000_000:
            stamp_sec += 1
            stamp_nsec -= 1_000_000_000
        relay = TransformStamped()
        relay.header.frame_id, relay.child_frame_id = "map", "odom"
        relay.header.stamp.sec, relay.header.stamp.nanosec = stamp_sec, stamp_nsec
        relay.transform = source.transform
        self.tf.sendTransform(relay)
        self.map_odom_relay_count += 1
        if self.map_odom_relay_count == 1 or self.map_odom_relay_count % 500 == 0:
            source_stamp = source.header.stamp.sec + source.header.stamp.nanosec * 1e-9
            self.get_logger().info(
                f"SIM_MAP_ODOM_RELAY published count={self.map_odom_relay_count} "
                f"source_age_s={now_s - source_stamp:.3f} "
                f"relay_lead_s={stamp_s - now_s:.3f}")

    # ------------------------------------------------------------------ Gazebo
    def _setup_gazebo(self):
        """gz sim aporta fisica de la base, lidar y camaras; este nodo sigue siendo el 'firmware'
        (Piper, pan-tilt, voz, Nav2 falso) y reenvia los sensores con los nombres del robot real."""
        self.gz_joint_pubs = {j: self.create_publisher(Float64, gz_model.joint_cmd_topic(j), 10)
                              for j in gz_model.POSITION_JOINTS}
        self.gz_cmd = self.create_publisher(Twist, f"/model/{gz_model.MODEL}/cmd_vel", 10)
        sub = lambda t, n, f: self.create_subscription(t, n, f, qos_profile_sensor_data, callback_group=self.cb)
        sub(Odometry, f"/model/{gz_model.MODEL}/odometry", self._gz_odom)
        sub(LaserScan, "/sim/gz/scan", self._gz_scan)
        sub(LaserScan, "/sim/gz/rear_scan", self._gz_rear_scan)
        sub(Image, "/sim/gz/head/image", self._gz_head)
        sub(Image, "/sim/gz/head/depth_image", self._gz_depth)
        sub(CameraInfo, "/sim/gz/head/camera_info", self._gz_info)
        sub(Image, "/sim/gz/wrist", self._gz_wrist)
        self.gripper_stall = GripperStall()
        sub(JointState, "/sim/gz/joint_states", self._gz_finger_states)
        if os.environ.get("ATHOME_SIM_REAL_PERCEPTION") == "1":
            from base_athome_sim.restaurant_humans import setup_measured_head
            setup_measured_head(self)
        from base_athome_sim.restaurant_humans import rendered_restaurant
        if rendered_restaurant() or GZ_WORLD == "restaurant":
            for model_name in gz_model.restaurant_counter_item_model_names():
                sub(Odometry, gz_model.restaurant_item_odometry_topic(model_name),
                    lambda msg, name=model_name: self._restaurant_item_odom(name, msg))
        if rendered_restaurant():
            from base_athome_sim.restaurant_humans import ParticipantGestures
            self.restaurant_gestures = ParticipantGestures()
            self.rendered_human_gestures = {2: True}
            sub(String, "/restaurant/state", self._restaurant_human_state)
        elif GZ_WORLD == "restaurant":
            sub(String, "/restaurant/state", self._restaurant_base_state)
        if GZ_WORLD == "restaurant":
            sub(PoseWithCovarianceStamped, "/amcl_pose", self._restaurant_map_pose)
            self.create_timer(0.5, self._log_initial_restaurant_pose)
            # Wake at 20 Hz and gate by ROS simulation time to sustain a 10 Hz
            # evidence stream even when individual callbacks are delayed.
            self.create_timer(0.05, self._restaurant_pose_log)
        self.get_logger().info("Modo Gazebo: base/lidar/camaras desde gz sim")
        if self.bag_map is not None:
            self.pub_bag_attach = self.create_publisher(Empty, gz_model.BAG_ATTACH_TOPIC, 10)
            self.pub_bag_detach = self.create_publisher(Empty, gz_model.BAG_DETACH_TOPIC, 10)
            self.bag_top = float(self.bag_map[2])
            self.gz_arm_shown, self.gz_arm_ready = None, False
            self.gz_moving = False
            self.gz_ready_t = time.time() + 15.0     # la union nace atada al crear la bolsa: soltarla hasta entonces
            sub(Odometry, gz_model.BAG_ODOM_TOPIC, self._gz_bag_odom)
            self.create_timer(0.2, self._gz_people)
            threading.Thread(target=self._gz_spawn_scene, daemon=True).start()
        elif self.customers_xy:     # Restaurant: clientes sin bolsa. Sin esto /sim/people nunca se publicaba
            self.create_timer(0.2, self._gz_people)
            threading.Thread(target=self._gz_spawn_scene, daemon=True).start()
        elif self.auto_guest:
            threading.Thread(target=self._gz_spawn_scene, daemon=True).start()

    def _gz_finger_states(self, msg):
        """Keep measured Gazebo arm joints and finger width as the source of feedback."""
        from base_athome_sim.sim_core import gripper_width_from_joint_states
        positions = {str(name).rsplit("::", 1)[-1]: value
                     for name, value in zip(msg.name, msg.position)}
        if not self.gz_initial_arm_logged and all(f"joint{index}" in positions for index in range(1, 9)):
            self.gz_initial_arm_logged = True
            actual = [float(positions[f"joint{index}"]) for index in range(1, 9)]
            expected = None
            pose_name = os.environ.get("ATHOME_SIM_ARM_START_POSE", "").strip()
            if pose_name:
                try:
                    expected = gz_model.initial_joint_positions_from_pose_config(ARM_CONFIG, pose_name)
                except (OSError, ValueError) as exc:
                    self.get_logger().warn(f"SIM_INITIAL_ARM_CONFIG invalid: {exc}")
            expected_text = ("unavailable" if expected is None else ",".join(
                f"{expected[f'joint{index}']:.4f}" for index in range(1, 9)))
            error = ("unavailable" if expected is None else f"{max(abs(actual[i] - expected[f'joint{i + 1}']) for i in range(8)):.4f}")
            self.get_logger().info(
                "SIM_INITIAL_ARM_JOINTS measured=" + ",".join(f"{value:.4f}" for value in actual)
                + f" expected={expected_text} max_error={error}")
        arm_names = [f"joint{i}" for i in range(1, 7)]
        try:
            arm_rad = [float(positions[name]) for name in arm_names]
            finger_rad = [float(positions["joint7"]), float(positions["joint8"])]
            if not all(math.isfinite(value) for value in arm_rad + finger_rad):
                raise ValueError("non-finite measured arm joint")
            stamp = msg.header.stamp
            sample = {
                "positions_rad": arm_rad + finger_rad,
                "joints_deg": [math.degrees(value) for value in arm_rad],
                "gripper_mm": 1000.0 * (finger_rad[0] - finger_rad[1]),
                "stamp": stamp,
                "stamp_sec": int(stamp.sec),
                "stamp_nanosec": int(stamp.nanosec),
            }
            received_at = self._gz_feedback_clock()
            with self.lock:
                previous = self.gz_arm_feedback
                if not measured_sample_stamp_advances(previous, sample):
                    return
                if previous is not None and max(
                        abs(a - b) for a, b in zip(sample["joints_deg"], previous["joints_deg"])) > 0.15:
                    self.gz_arm_moving_until = received_at + 0.20
                self.gz_arm_feedback = sample
                self.gz_arm_feedback_at = received_at
                previous_received_at = self.gz_joint_state_last_received_at
                self.gz_joint_state_last_received_at = received_at
                if previous_received_at is not None and received_at > previous_received_at:
                    self.gz_joint_state_periods.append(received_at - previous_received_at)
                    self.gz_joint_state_periods = self.gz_joint_state_periods[-100:]
                report_now = time.monotonic()
                if report_now - self.gz_joint_state_last_period_report >= 5.0:
                    periods = self.gz_joint_state_periods
                    if periods:
                        mean_period = sum(periods) / len(periods)
                        max_period = max(periods)
                        effective_hz = len(periods) / sum(periods)
                        self.get_logger().info(
                            f"SIM_GZ_JOINT_STATE_PERIOD samples={len(periods)} "
                            f"mean_s={mean_period:.4f} max_s={max_period:.4f} "
                            f"effective_hz={effective_hz:.1f}")
                    else:
                        self.get_logger().warn("SIM_GZ_JOINT_STATE_PERIOD no advancing samples")
                    self.gz_joint_state_last_period_report = report_now
                if not self.gz_arm_state_initialized:
                    if (self.gz_start_joint_positions is not None
                            and not sim_arm_start_state_matches(
                                sample["positions_rad"], self.gz_start_joint_positions)):
                        return
                    if self.gz_start_joint_positions is not None:
                        self.get_logger().info(
                            "SIM_INITIAL_ARM_SETTLED measured=" + ",".join(
                                f"{value:.4f}" for value in sample["positions_rad"]))
                    state = arm_command_state_from_measured_sample(sample)
                    if state is not None:
                        self.arm.q, self.arm.gripper_mm = state
                        self.arm.gripper_cmd_mm = self.arm.gripper_target_mm = state[1]
                        self.gz_arm_state_initialized = True
        except (KeyError, TypeError, ValueError, OverflowError):
            with self.lock:
                self.gz_arm_feedback = None
                self.gz_arm_feedback_at = 0.0
                self.gz_arm_moving_until = 0.0
        width = gripper_width_from_joint_states(msg.name, msg.position)
        if width is not None:
            self.gripper_stall.measure(width, time.time())

    def _physical_grasp(self):
        forced = os.environ.get("ATHOME_SIM_PHYSICAL_GRASP")
        if forced in ("0", "1"):
            return forced == "1"
        return False

    def _gz_gripper_feedback(self, now):
        if not self._physical_grasp():
            return
        if not gripper_stall_enabled_after_detach(
                self.restaurant_detached_pending_release, self.arm.gripper_target_mm):
            self.arm.grip_block_mm = None
            return
        if self.restaurant_detached_pending_release:
            self.restaurant_detached_pending_release = False
            self.gripper_stall.clear()
        self.arm.grip_block_mm = self.gripper_stall.block_mm(self.arm.gripper_target_mm, now)
        if self.arm.grip_block_mm is not None and not self.arm._grip:
            self.arm.gripper_mm = max(self.arm.gripper_cmd_mm, self.arm.grip_block_mm)
        elif self.arm.grip_block_mm is None and not self.arm._grip:
            self.arm.gripper_mm = self.arm.gripper_cmd_mm

    def _tcp_in_map(self):
        tcp = self._lookup("base_footprint", "tcp_link")[1]
        c, s = math.cos(self.base.yaw), math.sin(self.base.yaw)
        return (self.base.x + c * tcp[0] - s * tcp[1], self.base.y + s * tcp[0] + c * tcp[1], float(tcp[2]) - self.bag_top / 2.0 + 0.04)

    def _gz_run(self, cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError) as e:
            self.get_logger().warn(f"gz service: {e}")
            return ""

    def _customers(self):
        """Devuelve las personas clientes con mano levantada segun ATHOME_SIM_CUSTOMERS."""
        out = []
        robot_pos = (self.base.x, self.base.y)
        for c in self.customers_xy:
            out.append(sim_scene.make_customer(
                c["xy"],
                facing_xy=robot_pos,
                hand=c.get("hand", "none"),
                seated=c.get("seated", False)
            ))
        return out

    def _restaurant_human_state(self, msg):
        """Retire rendered and logical hand signals after the participant is served."""
        self._restaurant_base_state(msg)
        try:
            state = json.loads(msg.data)
            raised = self.restaurant_gestures.observe(state)
        except (ValueError, TypeError, KeyError):
            return
        for index in range(3):
            if index < len(self.customers_xy):
                self.customers_xy[index]["hand"] = "right" if index in raised else "none"
            desired = index in raised
            if self.rendered_human_gestures.get(index) != desired:
                self.rendered_human_gestures[index] = desired
                threading.Thread(target=self._replace_rendered_human, args=(index, desired), daemon=True).start()

    def _restaurant_base_state(self, msg):
        """Discard motion requests and hold zero velocity during simulated arm manipulation."""
        try:
            snapshot = json.loads(msg.data)
            state = snapshot.get("state")
        except (ValueError, TypeError, AttributeError):
            return
        info = snapshot.get("info") or {}
        if state == "WAITING_BARMAN" and info.get("placement") == "gripper":
            try:
                slot = max(int(snapshot.get("customer_idx", 0)),
                           int(snapshot.get("served", 0))) + 1
                slot = max(1, slot)
            except (TypeError, ValueError, OverflowError):
                slot = 1
            self.restaurant_handoff_model = f"restaurant_counter_can_{slot}"
            self.restaurant_handoff_at = time.monotonic()
        hold = restaurant_base_hold_requested(state, info)
        if hold and not self.restaurant_base_hold:
            self.base.goal = None
            self.restaurant_base_hold_yaw = self.base.yaw
            self.restaurant_base_hold_pose = (self.base.x, self.base.y, self.base.yaw)
            self.restaurant_base_hold_started_state = state
            self.get_logger().info(
                f"SIM_RESTAURANT_BASE_HOLD_START state={state} "
                f"odom_xy_yaw={self.base.x:.4f},{self.base.y:.4f},{self.base.yaw:.4f}")
        elif not hold and self.restaurant_base_hold:
            start = self.restaurant_base_hold_pose
            end = (self.base.x, self.base.y, self.base.yaw)
            if start is not None:
                distance = math.hypot(end[0] - start[0], end[1] - start[1])
                yaw_drift = math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2]))
                self.get_logger().info(
                    f"SIM_RESTAURANT_BASE_HOLD_END from={self.restaurant_base_hold_started_state} to={state} "
                    f"odom_start={start[0]:.4f},{start[1]:.4f},{start[2]:.4f} "
                    f"odom_end={end[0]:.4f},{end[1]:.4f},{end[2]:.4f} "
                    f"drift_m={distance:.4f} yaw_drift_rad={yaw_drift:.4f}")
            self.restaurant_base_hold_pose = None
            self.restaurant_base_hold_started_state = None
        if hold:
            self.cmd_vel = (0.0, 0.0, 0.0)
        else:
            self.restaurant_base_hold_yaw = None
        self.restaurant_base_hold = hold

    def _stage_restaurant_handoff_can(self):
        """Simulate the barman placing the can into the open hand for Plan B."""
        model = self.restaurant_handoff_model
        if (not self.gz or not model or time.monotonic() - self.restaurant_handoff_at > 10.0
                or self.arm.gripper_cmd_mm < 65.0):
            return False
        transform = self._lookup("base_footprint", "tcp_link")
        if transform is None:
            self.get_logger().warn("Restaurant barman handoff: TCP unavailable")
            return False
        tcp = transform[1]
        c, s = math.cos(self.base.yaw), math.sin(self.base.yaw)
        pose = (self.base.x + c * float(tcp[0]) - s * float(tcp[1]),
                self.base.y + s * float(tcp[0]) + c * float(tcp[1]), float(tcp[2]))
        result = self._gz_run(gz_model.set_pose_command(GZ_WORLD, model, pose))
        if "true" not in result.lower():
            self.get_logger().warn(f"Restaurant barman handoff: no pude colocar {model} en la pinza")
            return False
        self.restaurant_item_model = model
        self.bag_map = list(pose)
        self.rigid_object = True
        self.restaurant_item_contact = False
        self.get_logger().info(
            f"Restaurant barman placed {model} in open gripper at {tuple(round(v, 3) for v in pose)}; waiting for measured grip")
        return True

    def _restaurant_map_pose(self, msg):
        """Cache AMCL map pose for campaign evidence; docking drift guard uses /odom."""
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.restaurant_map_pose = (p.x, p.y, math.atan2(
            2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def _log_initial_restaurant_pose(self):
        """Compare first Gazebo odometry and SLAM map pose against the requested spawn pose."""
        if GZ_WORLD != "restaurant" or self.gz_initial_pose_logged:
            return
        expected_values = [float(value) for value in os.environ.get("ATHOME_SIM_START", "0,0,0").split(",")]
        if len(expected_values) != 3:
            self.get_logger().error("SIM_INITIAL_POSE invalid ATHOME_SIM_START; expected x,y,yaw_deg")
            self.gz_initial_pose_logged = True
            return
        expected = (expected_values[0], expected_values[1], math.radians(expected_values[2]))
        odom = (float(self.base.x), float(self.base.y), float(self.base.yaw))
        tf = self._lookup("map", "base_footprint")
        map_pose = None
        if tf is not None:
            rotation, translation = tf
            map_pose = (float(translation[0]), float(translation[1]),
                        math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
        timed_out = time.monotonic() - self.gz_initial_pose_started >= 30.0
        if map_pose is None and not timed_out:
            return
        wrap = lambda angle: math.atan2(math.sin(angle), math.cos(angle))
        errors = [math.hypot(odom[0] - expected[0], odom[1] - expected[1]),
                  abs(wrap(odom[2] - expected[2]))]
        if map_pose is not None:
            errors.extend((math.hypot(map_pose[0] - expected[0], map_pose[1] - expected[1]),
                           abs(wrap(map_pose[2] - expected[2]))))
        self.gz_initial_pose_logged = True
        fmt = lambda pose: "unavailable" if pose is None else ",".join(f"{value:.4f}" for value in pose)
        matched = map_pose is not None and max(errors) <= 0.05
        self.get_logger().info(
            f"SIM_INITIAL_POSE expected_gz={fmt(expected)} odom={fmt(odom)} map={fmt(map_pose)} "
            f"max_error={max(errors):.4f} match={'yes' if matched else 'no'}")

    def _restaurant_pose_log(self):
        """Record estimated map pose and physical odometry at 10 Hz while manipulating."""
        if not self.restaurant_base_hold:
            return
        now = self.get_clock().now().nanoseconds
        if now - self.restaurant_pose_log_at < 100_000_000:
            return
        self.restaurant_pose_log_at = now
        odom = (self.base.x, self.base.y, self.base.yaw)
        map_pose = self.restaurant_map_pose
        if map_pose is None:
            transform = self._lookup("map", "base_footprint")
            if transform is not None:
                rotation, translation = transform
                map_pose = (float(translation[0]), float(translation[1]),
                            math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
            elif not self.real_nav:
                map_pose = odom  # fake navigation publishes map->odom identity
        fmt = lambda pose: "unavailable" if pose is None else ",".join(f"{v:.4f}" for v in pose)
        self.get_logger().info(
            f"RESTAURANT_BASE_POSE map_xy_yaw={fmt(map_pose)} odom_xy_yaw={fmt(odom)}")

    def _replace_rendered_human(self, index, raised):
        from base_athome_sim.restaurant_humans import gesture_model_sdf, RENDERED_MODELS
        name = RENDERED_MODELS[index][0]
        for attempt in range(3):
            self._gz_run(gz_model.remove_command(GZ_WORLD, name))
            created = self._gz_run(gz_model.create_command(GZ_WORLD, gesture_model_sdf(index, raised)))
            if "true" in created.lower():
                return
            time.sleep(.5 * (attempt + 1))
        self.rendered_human_gestures.pop(index, None)
        self.get_logger().error(f"No se pudo cambiar la pose visible de {name}")

    def _gz_spawn_scene(self):
        """Crea la bolsa, la persona y los clientes cuando el mundo ya corre."""
        if getattr(self, "receptionist_humans", None) is not None:
            time.sleep(3.0)
            self.receptionist_humans.spawn()
            return
        models = []
        from base_athome_sim.restaurant_humans import rendered_restaurant
        if not (rendered_restaurant() or GZ_WORLD == "restaurant"):
            if self.rigid_object and self.bag_map is not None:
                models.append(gz_model.table_object_sdf(self.bag_map))
            elif self.bag_map is not None:
                models.append(gz_model.bag_sdf(self.bag_map[:2], self.bag_top))
        if self.person_xy is not None and self._scene_person_enabled():
            models.append(gz_model.person_sdf(self.person_xy))
        if self.auto_guest:
            models.append(gz_model.customer_sdf("recep_guest", self.person_xy, hand="none", seated=False))
            models.append(gz_model.customer_sdf("recep_host", self.receptionist_host_xy, hand="none", seated=False))
            models.append(gz_model.receptionist_drop_sdf(self.receptionist_drop_xy))
        if self.bag2 is not None:
            models.append(gz_model.bag_sdf(self.bag2[:2], self.bag2[2]).replace(f'name="{gz_model.BAG_MODEL}"', 'name="bolsa_2"')
                          .replace(gz_model.BAG_ODOM_TOPIC, "/sim/gz/bolsa_2/odometry"))
        from base_athome_sim.restaurant_humans import rendered_restaurant
        for i, c in enumerate(self.customers_xy):
            if rendered_restaurant():
                continue  # textured fixed humans already belong to restaurant.sdf
            name = f"cliente_{i+1}"
            models.append(gz_model.customer_sdf(name, c["xy"], hand=c.get("hand", "right"), seated=c.get("seated", False)))
        time.sleep(3.0)
        for sdf in models:
            for _ in range(10):
                if "true" in self._gz_run(gz_model.create_command(GZ_WORLD, sdf)):
                    break
                time.sleep(1.0)
            else:
                self.get_logger().error("No pude crear un modelo en Gazebo")
        self.get_logger().info("Modelos de escena creados en Gazebo")

    def _gz_bag_odom(self, m):
        if time.time() < self.gz_ready_t:
            self.pub_bag_detach.publish(Empty())
        if not self.obj_attached:
            p = m.pose.pose.position
            self.bag_map = [p.x, p.y, p.z if self.rigid_object else p.z + self.bag_top / 2.0]

    def _restaurant_item_odom(self, model_name, msg):
        """Track the can and keep Gazebo's late-spawn default joint from grabbing it."""
        handoff_model = getattr(self, "restaurant_handoff_model", None)
        handoff_age = time.monotonic() - getattr(self, "restaurant_handoff_at", 0.0)
        pending_handoff = model_name == handoff_model and 0.0 <= handoff_age < 10.0
        if not getattr(self, "obj_attached", False) and not pending_handoff:
            pub_detach = self.pub_restaurant_can_detach.get(model_name)
            # Harmonic's DetachableJoint starts with attachRequested=true.  A
            # counter can spawned by the actor minutes after sim startup can
            # therefore attach to link6 unless detached while it is ungrasped.
            if pub_detach:
                pub_detach.publish(Empty())
        p = msg.pose.pose.position
        if not all(math.isfinite(value) for value in (p.x, p.y, p.z)):
            return
        first = model_name not in self.restaurant_item_poses
        with self.lock:
            self.restaurant_item_poses[model_name] = (time.monotonic(), (p.x, p.y, p.z))
        if first:
            self.get_logger().info(f"Odom de item mostrador registrado: {model_name} en [{p.x:.2f}, {p.y:.2f}, {p.z:.2f}]")

    def _sync_restaurant_item_target(self, tcp=None):
        """Expose the freshest actor-created counter can to the simulated gripper model."""
        if not self.gz or not self.restaurant_item_poses:
            return
        now = time.monotonic()
        with self.lock:
            candidates = [(name, pose) for name, (received, pose) in self.restaurant_item_poses.items()
                          if now - received <= 2.0]
        if not candidates:
            if not getattr(self, "obj_attached", False) and not os.environ.get("ATHOME_SIM_TABLE_OBJECT"):
                self.restaurant_item_model = None
                self.bag_map = None
                self.rigid_object = False
                self.restaurant_item_contact = False
            return
        c, s = math.cos(self.base.yaw), math.sin(self.base.yaw)
        def distance(candidate):
            _, (x, y, z) = candidate
            dx, dy = x - self.base.x, y - self.base.y
            local = (c * dx + s * dy, -s * dx + c * dy, z)
            return math.dist(local, tcp) if tcp is not None else math.hypot(dx, dy)
        name, pose = min(candidates, key=distance)
        self.restaurant_item_model = name
        self.bag_map = list(pose)
        self.rigid_object = True

    def _gz_people(self):
        """Publica los puntos del cuerpo y muestra en Gazebo el brazo con el que senala."""
        people = []
        person = self._person()
        if person is not None:
            people.append(person)
        people.extend(self._customers())
        self.pub_people.publish(String(data=json.dumps(dict(people=people))))
        if person is None:
            return
        pointing = self.person_pointing and not self.obj_attached and not self.bag_delivered
        if pointing != self.gz_arm_shown:
            self.gz_arm_shown = pointing
            target = (self.bag_map[0] + self.point_err, self.bag_map[1] - self.point_err)
            kp = sim_scene.make_person(person["xy"], (self.base.x, self.base.y), target)["keypoints"]
            threading.Thread(target=self._gz_show_arm, args=(kp[6], kp[10], pointing), daemon=True).start()
        if (self.person_sway or self.walk_start is not None) and not self.gz_moving:
            self.gz_moving = True
            threading.Thread(target=self._gz_move_person, args=(self._person(ahead=True), pointing and self.gz_arm_ready),
                             daemon=True).start()

    def _gz_move_person(self, person, with_arm):
        """Lleva el cuerpo visible, y el brazo, a donde estan sus puntos."""
        try:
            self._gz_run(gz_model.set_pose_command(GZ_WORLD, "persona", (person["xy"][0], person["xy"][1], 0.81)))
            self.gz_shown_xy = list(person["xy"])
            if with_arm:
                kp = person["keypoints"]
                center, _, pitch, yaw = gz_model.arm_with_hand(kp[6], kp[10])
                self._gz_run(gz_model.set_pose_command(GZ_WORLD, "brazo_persona", center, pitch, yaw))
            time.sleep(0.3)
        finally:
            self.gz_moving = False

    def _gz_show_arm(self, shoulder, wrist, pointing):
        self.gz_arm_ready = False
        self._gz_run(gz_model.remove_command(GZ_WORLD, "brazo_persona"))
        if pointing:
            self._gz_run(gz_model.create_command(GZ_WORLD, gz_model.arm_sdf("brazo_persona", shoulder, wrist)))
            time.sleep(1.0)      # que llegue a los cuadros de la camara
            self.gz_arm_ready = True

    def _gz_tick(self, now):
        v = w = 0.0
        if self.restaurant_base_hold:
            self.base.goal = None
            self.cmd_vel = (0.0, 0.0, 0.0)
            hold_cmd = Twist()
            if self.restaurant_base_hold_yaw is not None:
                hold_cmd.angular.z = restaurant_base_hold_angular_command(
                    self.base.yaw, self.restaurant_base_hold_yaw)
            self.gz_cmd.publish(hold_cmd)
            return
        if self.receptionist_follow_active:
            cmd = Twist()
            self.gz_cmd.publish(cmd)
            return
        g = self.base.goal
        if g is not None:
            dx, dy = g[0] - self.base.x, g[1] - self.base.y
            d = math.hypot(dx, dy)
            wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
            if d > 0.08:
                err = wrap(math.atan2(dy, dx) - self.base.yaw)
                w = max(-1.0, min(1.0, 2.0 * err))
                v = min(self.base.speed, 0.8 * d) if abs(err) < 0.5 else 0.0
            else:
                err = wrap(g[2] - self.base.yaw)
                w = max(-1.0, min(1.0, 2.0 * err))
                if abs(err) < 0.05:
                    self.base.goal, w = None, 0.0
        elif now - self.cmd_vel[2] < 0.5:
            v, w = self.cmd_vel[0], self.cmd_vel[1]
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.gz_cmd.publish(cmd)

    def _gz_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.base.x, self.base.y = p.x, p.y
        self.base.yaw = 2 * math.atan2(q.z, q.w)
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = "odom", "base_footprint"
        self.pub_odom.publish(m)

    def _restamp(self, m, frame):
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), frame
        return m

    def _gz_scan(self, m):
        self.pub_scan.publish(self._restamp(m, "laser"))

    def _gz_info(self, m):
        self.pub_info.publish(self._restamp(m, "camera_color_optical_frame"))

    def _gz_wrist(self, m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else img
        self.pub_wrist.publish(to_image(np.ascontiguousarray(bgr), "bgr8", self.get_clock().now().to_msg(), "arm_camera_optical_frame"))

    def _gz_head(self, m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else img.copy()
        self._overlay_guest(bgr, None)
        self.pub_rgb.publish(to_image(bgr, "bgr8", self.get_clock().now().to_msg(), "camera_color_optical_frame"))

    def _gz_depth(self, m):
        d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        mm = np.nan_to_num(d * 1000.0, nan=0.0, posinf=0.0, neginf=0.0).clip(0, 65535).astype(np.uint16)
        self._overlay_guest(None, mm)
        self.pub_depth.publish(to_image(mm, "16UC1", self.get_clock().now().to_msg(), "camera_color_optical_frame"))

    def _arm_status(self):
        # 20 Hz como el controlador real (50 Hz): a 2 Hz los consumidores lo veian "viejo" (teleop sin feedback)
        wall_now = time.monotonic()
        if self.arm_status_period_last is not None and wall_now > self.arm_status_period_last:
            self.arm_status_periods.append(wall_now - self.arm_status_period_last)
            self.arm_status_periods = self.arm_status_periods[-100:]
        self.arm_status_period_last = wall_now
        if wall_now - self.arm_status_period_report_at >= 5.0:
            periods = self.arm_status_periods
            if periods:
                ordered = sorted(periods)
                p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
                self.get_logger().info(
                    f"SIM_ARM_STATUS_PERIOD samples={len(periods)} mean_s={sum(periods) / len(periods):.4f} "
                    f"p95_s={p95:.4f} max_s={max(periods):.4f} effective_hz={len(periods) / sum(periods):.1f}")
            self.arm_status_period_report_at = wall_now
        if self.gz:
            with self.lock:
                feedback = self.gz_arm_feedback
                now = self._gz_feedback_clock()
                age = now - self.gz_arm_feedback_at
                moving = now < self.gz_arm_moving_until
            status = arm_status_from_measured_sample(
                feedback, age, enabled=self.arm.enabled, is_moving=moving)
            if (self.gz and GZ_WORLD == "restaurant" and self.obj_attached
                    and self.restaurant_item_model and status.get("feedback_ok") is True):
                measured_width = status.get("gripper_mm")
                attached_width = 1000.0 * float(self.get_parameter("object_diameter_m").value)
                if not self.restaurant_gripper_attach_reported:
                    self.get_logger().info(
                        f"SIM_RESTAURANT_GRIPPER_ATTACH model={self.restaurant_item_model} "
                        f"measured_width_mm={measured_width} attached={self.obj_attached} "
                        f"reported_width_mm={attached_width:.1f}")
                    self.restaurant_gripper_attach_reported = True
                status = apply_attached_object_gripper_width(
                    status, attached=True, diameter_mm=attached_width)
        else:
            now = self.get_clock().now().to_msg()
            status = {
                "is_moving": self.arm.moving, "enabled": self.arm.enabled, "feedback_ok": True,
                "gripper_mm": round(self.arm.gripper_mm, 2), "xyz_mm": [0.0, 0.0, 0.0],
                "joints_deg": [round(v, 2) for v in self.arm.q], "sim": True,
                "measured_stamp": {"sec": now.sec, "nanosec": now.nanosec},
            }
        self.pub_arm_status.publish(String(data=json.dumps(status)))

    def _restaurant_can_attached(self, model_name):
        """Mirror Gazebo's detachable-joint attach request in Restaurant sim feedback."""
        if not self.gz or GZ_WORLD != "restaurant":
            return
        self.restaurant_item_model = model_name
        self.obj_attached = True
        self.rigid_object = True
        self.restaurant_item_contact = True
        self.restaurant_gripper_attach_reported = False

    def _restaurant_can_detached(self, model_name):
        """Clear the synthetic can state after Gazebo detaches it for handoff."""
        if not self.gz or GZ_WORLD != "restaurant" or self.restaurant_item_model != model_name:
            return
        self.obj_attached = False
        self.restaurant_item_contact = False
        self.restaurant_gripper_attach_reported = False
        self.restaurant_detached_pending_release = True
        self.gripper_stall.clear()
        self.arm.grip_block_mm = None
        # The actor removes this model immediately after publishing detach.  Drop
        # its synthetic pose too, so _gz_tick cannot re-attach it while the
        # gripper is still closed during the spoken countdown.
        self.restaurant_item_model = None
        self.bag_map = None
        self.rigid_object = False
        with self.lock:
            self.restaurant_item_poses.clear()

    def _gz_feedback_clock(self):
        """Use simulation time for measured Gazebo joint freshness at any RTF."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _slow(self):
        stamp = self.get_clock().now().to_msg()
        st = ("LISTENING" if self.mic.active else ("PROCESSING" if self.mic.pending else
                                                   ("MUTED" if self.mic.muted else "IDLE")))
        self.pub_vstatus.publish(String(data=st))
        self.pub_level.publish(Float32(data=0.02 if st == "LISTENING" else 0.0))
        a = PoseWithCovarianceStamped()
        a.header.stamp, a.header.frame_id = stamp, "map"
        a.pose.pose.position.x, a.pose.pose.position.y = self.base.x, self.base.y
        _, _, a.pose.pose.orientation.z, a.pose.pose.orientation.w = yaw_quat(self.base.yaw)
        if not self.real_nav:
            self.pub_amcl.publish(a)
        self._update_object(stamp)
        self._publish_scene(stamp)

    def _fast(self):
        """Odometria y elevador a 20 Hz: el tramo final hasta la bolsa se controla con la odometria."""
        stamp = self.get_clock().now().to_msg()
        if not self.gz:
            o = Odometry()
            o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, "odom", "base_footprint"
            o.pose.pose.position.x, o.pose.pose.position.y = self.base.x, self.base.y
            _, _, o.pose.pose.orientation.z, o.pose.pose.orientation.w = yaw_quat(self.base.yaw)
            self.pub_odom.publish(o)
        js = JointState(name=["torso_lift_joint"], position=[self.torso_pos])
        js.header.stamp = stamp
        self.pub_torso_js.publish(js)
        moving = abs(self.torso_goal - self.torso_pos) > 1e-3
        self.pub_torso_st.publish(String(data=json.dumps(dict(connected=True, fresh=True, calibrated=True,
                                                              position_m=round(self.torso_pos, 4), moving=moving))))

    def _torso_goal(self, msg):
        if math.isnan(msg.data):
            self.torso_goal = self.torso_pos
        else:
            self.torso_goal = max(-0.59, min(0.0, float(msg.data)))
            self.get_logger().info(f"Elevador (sim) hacia {self.torso_goal:+.2f} m")

    def _heard_robot(self, msg):
        """La persona simulada obedece: senala cuando el robot se lo pide."""
        text = msg.data.lower()
        if self.guest_present and "please take a seat" in text:
            seat_xy = [3.5, 1.5] if "left" in text else [3.5, -1.5]
            self.receptionist_seat_xy = seat_xy
            if self.gz:
                threading.Thread(target=self._gz_seat_guest, args=(seat_xy, self.guest_idx + 1), daemon=True).start()
            else:
                self._record_seated_guest(seat_xy, self.guest_idx + 1)
        if "follow you" in text and self.person_walk and self.walk_start is None:
            self.walk_start = time.time()
        if "point" in text and self.person_points:
            self.person_pointing = True
        if "between my fingers" in text and self.bag_map is not None and self.person_hands_over:   # plan B: se la da en la mano
            self.bag_in_hand = True

    def _publish_receptionist_state(self):
        self.pub_receptionist_state.publish(String(data=json.dumps({
            "guest_index": self.guest_idx + 1,
            "seated_guests": self.receptionist_seated_guests,
        })))

    def _record_seated_guest(self, xy, guest_index):
        self.receptionist_seated_guests = [g for g in self.receptionist_seated_guests
                                           if g["guest_index"] != guest_index]
        self.receptionist_seated_guests.append({"guest_index": guest_index, "xy": list(xy)})

    def _gz_seat_guest(self, xy, guest_index):
        """Walk the visible actor to the offered chair, then show its seated pose."""
        if getattr(self, "receptionist_humans", None) is not None:
            self.receptionist_humans.seat(xy, guest_index)
            return
        start = list(self.person_xy)
        for step in range(1, 9):
            t = step / 8.0
            pos = [start[i] + t * (xy[i] - start[i]) for i in range(2)]
            self._gz_run(gz_model.set_pose_command(GZ_WORLD, "recep_guest", (pos[0], pos[1], 0.81)))
            time.sleep(0.2)
        self._gz_run(gz_model.remove_command(GZ_WORLD, "recep_guest"))
        sdf = gz_model.customer_sdf("recep_guest", xy, hand="none", seated=True)
        self._gz_run(gz_model.create_command(GZ_WORLD, sdf))
        self._record_seated_guest(xy, guest_index)
        self._publish_receptionist_state()

    def _gz_reset_reception_guest(self):
        """Bring the next scripted guest back to the reception area before the HRI greeting."""
        if getattr(self, "receptionist_humans", None) is not None:
            self.receptionist_humans.arrive(self.guest_idx + 1)
        else:
            self._gz_run(gz_model.remove_command(GZ_WORLD, "recep_guest"))
            sdf = gz_model.customer_sdf("recep_guest", self.person_xy, hand="none", seated=False)
            self._gz_run(gz_model.create_command(GZ_WORLD, sdf))
        if self.guest_idx >= 1:
            self._gz_run(gz_model.remove_command(GZ_WORLD, self.receptionist_bag_model))
            xyz = (self.person_xy[0] + 0.30, self.person_xy[1] - 0.28, 0.95)
            self._gz_run(gz_model.create_command(
                GZ_WORLD, gz_model.receptionist_bag_sdf(xyz, self.receptionist_bag_model)))
            self.receptionist_bag_state = "ready"
            self._publish_receptionist_task_state()

    def _publish_receptionist_task_state(self):
        self.pub_receptionist_task.publish(String(data=json.dumps({
            "bag_state": self.receptionist_bag_state,
            "host_xy": self.receptionist_host_xy,
            "drop_xy": self.receptionist_drop_xy,
        })))

    def _receptionist_task_cb(self, msg):
        if not self.gz:
            return
        action = msg.data.strip().lower()
        if action == "handover" and self.receptionist_bag_state == "ready":
            threading.Thread(target=self._gz_receptionist_handover, daemon=True).start()
        elif action == "follow" and self.receptionist_bag_state == "received" and not self.receptionist_follow_active:
            threading.Thread(target=self._gz_receptionist_follow_host, daemon=True).start()
        elif action == "drop" and self.receptionist_bag_state == "host_at_drop":
            threading.Thread(target=self._gz_receptionist_drop_bag, daemon=True).start()

    def _gz_move_receptionist_bag_to_gripper(self):
        """Deprecated: retained as no-op; bag is held by physical gripper via DetachableJoint."""
        pass

    def _gz_receptionist_handover(self):
        """Physical simulation handover: position bag handle at gripper, attach via joint."""
        self.receptionist_bag_state = "handover"
        self._publish_receptionist_task_state()
        guest = next((g for g in self.receptionist_seated_guests if g["guest_index"] == 2), None)
        if guest:
            start = [float(guest["xy"][0]) - 0.35, float(guest["xy"][1]), 0.95]
        else:
            start = [self.person_xy[0] + 0.30, self.person_xy[1] - 0.28, 0.95]
        self._gz_run(gz_model.set_pose_command(GZ_WORLD, self.receptionist_bag_model, start))
        time.sleep(0.5)

        # 5.2 El simulador posiciona la manija de la bolsa entre los dedos de la pinza
        tf = self._lookup("map", "tcp_link")
        if tf is not None:
            xyz = tf[1]
            gripper_slot = (float(xyz[0]), float(xyz[1]), round(float(xyz[2]) - 0.05, 3))
        else:
            c, s = math.cos(self.base.yaw), math.sin(self.base.yaw)
            gripper_slot = (self.base.x + 0.45 * c, self.base.y + 0.45 * s, 0.75)
        self._gz_run(gz_model.set_pose_command(GZ_WORLD, self.receptionist_bag_model, gripper_slot))
        self.receptionist_bag_state = "at_gripper"
        self._publish_receptionist_task_state()
        time.sleep(1.0)

        # Fijar mecánicamente a la pinza mediante joint dinámico (DetachableJoint)
        if hasattr(self, "pub_bag_attach"):
            self.pub_bag_attach.publish(Empty())
        if hasattr(self, "pub_recep_bag_attach"):
            self.pub_recep_bag_attach.publish(Empty())
        self.receptionist_bag_held = True
        self.obj_attached = True
        self.receptionist_bag_state = "received"
        self._publish_receptionist_task_state()
        self.get_logger().info("Receptionist sim: bolsa recibida y acoplada a la pinza mediante detachable joint")

    def _gz_receptionist_follow_host(self):
        """Walk the visible host while the robot follows reactively via cmd_vel."""
        if getattr(self, "receptionist_humans", None) is not None:
            self.receptionist_humans.follow_host()
            return
        self.receptionist_bag_state = "following"
        self.receptionist_follow_active = True
        self.receptionist_host_path_done = False
        self._publish_receptionist_task_state()
        if hasattr(self, "pub_voice"):
            self.pub_voice.publish(String(data="Follow me"))
        start = tuple(self.receptionist_host_xy)
        self.base.goal = None
        path = [start, (2.60, 1.90), (2.60, 0.45), tuple(self.receptionist_drop_xy)]
        for a, b in zip(path, path[1:]):
            distance = math.dist(a, b)
            steps = max(2, int(distance / 0.45))
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            for step in range(1, steps + 1):
                t = step / steps
                xy = [a[i] + t * (b[i] - a[i]) for i in range(2)]
                self.receptionist_host_xy = xy
                self.receptionist_host_yaw = yaw
                self._gz_run(gz_model.set_pose_command(
                    GZ_WORLD, "recep_host", (xy[0], xy[1], 0.81), yaw=yaw))
                time.sleep(0.05)
        self.receptionist_host_path_done = True
        self.receptionist_follow_active = False
        self.receptionist_bag_state = "host_at_drop"
        self._publish_receptionist_task_state()
        if hasattr(self, "pub_voice"):
            self.pub_voice.publish(String(data="You can drop the bag here."))
        self.get_logger().info("Receptionist sim: anfitrión llegó a la zona de entrega y autoriza soltar la bolsa")

    def _gz_move_receptionist_base(self, xy, yaw):
        """Deprecated: retained as no-op; robot base motion is governed purely by cmd_vel."""
        pass

    def _gz_receptionist_drop_bag(self):
        self.receptionist_bag_held = False
        self.obj_attached = False
        if hasattr(self, "pub_bag_detach"):
            self.pub_bag_detach.publish(Empty())
        if hasattr(self, "pub_recep_bag_detach"):
            self.pub_recep_bag_detach.publish(Empty())
        self._gz_run(gz_model.set_pose_command(
            GZ_WORLD, self.receptionist_bag_model,
            (self.receptionist_drop_xy[0], self.receptionist_drop_xy[1], 0.15)))
        self.receptionist_bag_state = "dropped"
        self._publish_receptionist_task_state()
        self.get_logger().info("Receptionist sim: bolsa colocada en la zona marcada")

    def _walked_to(self, elapsed, start):
        """Camina por los puntos de ATHOME_SIM_PERSON_WALK a velocidad de persona, y se queda en el ultimo."""
        left, here = self.PERSON_WALK_MPS * elapsed, list(start)
        for stop in self.person_walk:
            d = math.hypot(stop[0] - here[0], stop[1] - here[1])
            if left < d:
                return [here[0] + (stop[0] - here[0]) * left / d, here[1] + (stop[1] - here[1]) * left / d]
            left -= d
            here = list(stop)
        return here

    def _scene_person_enabled(self):
        """La persona de la mision de la bolsa: solo si se pide (ATHOME_SIM_PERSON) o si hay bolsa sin clientes.

        En Restaurant (ATHOME_SIM_CUSTOMERS) el objeto de la barra tambien usa ATHOME_SIM_BAG; sin esta guarda aparecia
        una tercera "persona" (un prisma) en el medio del salon, publicada en /sim/people y apuntando.
        """
        return "ATHOME_SIM_PERSON" in os.environ or (self.bag_map is not None and not self.customers_xy)

    def _person(self, ahead=False):
        """La persona ahora: mira al robot y, si corresponde, apunta a la bolsa con el error de una persona real."""
        if not self._scene_person_enabled():
            return None
        xy = list(self.person_xy)
        if self.person_sway:
            xy[1] += self.person_sway * math.sin(2.0 * math.pi * time.time() / 6.0)
        if self.walk_after_s and self.person_walk and self.walk_start is None and time.time() - self.t_start > self.walk_after_s:
            self.walk_start = time.time()
        if self.walk_start is not None:
            xy = self._walked_to(time.time() - self.walk_start, xy)
        if not ahead and self.gz and self.gz_shown_xy is not None and (self.person_sway or self.walk_start is not None):
            xy = list(self.gz_shown_xy)     # los puntos del cuerpo, donde esta el cuerpo visible, no por delante de el
        target = None
        # En Gazebo el brazo visible tarda en aparecer: el gesto cuenta recien cuando la camara lo puede medir
        if self.person_pointing and not self.obj_attached and not self.bag_delivered and (not self.gz or self.gz_arm_ready):
            if self.bag_map is not None:
                target = (self.bag_map[0] + self.point_err, self.bag_map[1] - self.point_err)
        return sim_scene.make_person(xy, (self.base.x, self.base.y), target)

    def _world_faces(self, person):
        faces = sim_scene.person_faces(person) if person else []
        if self.bag_map is not None:
            if self.obj_attached:
                c, s = math.cos(self.base.yaw), math.sin(self.base.yaw)
                xy = (self.base.x + c * self.obj[0] - s * self.obj[1], self.base.y + s * self.obj[0] + c * self.obj[1])
                faces += sim_scene.bag_faces(xy, self.obj[2], center_z=self.obj[2] - 0.15)
            else:
                faces += sim_scene.bag_faces(self.bag_map[:2], self.bag_map[2])
        if self.bag2 is not None:
            faces += sim_scene.bag_faces(self.bag2[:2], self.bag2[2])
        return faces

    def _publish_scene(self, stamp):
        """Persona, gesto y punto de basura, para verlos en RViz."""
        if self.bag_map is None or not self._scene_person_enabled():
            return
        marks = []

        def mark(mid, kind, frame="map"):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, frame
            m.ns, m.id, m.type, m.action = "sim_bag_scene", mid, kind, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.01   # RViz marca como error una escala 0, aun al borrar
            marks.append(m)
            return m
        person = self._person()
        if person is None:
            return
        px, py = person["xy"]
        body = mark(1, Marker.CYLINDER)
        body.pose.position.x, body.pose.position.y, body.pose.position.z = px, py, 0.8
        body.scale.x, body.scale.y, body.scale.z = 0.35, 0.35, 1.6
        body.color.r, body.color.g, body.color.b, body.color.a = 0.2, 0.5, 0.9, 0.35
        if self.person_pointing and not self.obj_attached and not self.bag_delivered:
            shoulder = person["keypoints"][6]
            ray = mark(2, Marker.LINE_STRIP)
            ray.scale.x = 0.03
            ray.color.r, ray.color.g, ray.color.a = 1.0, 0.9, 1.0
            ray.points = [Point(x=shoulder[0], y=shoulder[1], z=shoulder[2]),
                          Point(x=self.bag_map[0] + self.point_err, y=self.bag_map[1] - self.point_err, z=0.0)]
        else:
            gone = mark(2, Marker.LINE_STRIP)
            gone.action = Marker.DELETE
        for waypoint, wid in (("trash_drop", 3),):
            wp = self._waypoints().get(waypoint)
            if wp:
                t = mark(wid, Marker.CUBE)
                t.pose.position.x, t.pose.position.y, t.pose.position.z = float(wp["x"]) + 0.4, float(wp["y"]), 0.02
                t.scale.x, t.scale.y, t.scale.z = 0.8, 0.8, 0.04
                t.color.g, t.color.r, t.color.a = 0.8, 0.2, 0.6
                label = mark(wid + 1, Marker.TEXT_VIEW_FACING)
                label.pose.position.x, label.pose.position.y, label.pose.position.z = float(wp["x"]), float(wp["y"]), 0.5
                label.scale.z = 0.25
                label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 1.0, 1.0
                label.text = "BASURA"
        self.pub_world.publish(MarkerArray(markers=marks))

    def _lookup(self, target, source):
        try:
            tr = self.tf_buf.lookup_transform(target, source, rclpy.time.Time())
        except Exception:
            return None
        t, r = tr.transform.translation, tr.transform.rotation
        x, y, z, w = r.x, r.y, r.z, r.w
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return R, np.array([t.x, t.y, t.z])

    def _update_object(self, stamp):
        """Agarre simulado: si la pinza se cierra con el TCP a <6 cm del objeto, el objeto la sigue."""
        tcp = self._lookup("base_footprint", "tcp_link")
        self._sync_restaurant_item_target(tcp[1] if tcp is not None else None)
        if self._physical_grasp():
            return
        if tcp is not None:
            # Una bolsa se toma por el nudo, debajo de su parte mas alta, y lo que queda entre los dedos es fino
            bag_like = self.bag_map is not None and not self.rigid_object
            target = np.array(self.obj) - ([0.0, 0.0, self.BAG_KNOT_BELOW_TOP_M] if bag_like else 0.0)
            d = float(np.linalg.norm(tcp[1] - target))
            dia_mm = self.BAG_KNOT_MM if bag_like else 1000.0 * float(self.get_parameter("object_diameter_m").value)
            # Objeto entre los dedos (TCP a < 14 cm en latas/botellas con avance): la pinza no puede cerrar por debajo de su diametro
            hold_dist = 0.14 if (self.restaurant_item_model or self.rigid_object) else 0.06
            holds = d < hold_dist
            if self.grasp_slips and not self.bag_in_hand:
                holds = False          # el nudo se escapa: la pinza cierra vacia
            if self.bag_in_hand and not self.obj_attached:
                holds = True
            contact_mm = dia_mm if (holds or self.obj_attached) else None
            self.arm.grip_block_mm, self.restaurant_detached_pending_release = (
                restaurant_gripper_contact_block(
                    self.arm.gripper_target_mm, contact_mm,
                    item_present=bool(self.restaurant_item_model or self.rigid_object or self.obj_attached),
                    detach_pending_release=self.restaurant_detached_pending_release))
            closing = (self.arm.gripper_cmd_mm <= dia_mm + 2.0) and (self.arm.gripper_mm <= dia_mm + 2.0)
            if self.restaurant_item_model and not closing:
                self.restaurant_item_contact = False
            if closing and not self.gripper_was_closed and self.bag_map is not None:     # la verdad, para medir el error
                self.get_logger().info(f"Pinza cerrada: TCP {np.round(tcp[1], 3).tolist()}, objeto "
                                       f"{np.round(self.obj, 3).tolist()}, a {d * 100:.1f} cm")
            self.gripper_was_closed = closing
            if not self.obj_attached and closing and holds:
                self.obj_attached = True
                model = self.restaurant_item_model or "restaurant_counter_can_1"
                self.restaurant_item_model = model
                if not self.restaurant_item_contact:
                    self.get_logger().info(f"Contacto y agarre de pinza con objeto físico {model}")
                self.restaurant_item_contact = True
                pub_attach = self.pub_restaurant_can_attach.get(model)
                if pub_attach:
                    pub_attach.publish(Empty())
                else:
                    self.get_logger().info("Objeto AGARRADO (sim)")
                    if self.gz and self.bag_map is not None:
                        if self.bag_in_hand:   # la persona la acerco: llevarla a la pinza antes de unirla
                            self._gz_run(gz_model.set_pose_command(GZ_WORLD, gz_model.BAG_MODEL, self._tcp_in_map()))
                        self.pub_bag_attach.publish(Empty())
                self.restaurant_handoff_model = None
                self.restaurant_handoff_at = 0.0
            elif self.obj_attached and (self.arm.gripper_cmd_mm > dia_mm + 5.0 or self.arm.gripper_mm > dia_mm + 10.0):
                self.obj_attached = False
                self.get_logger().info("Objeto SOLTADO (sim)")
                if self.restaurant_item_model:
                    pub_detach = self.pub_restaurant_can_detach.get(self.restaurant_item_model)
                    if pub_detach:
                        pub_detach.publish(Empty())
                if self.gz and self.bag_map is not None:
                    self.pub_bag_detach.publish(Empty())
                if self.bag_map is not None:    # la bolsa cae al piso donde se solto, parada sobre su base
                    c, s = math.cos(self.base.yaw), math.sin(self.base.yaw)
                    self.bag_map = [self.base.x + c * self.obj[0] - s * self.obj[1],
                                    self.base.y + s * self.obj[0] + c * self.obj[1], 0.30]
                    self.bag_delivered = True
                    self.get_logger().info(f"Bolsa en el piso (sim): ({self.bag_map[0]:.2f}, {self.bag_map[1]:.2f})")
            if self.obj_attached:
                self.obj = (tcp[1] + ([0.0, 0.0, self.BAG_KNOT_BELOW_TOP_M] if bag_like else 0.0)).tolist()
        if self.bag_map is not None and not self.obj_attached:
            c, s = math.cos(self.base.yaw), math.sin(self.base.yaw)
            dx, dy = self.bag_map[0] - self.base.x, self.bag_map[1] - self.base.y
            self.obj = [c * dx + s * dy, -s * dx + c * dy, self.bag_map[2]]
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, "base_footprint"
        m.ns, m.id, m.type, m.action = "sim_object", 0, Marker.CYLINDER, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = self.obj
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = float(self.get_parameter("object_diameter_m").value)
        m.scale.z = 0.23
        m.color.r, m.color.g, m.color.b, m.color.a = 0.85, 0.1, 0.85, 1.0
        if self.bag_map is not None and not self.rigid_object:  # bag proxy: grasp point below its top
            m.type = Marker.CUBE
            m.pose.position.z = self.obj[2] - 0.15
            m.scale.x, m.scale.y, m.scale.z = 0.28, 0.28, 0.30
            m.color.r, m.color.g, m.color.b = 0.05, 0.05, 0.05
        self.pub_world.publish(MarkerArray(markers=[m]))

    def _overlay_guest(self, rgb, depth, W=640, H=480, f=615.0):
        """Dibuja al invitado (rostro real de una imagen + cuerpo) donde lo veria la RealSense segun
        el pan-tilt. Se usa igual sobre la imagen sintetica o sobre el render de Gazebo."""
        if (os.environ.get("ATHOME_SIM_REAL_PERCEPTION") == "1"
                or os.environ.get("ATHOME_SIM_RENDERED_HUMANS") == "1"):
            return
        if not (self.guest_present and self.faces):
            return
        face = self.faces[self.guest_idx % len(self.faces)]
        u = W / 2 + f * math.tan(self.pt.pan * D2R)           # pan + izquierda: el invitado se ve a la derecha
        v = H / 2 + f * math.tan((self.pt.tilt - 5.0) * D2R)  # rostro ~5 grados arriba del eje
        size = int(f * 0.45 / self.guest_d)                    # el rostro ocupa ~1/3 de la foto: cara real ~15 cm
        if not (20 < size < H and -size < u < W + size):
            return
        fi = cv2.resize(face, (size, int(size * face.shape[0] / face.shape[1])))
        x0, y0 = int(u - fi.shape[1] / 2), int(v - fi.shape[0] / 2)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + fi.shape[1]), min(H, y0 + fi.shape[0])
        if xb <= xa or yb <= ya:
            return
        bx0, bx1 = max(0, xa - size // 4), min(W, xb + size // 4)
        if rgb is not None:
            rgb[ya:yb, xa:xb] = fi[ya - y0:yb - y0, xa - x0:xb - x0]
            rgb[yb:H, bx0:bx1] = (90, 60, 40)
        if depth is not None:
            depth[ya:H, bx0:bx1] = int(self.guest_d * 1000)

    def _cameras(self):
        stamp = self.get_clock().now().to_msg()
        # --- RealSense de la cabeza (640x480, fx ~615) ---
        W, H, f = 640, 480, 615.0
        person = self._person()
        customers = self._customers()
        all_people = ([person] if person else []) + customers
        head = self._lookup("camera_color_optical_frame", "map") if (self.bag_map is not None or self.customers_xy or "ATHOME_SIM_PERSON" in os.environ) else None
        if head is not None:     # la escena sale de la geometria: la percepcion real corre en lazo cerrado
            self.pub_people.publish(String(data=json.dumps(dict(people=all_people))))
            faces = self._world_faces(person)
            for c in customers:
                faces += sim_scene.person_faces(c)
            rgb, depth = sim_scene.render(head[0], head[1], (f, f, W / 2, H / 2), W, H, faces)
            if self.head_pix_err:
                noise = self.rng.normal(0.0, self.head_pix_err, depth.shape)
                depth = np.where(depth > 0, np.clip(depth + noise, 1, 65535), 0).astype(np.uint16)
        else:
            rgb = np.full((H, W, 3), (70, 75, 80), np.uint8)
            rgb[int(H * 0.62):] = (60, 90, 120)  # piso
            depth = np.full((H, W), 3000, np.uint16)
        self._overlay_guest(rgb, depth)
        self.pub_rgb.publish(to_image(rgb, "bgr8", stamp, "camera_color_optical_frame"))
        self.pub_depth.publish(to_image(depth, "16UC1", stamp, "camera_color_optical_frame"))
        info = CameraInfo(width=W, height=H, distortion_model="plumb_bob")
        info.header.stamp, info.header.frame_id = stamp, "camera_color_optical_frame"
        info.k = [f, 0.0, W / 2, 0.0, f, H / 2, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, W / 2, 0.0, 0.0, f, H / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = [0.0] * 5
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.pub_info.publish(info)
        # --- Camara de muneca: lata magenta proyectada con la pose real del brazo (TF) ---
        wimg = np.full((480, 640, 3), (150, 160, 165), np.uint8)
        cam = self._lookup("arm_camera_optical_frame", "base_footprint")
        wmap = self._lookup("arm_camera_optical_frame", "map") if self.bag_map is not None else None
        if wmap is not None:
            Rcw, tcw = wmap
            if self.wrist_tilt_err:   # el montaje real no es el del modelo: la imagen sale de una camara corrida
                a = self.wrist_tilt_err
                Rcw = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]) @ Rcw
            K = (self.wfx, self.wfx, self.wcx, self.wcy)
            wimg, _ = sim_scene.render(Rcw, tcw, K, 640, 480, self._world_faces(person))
            for poly in (((0, 480), (0, 395), (105, 362), (120, 480)), ((640, 480), (640, 395), (535, 362), (520, 480))):
                cv2.fillConvexPoly(wimg, np.array(poly, np.int32), (25, 25, 25))     # dedos de la garra, como en el robot real
            if self.wrist_ghost:      # falso positivo real: una caja sobre la garra (imagen del robot, 2026-11-05)
                cv2.rectangle(wimg, (30, 400), (90, 470), sim_scene.BAG_BGR, -1)
        elif cam is not None and not self.obj_attached:
            pc = cam[0] @ np.array(self.obj) + cam[1]
            if pc[2] > 0.05:
                u, v = self.wcx + self.wfx * pc[0] / pc[2], self.wcy + self.wfx * pc[1] / pc[2]
                hw, hh = self.wfx * 0.075 / pc[2] / 2, self.wfx * 0.23 / pc[2] / 2
                cv2.rectangle(wimg, (int(u - hw), int(v - hh)), (int(u + hw), int(v + hh)), (200, 30, 200), -1)
        self.pub_wrist.publish(to_image(wimg, "bgr8", stamp, "arm_camera_optical_frame"))

    def _scan(self):
        s = LaserScan()
        s.header.stamp, s.header.frame_id = self.get_clock().now().to_msg(), "laser"
        s.angle_min, s.angle_max, s.angle_increment = -2.356, 2.356, 0.5 * D2R
        s.range_min, s.range_max, s.scan_time = 0.06, 5.0, 0.1
        n = int(round((s.angle_max - s.angle_min) / s.angle_increment)) + 1
        if self.map is None:
            s.ranges = [float("inf")] * n
        else:
            mp = self.map
            ang = self.base.yaw + s.angle_min + np.arange(n) * s.angle_increment
            rs = np.arange(0.06, 5.0, mp["res"] / 2)
            xs = self.base.x + np.outer(np.cos(ang), rs)
            ys = self.base.y + np.outer(np.sin(ang), rs)
            ci = ((xs - mp["ox"]) / mp["res"]).astype(int)
            ri = ((ys - mp["oy"]) / mp["res"]).astype(int)
            h, w = mp["occ"].shape
            inside = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
            hit = np.zeros_like(inside)
            hit[inside] = mp["occ"][ri[inside], ci[inside]]
            first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
            s.ranges = [float(rs[k]) if k >= 0 else float("inf") for k in first]
        self.pub_scan.publish(s)

    def _gz_rear_scan(self, msg):
        """Forward the independent Gazebo rear lidar used only by simulation docking."""
        if msg.header.frame_id != "restaurant_rear_laser":
            return
        if real_gap.rear_scan_blocked(os.environ):
            return  # el robot real no tiene scan trasero
        self.pub_restaurant_rear_scan.publish(msg)


def main():
    rclpy.init()
    node = SimRobot()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
