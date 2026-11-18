#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulación aislada y rápida de la escena de agarre en el mostrador del barman.

Spawnea al robot ya acoplado frente al mostrador (x=-1.30, y=0.0, yaw=180°),
con la lata colocada en el mostrador, e itera la secuencia de pregrasp,
barrido de joint 5, detección y agarre en 15-20 segundos.
"""
import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORLD_NAME = "restaurant"


def depth_array_from_image(msg):
    """Decode Gazebo's co-registered wrist depth image to meters, rejecting unknown encodings."""
    import numpy as np

    if msg.encoding == "32FC1":
        dtype, scale = np.float32, 1.0
    elif msg.encoding == "16UC1":
        dtype, scale = np.uint16, 0.001
    else:
        return None
    values = np.frombuffer(msg.data, dtype=dtype)
    if values.size != int(msg.height) * int(msg.width):
        return None
    depth = values.reshape((int(msg.height), int(msg.width))).astype(np.float32) * scale
    depth[~np.isfinite(depth) | (depth <= 0.03) | (depth > 5.0)] = np.nan
    return depth


def median_depth_in_box(depth, box, mask):
    """Read valid co-registered depth only from RGB-mask pixels inside the detection box."""
    import numpy as np

    if depth is None or mask is None or depth.shape != mask.shape:
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, x2 = max(0, x1), min(depth.shape[1], x2)
    y1, y2 = max(0, y1), min(depth.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    selected = roi[(mask[y1:y2, x1:x2] > 0) & np.isfinite(roi)]
    return float(np.median(selected)) if selected.size >= 10 else None


def focal_length_from_camera_info(info):
    """Return a valid horizontal pinhole focal length from ROS/Gazebo CameraInfo."""
    import math

    matrix = getattr(info, "k", None)
    if matrix is None or len(matrix) < 9:
        return None
    try:
        fx = float(matrix[0])
    except (TypeError, ValueError, OverflowError):
        return None
    return fx if math.isfinite(fx) and fx > 0 else None


def stop_process(proc, timeout_s=5.0):
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            print(f"ERROR: proceso {proc.pid} no terminó tras SIGTERM; se deja identificado y sin SIGKILL")
            return False
        except ProcessLookupError:
            return True
    return True


def wait_services(env, services, timeout_s=30.0):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            res = subprocess.run(["ros2", "service", "list"], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True, timeout=5)
            if res.returncode == 0:
                available = set(res.stdout.splitlines())
                if set(services).issubset(available):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Debug rapido de agarre en el mostrador")
    parser.add_argument("--gui", type=int, default=1, help="1 para mostrar Gazebo GUI en DISPLAY, 0 headless")
    parser.add_argument("--domain", type=int, default=42, help="Dominio ROS aislado")
    parser.add_argument("--can-y", type=float, default=0.0, help="Coordenada Y de la lata en mundo")
    parser.add_argument("--start-x", type=float, default=-0.90,
                        help="Inicio 35 cm atras de la pose 49ae234 (-1.25, 0, 180)")
    parser.add_argument("--start-yaw", type=float, default=180.0,
                        help="Orientacion de la pose de agarre 49ae234")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout maximo en segundos")
    parser.add_argument("--hold", type=float, default=20.0, help="Segundos para sostener la lata en pantalla para observacion visual")
    parser.add_argument("--retreat-m", type=float, default=0.20,
                        help="Retroceso corto tras confirmar elevacion (0.15 a 0.30 m)")
    parser.add_argument("--tag", default="agarre_debug", help="Prefijo para evidencia de camara en sim_lab/logs")
    args = parser.parse_args()
    if not 0.15 <= args.retreat_m <= 0.30 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        print("ERROR: --retreat-m debe estar entre 0.15 y 0.30 m y --tag solo usar letras/numeros/_/-")
        return 2

    env = dict(
        os.environ,
        ROS_DOMAIN_ID=str(args.domain),
        ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
        SIM_DOMAIN=str(args.domain),
        GZ_PARTITION=os.environ.get("GZ_PARTITION", f"counter_debug_d{args.domain}_{os.getpid()}"),
        GZ_IP="127.0.0.1",
        SIM="1",
        ATHOME_SIM="1",
        ARM="1",
        ATHOME_SIM_ARM_MOTION="1",
        ATHOME_SIM_MOVEIT="1",
        ATHOME_SIM_START=f"{args.start_x:.2f},0.0,{args.start_yaw:.2f}",
        ATHOME_SIM_WORLD="restaurant.sdf",
        ATHOME_SIM_OBJECT_XYZ=(
            f"{math.cos(math.radians(args.start_yaw)) * (-1.78 - args.start_x):.2f},"
            f"{-math.sin(math.radians(args.start_yaw)) * (-1.78 - args.start_x) - float(args.can_y):.2f},1.015"
        ),
        ATHOME_SIM_POSES=str(REPO / "src/base_athome_perception/config/arm_poses_and_limits.yaml"),
        DISPLAY=os.environ.get("SIM_DISPLAY", os.environ.get("DISPLAY", ":1")),
    )

    # Inyectar pythonpath del repo local y de vision
    venv_site = os.path.expanduser("~/ros2_vision_venv/")+"lib/python3.12/site-packages"
    repo_paths = f"{REPO}/simulation/scripts:{REPO}/simulation/base_athome_sim:{REPO}/src/base_athome_perception:{REPO}/src/base_athome_skills"
    curr_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_paths}:{venv_site}:{curr_pypath}" if curr_pypath else f"{repo_paths}:{venv_site}"

    # Inyectar modelos de humanos para Gazebo en restaurant.sdf
    human_models = str(REPO / "simulation/base_athome_sim/models/humans")
    curr_gz_res = env.get("GZ_SIM_RESOURCE_PATH", "")
    env["GZ_SIM_RESOURCE_PATH"] = f"{human_models}:{curr_gz_res}" if curr_gz_res else human_models

    children = []
    print(f"\n============================================================")
    print(f"DEBUG RAPIDO MOSTRADOR (Dominio {args.domain}, Can Y={args.can_y})")
    print(f"============================================================")

    t0 = time.time()
    try:
        # 1. Lanzar Gazebo con el robot ya acoplado al mostrador
        print(f"[1/5] Iniciando Gazebo con robot en x={args.start_x:.2f}, y=0.0, yaw={args.start_yaw:.1f}° (35 cm atras de 49ae234)...")
        gz_cmd = [
            "ros2", "launch", "base_athome_sim", "sim.launch.py",
            "gazebo:=true",
            "world:=restaurant.sdf",
            f"gz_gui:={'true' if args.gui else 'false'}",
            "rviz:=false",
        ]
        gz_proc = subprocess.Popen(gz_cmd, cwd=REPO, env=env, start_new_session=True)
        children.append(gz_proc)
        rviz_config = Path.home() / "sim_lab/tools/agarre_camaras.rviz"
        if args.gui:
            if rviz_config.is_file():
                rviz_proc = subprocess.Popen(["rviz2", "-d", str(rviz_config)], cwd=REPO,
                                             env=env, start_new_session=True)
                children.append(rviz_proc)
                print(f"RViz de camaras lanzado con {rviz_config}")
            else:
                print(f"ERROR: falta la configuracion RViz solicitada: {rviz_config}")

        # Esperar arranque de topico de reloj o lidar
        print("      Esperando /clock y servicios basicos de Gazebo...")
        clock_ok = False
        for i in range(20):
            try:
                res = subprocess.run(["ros2", "topic", "echo", "/clock", "--once"],
                                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0)
                if res.returncode == 0:
                    clock_ok = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
        print(f"      Gazebo reloj listo: {clock_ok}")

        # 2. Spawn de la lata en el mostrador
        print(f"[2/5] Spawneando lata en mostrador (-1.78, {args.can_y:.2f}, 1.015)...")
        from base_athome_sim import gz_model
        can_sdf = gz_model.restaurant_can_sdf((-1.78, args.can_y, 1.015), name="restaurant_counter_can_1")

        # Esperar a que el servicio /world/<world>/create este disponible
        print(f"      Esperando servicio /world/{WORLD_NAME}/create en Gazebo...")
        svc_ok = False
        for _ in range(30):
            try:
                res = subprocess.run(["gz", "service", "-l"], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3)
                if f"/world/{WORLD_NAME}/create" in res.stdout:
                    svc_ok = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
        print(f"      Servicio create listo: {svc_ok}")

        create_res = None
        for attempt in range(3):
            create_res = subprocess.run(
                gz_model.create_command(WORLD_NAME, can_sdf),
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
            )
            if "true" in (create_res.stdout or "").lower():
                break
            time.sleep(1.0)
        print(f"      Lata creada: {'ok' if ('true' in (create_res.stdout.lower() if create_res else '')) else 'error'} (out={create_res.stdout.strip() if create_res else ''}, err={create_res.stderr.strip() if create_res else ''})")
        # Asegurar que la lata no nazca atada al brazo por DetachableJoint
        subprocess.run(["ros2", "topic", "pub", "--once", "/sim/gz/restaurant_counter_can_1/detach", "std_msgs/msg/Empty", "{}"],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0)

        # 3. Lanzar MoveIt 2
        print("[3/5] Iniciando MoveIt 2 (move_group)...")
        moveit_cmd = ["ros2", "launch", "base_athome_moveit_config", "move_group.launch.py"]
        moveit_proc = subprocess.Popen(moveit_cmd, cwd=REPO, env=env, start_new_session=True)
        children.append(moveit_proc)

        # Esperar servicios de MoveIt
        print("      Esperando servicios /compute_ik y /plan_kinematic_path...")
        if not wait_services(env, ["/compute_ik", "/plan_kinematic_path"], timeout_s=25.0):
            print("ERROR: MoveIt no inicio a tiempo.")
            return 1
        print("      MoveIt listo.")

        # Asegurar pose inicial erguida y estable de la lata
        print(f"      Asegurando pose de la lata en (-1.78, {args.can_y:.2f}, 1.015)...")
        subprocess.run(
            gz_model.set_pose_command(WORLD_NAME, "restaurant_counter_can_1", (-1.78, float(args.can_y), 1.015)),
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0
        )
        time.sleep(0.5)

        # 4. Lanzar nodo de manipulacion y ejecutar la secuencia
        print("[4/5] Ejecutando rutina de pregrasp, barrido y agarre...")
        test_script = f"""
import os, rclpy, time, sys, math
from pathlib import Path
from base_athome_perception.arm_manipulation import CONFIG as ARM_CONFIG_PATH, Manipulacion
from base_athome_perception.restaurant_arm import RestaurantArm, RestaurantItemDetector, RestaurantMotionGuard, merge_restaurant_poses
from base_athome_perception import pose_store

rclpy.init()
class ArmTest(RestaurantMotionGuard, Manipulacion):
    def restaurant_permitted(self):
        return True

node = ArmTest()
# Preserve the exact image on which the simulated RGB can mask ran; node.frame
# may already have advanced by the time the blocking harness inspects a target.
for _ in range(100):
    if isinstance(node.detector, RestaurantItemDetector):
        break
    time.sleep(0.01)
# Keep detection on the production sim path so the debug verifies the deployed
# red-can mask and strict component-size checks directly.
_detector_detect = node.detector.detect
def traced_detect(frame):
    detections = _detector_detect(frame)
    if any(str(d[2]).casefold() == 'can' for d in detections):
        import cv2
        cv2.imwrite('/tmp/wrist_counter_detector_input.png', frame)
        node._harness_detection_frame = frame.copy()
        node._harness_detection_boxes = [d[0] for d in detections if str(d[2]).casefold() == 'can']
    return detections
node.detector.detect = traced_detect
# The Gazebo camera is rendered from its TF optical frame. In this harness,
# project the detector's measured pixel/size through that TF instead of relying
# on the real robot's manually calibrated wrist-to-TCP offset.
from base_athome_perception.moveit_bridge import base_link_offset_in_model_frame
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformException
from sim_counter_debug import (
    depth_array_from_image, focal_length_from_camera_info, median_depth_in_box,
)
from rclpy.qos import qos_profile_sensor_data
node._harness_image_stamp = None
node._wrist_detection_publisher = node.create_publisher(
    Image, '/restaurant/wrist_detection/image', qos_profile_sensor_data)
node._harness_camera_info_fx = None
node._harness_camera_info_cx = None
node._harness_camera_info_cy = None
def capture_wrist_camera_info(msg):
    fx = focal_length_from_camera_info(msg)
    if fx is not None:
        node._harness_camera_info_fx = fx
        node.calib.focal_px = fx
        if len(msg.k) >= 9:
            node._harness_camera_info_cx = float(msg.k[2])
            node._harness_camera_info_cy = float(msg.k[5])
            if not getattr(node, '_debug_wrist_info_logged', False):
                print(f'WRIST_CAMERA_INFO fx={{msg.k[0]}} fy={{msg.k[4]}} cx={{msg.k[2]}} cy={{msg.k[5]}}')
                node._debug_wrist_info_logged = True
node.create_subscription(CameraInfo, '/sim/gz/wrist/camera_info', capture_wrist_camera_info,
                         qos_profile_sensor_data)
node._gazebo_wrist_frame = None
def capture_gazebo_wrist(msg):
    import cv2
    import numpy as np
    if not getattr(node, '_debug_wrist_message_logged', False):
        print(f'WRIST_GZ_RX encoding={{msg.encoding}} size={{msg.width}}x{{msg.height}} bytes={{len(msg.data)}}')
        node._debug_wrist_message_logged = True
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size != int(msg.height) * int(msg.width) * 3:
        print(f'WRIST_GZ_REJECT bytes={{raw.size}} expected={{int(msg.height) * int(msg.width) * 3}}')
        return
    image = raw.reshape((int(msg.height), int(msg.width), 3))
    if msg.encoding.casefold() == 'rgb8':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif msg.encoding.casefold() not in ('bgr8', '8uc3'):
        print(f'WRIST_GZ_REJECT unsupported_encoding={{msg.encoding}}')
        return
    node._gazebo_wrist_frame = image.copy()
    node._harness_image_stamp = msg.header.stamp
node.create_subscription(Image, '/sim/gz/wrist', capture_gazebo_wrist,
                         qos_profile_sensor_data)
def capture_wrist_stamp(msg):
    if node._gazebo_wrist_frame is None:
        node._harness_image_stamp = msg.header.stamp
node.create_subscription(Image, '/arm_camera/image_raw', capture_wrist_stamp, qos_profile_sensor_data)
_kinematic_object_point = node._object_point
def camera_tf_object_point(box, z, q_deg):
    try:
        exact_frame = getattr(node, '_harness_detection_frame', None)
        raw_boxes = getattr(node, '_harness_detection_boxes', [])
        if exact_frame is not None and raw_boxes:
            import cv2
            cv2.imwrite('/tmp/wrist_counter_object_point_input.png', exact_frame)
            print(f'VISION_SYNC box_tracker={{tuple(round(float(v), 1) for v in box)}} '
                  f'box_rgb={{tuple(round(float(v), 1) for v in raw_boxes[-1])}}')
        # Use TF at the rendered frame's timestamp. Latest TF can belong to a
        # later arm pose while image detection is still processing, shifting a
        # perfectly visible can by many centimeters during pregrasp motion.
        stamp = node._harness_image_stamp
        query_time = rclpy.time.Time.from_msg(stamp) if stamp is not None else rclpy.time.Time()
        try:
            tr = node.tf_buf.lookup_transform('base_footprint', 'arm_camera_optical_frame', query_time)
        except TransformException:
            if stamp is None:
                raise
            frame_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            age = abs(node.get_clock().now().nanoseconds - frame_ns) / 1e9
            if age > 0.25:
                raise
            tr = node.tf_buf.lookup_transform('base_footprint', 'arm_camera_optical_frame',
                                              rclpy.time.Time())
            if not getattr(node, '_debug_tf_latest_fallback_logged', False):
                print(f'VISION_TF latest fallback age={{age:.3f}}s; image was newer/older than cached TF')
                node._debug_tf_latest_fallback_logged = True
        t, q = tr.transform.translation, tr.transform.rotation
        node._harness_camera_translation = (float(t.x), float(t.y), float(t.z))
        R = __import__('numpy').array([
            [1 - 2*(q.y*q.y + q.z*q.z), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
            [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x*q.x + q.z*q.z), 2*(q.y*q.z - q.x*q.w)],
            [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x*q.x + q.y*q.y)],
        ])
        x1, y1, x2, y2 = box
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Estimate center range from apparent can diameter. Project that center
        # pixel through the wrist CameraInfo and measured optical-frame TF.
        point_camera = __import__('numpy').array([
            (u - node.calib.cx_px) * z / node.calib.focal_px,
            (v - node.calib.cy_px) * z / node.calib.focal_px,
            z,
        ])
        point_footprint = R @ point_camera + __import__('numpy').array([t.x, t.y, t.z])
        return point_footprint - __import__('numpy').array(base_link_offset_in_model_frame())
    except Exception:
        return _kinematic_object_point(box, z, q_deg)
node._object_point = camera_tf_object_point
import threading
spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
spin_thread.start()

def shutdown_node():
    try:
        rclpy.shutdown()
    except Exception:
        pass
    spin_thread.join(timeout=2.0)
    try:
        node.destroy_node()
    except Exception:
        pass

# Cargar poses
config_path = '{REPO}/simulation/base_athome_sim/config/restaurant_arm_poses.yaml'
poses = merge_restaurant_poses(node.poses, config_path)
node.poses = poses
from base_athome_perception.restaurant_arm import RESTAURANT_REQUIRED_POSES, poses_calibrated
print(f'Poses Restaurant calibradas en sim: {{poses_calibrated(poses)}}; estados={{[(name, poses.get(name, {{}}).get("calibrated")) for name in RESTAURANT_REQUIRED_POSES]}}')
if not poses_calibrated(poses):
    raise RuntimeError('Faltan poses Restaurant calibradas en configuracion de simulacion')
if poses.get('restaurant_carry', {{}}).get('calibrated') is not True:
    raise RuntimeError('Falta restaurant_carry calibrada en configuracion de simulacion')

arm = RestaurantArm(lambda: node, poses, lambda: True,
                    unsafe_fn=lambda: print('UNSAFE'),
                    warn_fn=lambda m_txt: print(f'ARM_WARN: {{m_txt}}'))

print('Esperando feedback...')
for _ in range(150):
    if node.feedback_ok and node.motion_ready():
        break
    time.sleep(0.1)
print(f'Feedback listo: ok={{node.feedback_ok}}, motion_ready={{node.motion_ready()}}')

# 1. Asegurar pinza completamente abierta
node.gripper(105.0)

# 2. Inyectar mostrador en MoveIt PlanningScene para evitar colisión con el balcón
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

counter = CollisionObject(id='restaurant_counter_bar')
counter.header.frame_id = 'base_footprint'
box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.60, 3.00, 0.86])
c_pose = Pose()
c_pose.position.x = math.cos(math.radians({args.start_yaw})) * (-2.0 - {args.start_x})
c_pose.position.y = -math.sin(math.radians({args.start_yaw})) * (-2.0 - {args.start_x})
c_pose.position.z = 0.43
c_pose.orientation.w = 1.0
counter.primitives = [box]
counter.primitive_poses = [c_pose]
counter.operation = CollisionObject.ADD

# Add the spawned can to the planning scene as well as the counter bar.
can = CollisionObject(id='restaurant_counter_can_1')
can.header.frame_id = 'base_footprint'
cylinder = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.12, 0.033])
can_pose = Pose()
can_pose.position.x = math.cos(math.radians({args.start_yaw})) * (-1.78 - {args.start_x})
can_pose.position.y = -math.sin(math.radians({args.start_yaw})) * (-1.78 - {args.start_x})
can_pose.position.z = 1.015
can_pose.orientation.w = 1.0
can.primitives = [cylinder]
can.primitive_poses = [can_pose]
can.operation = CollisionObject.ADD

scene = PlanningScene(is_diff=True)
scene.world.collision_objects = [counter, can]
applied = node.moveit._call(node.moveit.cli_apply_scene, ApplyPlanningScene.Request(scene=scene))
print(f'Mostrador y lata en MoveIt PlanningScene: ok={{getattr(applied, "success", False)}}; '
      f'bar={{[round(v, 3) for v in (c_pose.position.x, c_pose.position.y, c_pose.position.z)]}}, '
      f'can={{[round(v, 3) for v in (can_pose.position.x, can_pose.position.y, can_pose.position.z)]}}')

# 3. Primero pregrasp con la base lejos; el acercamiento se hace después.
from base_athome_skills.restaurant_docking import RestaurantDocking
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from base_athome_perception.moveit_bridge import pose_from, quat_from_matrix
node.cb_group = ReentrantCallbackGroup()
node.restaurant_item_pose = None
node.restaurant_item_at = None
def can_odom(msg):
    p = msg.pose.pose.position
    node.restaurant_item_pose = (p.x, p.y, p.z)
    node.restaurant_item_at = time.monotonic()
node.create_subscription(Odometry, '/sim/gz/restaurant_counter_can_1/odometry', can_odom, 5,
                         callback_group=node.cb_group)
docking = RestaurantDocking(node, lambda: True, lambda: True)
arm_result = {{}}
arm_thread = threading.Thread(target=lambda: arm_result.update(ok=arm.present_gripper()), daemon=True)
arm_thread.start()
ready_deadline = time.monotonic() + 8.0
while time.monotonic() < ready_deadline:
    if (docking.pose is not None and docking.odom_at is not None
            and docking.scan is not None and docking.rear_scan is not None):
        break
    time.sleep(0.1)
if docking.pose is None or docking.scan is None:
    print('ERROR: sin odometria o lidar frescos; no se mueve la base')
    shutdown_node()
    sys.exit(2)

base_pub = node.create_publisher(Twist, '/cmd_vel', 5)
state_pub = node.create_publisher(String, '/restaurant/state', 5)
label = 'bottle|can'
node.select_item(label)
can_wait = time.monotonic() + 3.0
while node.restaurant_item_pose is None and time.monotonic() < can_wait:
    time.sleep(0.05)
can_pose0 = node.restaurant_item_pose
print(f'Pose de referencia de lata por odometria Gazebo: {{can_pose0}}')

def fresh_scan():
    with docking.lock:
        scan, rear = docking.scan, docking.rear_scan
        scan_at, rear_at = (scan.received_at if scan else None), (rear.received_at if rear else None)
    now = time.monotonic()
    return (scan if scan and scan_at is not None and now - scan_at <= 0.3 else None,
            rear if rear and rear_at is not None and now - rear_at <= 0.3 else None)

evidence_root = Path.home() / 'sim_lab' / 'logs'
evidence_root.mkdir(parents=True, exist_ok=True)
evidence_count = 0
node._visual_interrupt = False
node._debug_image_published_at = 0.0
from base_athome_perception import wrist_geometry as wg
def current_wrist_frame():
    raw = getattr(node, '_gazebo_wrist_frame', None)
    if raw is not None:
        return raw.copy()
    with node.lock:
        return None if node.frame is None else node.frame.copy()

def publish_detection_image(frame, bbox=None, *, accepted=False, reason="sin deteccion"):
    now = time.monotonic()
    if frame is None or now - node._debug_image_published_at < 0.10:
        return
    import cv2
    image = frame[:, :, :3].copy()
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        color = (0, 255, 0) if accepted else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    expected = getattr(node, '_harness_expected_pixel', None)
    projection = (f" proyeccion=({{expected[0]:.0f}},{{expected[1]:.0f}}) "
                  f"{{'dentro' if expected[2] else 'fuera'}}"
                  if expected is not None else " proyeccion=pendiente")
    label = f"{{'ACEPTADA' if accepted else 'DESCARTADA'}}: {{reason}}{{projection}}"
    cv2.putText(image, label[:100], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (0, 255, 0) if accepted else (0, 0, 255), 1, cv2.LINE_AA)
    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'arm_camera_optical_frame'
    msg.height, msg.width = int(image.shape[0]), int(image.shape[1])
    msg.encoding, msg.is_bigendian, msg.step = 'bgr8', 0, int(image.shape[1]) * 3
    msg.data = image.tobytes()
    node._wrist_detection_publisher.publish(msg)
    node._debug_image_published_at = now

def can_distance_error(box, joints_deg):
    if (can_pose0 is None or docking.pose is None or docking.odom_at is None
            or time.monotonic() - docking.odom_at >= 0.3 or node._harness_image_stamp is None):
        return None
    width = float(box[2]) - float(box[0])
    distance = wg.size_distance(width, node.calib)
    if distance is None or not math.isfinite(float(distance)):
        return None
    obj = node._object_point(box, distance, joints_deg)
    bx, by, yaw = docking.pose
    wx = bx + float(obj[0]) * math.cos(yaw) - float(obj[1]) * math.sin(yaw)
    wy = by + float(obj[0]) * math.sin(yaw) + float(obj[1]) * math.cos(yaw)
    return math.hypot(wx - can_pose0[0], wy - can_pose0[1])

def wrist_can_expected_size_px():
    # Project the known sim can silhouette through fresh wrist TF and CameraInfo.
    if can_pose0 is None or docking.pose is None or node._harness_image_stamp is None:
        return None
    import numpy as np
    stamp = node._harness_image_stamp
    query_time = rclpy.time.Time.from_msg(stamp)
    try:
        tr = node.tf_buf.lookup_transform('base_footprint', 'arm_camera_optical_frame', query_time)
    except TransformException:
        age = abs(node.get_clock().now().nanoseconds -
                  (int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))) / 1e9
        if age > 0.25:
            return None
        tr = node.tf_buf.lookup_transform('base_footprint', 'arm_camera_optical_frame',
                                          rclpy.time.Time())
    t, q = tr.transform.translation, tr.transform.rotation
    rotation = np.asarray([
        [1 - 2*(q.y*q.y + q.z*q.z), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
        [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x*q.x + q.z*q.z), 2*(q.y*q.z - q.x*q.w)],
        [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x*q.x + q.y*q.y)],
    ])
    bx, by, yaw = docking.pose
    radius, half_height = 0.0375, 0.115
    corners = []
    for dx in (-radius, radius):
        for dy in (-radius, radius):
            for dz in (-half_height, half_height):
                wx, wy = can_pose0[0] + dx, can_pose0[1] + dy
                base_point = np.asarray([
                    math.cos(yaw)*(wx-bx) + math.sin(yaw)*(wy-by),
                    -math.sin(yaw)*(wx-bx) + math.cos(yaw)*(wy-by),
                    can_pose0[2] + dz,
                ])
                camera_point = rotation.T @ (base_point - np.asarray([t.x, t.y, t.z]))
                if camera_point[2] <= 0:
                    return None
                fx = node._harness_camera_info_fx or node.calib.focal_px
                cx = node._harness_camera_info_cx or node.calib.cx_px
                cy = node._harness_camera_info_cy or node.calib.cy_px
                corners.append((fx*camera_point[0]/camera_point[2]+cx,
                                fx*camera_point[1]/camera_point[2]+cy))
    pixels = np.asarray(corners)
    return (float(np.ptp(pixels[:, 0])), float(np.ptp(pixels[:, 1])))

def save_wrist_snapshot(stage, frame=None):
    image = frame
    if image is None:
        image = current_wrist_frame()
    if image is None:
        print(f'Wrist sin frame para evidencia en etapa={{stage}}.')
        return None
    import cv2
    path = evidence_root / f'{args.tag}_wrist_{{stage}}.png'
    if cv2.imwrite(str(path), image[:, :, :3]):
        print(f'Evidencia de muñeca: {{path}}; shape={{image.shape}}')
        return path
    print(f'ERROR al guardar evidencia de muñeca: {{path}}')
    return None

def quick_can_visible():
    frame = current_wrist_frame()
    if frame is None:
        return False, None, None
    detections = node.detector._detect_sim_can(frame, wrist_can_expected_size_px())
    bbox = detections[0][0] if detections else None
    accepted = False
    if bbox is None:
        reason = f"size/shape gate: {{node.detector._sim_can_last_rejection}}"
    else:
        with node.lock:
            q_deg = list(node.joints_deg)
        error = can_distance_error(bbox, q_deg)
        accepted = error is not None and error <= 0.12
        reason = (f"size={{bbox[2]-bbox[0]:.0f}}x{{bbox[3]-bbox[1]:.0f}}px; distancia gate={{error:.2f}}m"
                  if error is not None else "distancia gate no disponible")
    publish_detection_image(frame, bbox, accepted=accepted, reason=reason)
    return (accepted, bbox, frame)

def diagnose_can_framing():
    if docking.pose is None or can_pose0 is None or node._harness_image_stamp is None:
        print('Proyeccion TF no disponible: falta pose base/lata o sello de imagen.')
        return
    import numpy as np
    # This framing diagnostic runs after the arm/base are stopped. Gazebo image
    # stamps can lead robot_state_publisher TF by one render frame; querying that
    # future stamp raises ExtrapolationException and hides the projection.
    stamp = node._harness_image_stamp
    query_time = rclpy.time.Time.from_msg(stamp)
    try:
        tr = node.tf_buf.lookup_transform('base_footprint', 'arm_camera_optical_frame',
                                          query_time)
    except TransformException:
        frame_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        age = abs(node.get_clock().now().nanoseconds - frame_ns) / 1e9
        if age > 0.25:
            raise
        # The arm and base are stationary at this diagnostic point, so the
        # latest measured transform is a safe substitute for a just-rendered
        # image timestamp that leads robot_state_publisher by one frame.
        tr = node.tf_buf.lookup_transform('base_footprint', 'arm_camera_optical_frame',
                                          rclpy.time.Time())
        print(f'FRAMING_TF latest fallback age={{age:.3f}}s; arm/base stationary')
    t, q = tr.transform.translation, tr.transform.rotation
    tf_stamp = tr.header.stamp
    rotation = np.asarray([
        [1 - 2*(q.y*q.y + q.z*q.z), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
        [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x*q.x + q.z*q.z), 2*(q.y*q.z - q.x*q.w)],
        [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x*q.x + q.y*q.y)],
    ])
    bx, by, yaw = docking.pose
    dx, dy = can_pose0[0] - bx, can_pose0[1] - by
    point_base = np.asarray([math.cos(yaw)*dx + math.sin(yaw)*dy,
                             -math.sin(yaw)*dx + math.cos(yaw)*dy, can_pose0[2]])
    point_camera = rotation.T @ (point_base - np.asarray([t.x, t.y, t.z]))
    cx = node._harness_camera_info_cx or node.calib.cx_px
    cy = node._harness_camera_info_cy or node.calib.cy_px
    fx = node._harness_camera_info_fx or node.calib.focal_px
    px = (fx * point_camera[0] / point_camera[2] + cx) if point_camera[2] > 0 else float('nan')
    py = (fx * point_camera[1] / point_camera[2] + cy) if point_camera[2] > 0 else float('nan')
    frame = current_wrist_frame()
    width = frame.shape[1] if frame is not None else None
    height = frame.shape[0] if frame is not None else None
    inside = bool(width and height and 0 <= px < width and 0 <= py < height and point_camera[2] > 0)
    node._harness_expected_pixel = (px, py, inside)
    print(f'FRAMING_TF image_stamp={{(node._harness_image_stamp.sec, node._harness_image_stamp.nanosec)}} '
          f'tf_stamp={{(tf_stamp.sec, tf_stamp.nanosec)}} '
          f'base={{tuple(round(float(v), 3) for v in docking.pose)}} '
          f'camera_t={{tuple(round(float(v), 3) for v in (t.x, t.y, t.z))}} '
          f'can_camera={{[round(float(v), 3) for v in point_camera]}} pixel=({{px:.1f}},{{py:.1f}}) '
          f'image={{width}}x{{height}} in_frame={{inside}}')
    publish_detection_image(frame, accepted=False,
                            reason=f"proyeccion esperada ({{px:.0f}},{{py:.0f}}); {{'dentro' if inside else 'fuera'}}")

def turn_to(yaw_target, timeout=12.0, interrupt_on_detection=True):
    end = time.monotonic() + timeout
    msg = Twist()
    try:
        while time.monotonic() < end:
            scan, rear = fresh_scan()
            if scan is None or rear is None or docking.pose is None:
                return False
            # The stock dock rotation guard treats chassis self-returns as obstacles
            # in this Gazebo model. Only reject a close external return in the
            # direction of travel, ignoring the sensor's own <12 cm echo.
            err = math.atan2(math.sin(yaw_target - docking.pose[2]), math.cos(yaw_target - docking.pose[2]))
            if abs(err) <= math.radians(0.7):
                return True
            visible, box, frame = quick_can_visible()
            if visible and interrupt_on_detection:
                node._visual_interrupt = True
                save_wrist_snapshot('primer_frame_valido', frame)
                print('Barrido angular interrumpido en primer frame con deteccion valida.')
                return False
            scan_for_turn = scan if err >= 0 else rear
            sector = [r for a, r in zip(scan_for_turn.angles, scan_for_turn.ranges)
                      if abs(a) <= math.radians(35) and math.isfinite(r)
                      and scan_for_turn.range_min <= r <= scan_for_turn.range_max]
            if sector and min(sector) < 0.24 and min(sector) > 0.12:
                print(f'Giro bloqueado: retorno cercano en el sector (min={{min(sector):.3f}} m).')
                return False
            msg.angular.z = math.copysign(0.08, err)
            base_pub.publish(msg)
            time.sleep(0.05)
        return False
    finally:
        base_pub.publish(Twist())

early_can_seen = False
while arm_thread.is_alive():
    visible, box, frame = quick_can_visible()
    if visible and not early_can_seen:
        early_can_seen = True
        save_wrist_snapshot('durante_pregrasp', frame)
        print(f'Can visible durante pregrasp; se espera a completar la pose, bbox={{box}}.')
    arm_thread.join(timeout=0.05)
arm_ok = bool(arm_result.get("ok"))
if arm_thread.is_alive() or not arm_ok:
    print(f'ARM_DIAG failure={{arm.last_failure!r}}; '
          f'exec={{getattr(node, "last_exec_msg", "")!r}}; '
          f'feedback_ok={{getattr(node, "feedback_ok", None)}}; '
          f'joints_deg={{getattr(node, "joints_deg", None)}}')
    failed_frame = current_wrist_frame()
    if failed_frame is not None:
        publish_detection_image(failed_frame, accepted=False,
                                reason="pregrasp cancelado antes de avanzar")
    save_wrist_snapshot('pregrasp_fallido', failed_frame)
    print(f'ERROR: pregrasp no terminó antes de la búsqueda: arm_ok={{arm_ok}}.')
    shutdown_node()
    sys.exit(2)
if early_can_seen:
    print('Can vista durante pregrasp; se revalidara tras completar el avance protegido.')
    early_can_seen = False
if not turn_to(math.pi):
    if node._visual_interrupt:
        early_can_seen = True
    else:
        print('ERROR: no se pudo alinear yaw a 180 grados con lecturas frescas/guardas activas.')
        shutdown_node()
        sys.exit(2)
start_yaw = docking.pose[2]
pre_dock_visible = False
pre_dock_deadline = time.monotonic() + 3.0
while time.monotonic() < pre_dock_deadline:
    visible, box, frame = quick_can_visible()
    if visible:
        pre_dock_visible = True
        early_can_seen = True
        save_wrist_snapshot('can_detectada_antes_docking', frame)
        print(f'PRE_DOCK_CAN_REFERENCE: visual=True bbox={{box}} odom={{can_pose0}}')
        break
    time.sleep(0.05)
if not pre_dock_visible or can_pose0 is None:
    print('ERROR: no se confirma visualmente la lata con odometria antes del docking; no se avanza.')
    shutdown_node()
    sys.exit(2)

# This simulation-only look pose leaves the wrist camera 30 cm from the can at
# the final dock pose. It is planned before docking, while the can is clear of
# the gripper; post-dock re-detection checks against the saved odometry pose.
look_joints = [66.45, 146.56, -64.80, 98.79, -70.32, -39.56]
look_limit_check = pose_store.validate_joints(look_joints, path=ARM_CONFIG_PATH)
if not look_limit_check.valid:
    print(f'ERROR: restaurant_look excede limites articulares actuales: {{look_limit_check.error}}')
    shutdown_node()
    sys.exit(2)
node.poses['restaurant_look'] = {{
    'joints_deg': look_joints, 'gripper_mm': 105.0,
    'calibrated': True, 'source': 'sim_counter_camera_look_moveit_ik',
}}
look_ok, look_why = node.go_to_blocking('restaurant_look')
if not look_ok:
    print(f'ERROR: no se alcanzo pose restaurant_look con plan MoveIt: {{look_why}}; no se dockea.')
    shutdown_node()
    sys.exit(2)
node.counter_look_pose_active = True
print(f'POSE_LOOK: plan/ejecucion confirmados; joints_deg={{look_joints}}; can_ref_odom={{can_pose0}}')
save_wrist_snapshot('antes_docking_pose_look')
try:
    diagnose_can_framing()
except Exception as exc:
    print(f'Proyeccion TF antes del docking no disponible: {{type(exc).__name__}}: {{exc}}')
print('Ejecutando restaurant_docking con guardas de scan frontal/trasero y odometria frescos...')
if not docking.run():
    print(f'ERROR: restaurant_docking rechazo/fallo; result={{getattr(docking.controller, "result", None)!r}}, '
          f'travel={{docking.last_travel_m!r}}; base sin avance alternativo.')
    shutdown_node()
    sys.exit(2)
base_pub.publish(Twist())
dock_done_at = time.monotonic()
can_wait = time.monotonic() + 2.0
while (node.restaurant_item_pose is None or node.restaurant_item_at is None
       or node.restaurant_item_at < dock_done_at) and time.monotonic() < can_wait:
    time.sleep(0.05)
if (docking.pose is None or docking.odom_at is None or time.monotonic() - docking.odom_at > 0.3
        or node.restaurant_item_pose is None or node.restaurant_item_at is None
        or node.restaurant_item_at < dock_done_at or time.monotonic() - node.restaurant_item_at > 0.3):
    print('ERROR: docking completado pero falta odometria reciente de base/lata para medir alcance.')
    shutdown_node()
    sys.exit(2)
bx, by, dock_yaw = docking.pose
cx, cy, _ = node.restaurant_item_pose
can_forward = (cx - bx) * math.cos(dock_yaw) + (cy - by) * math.sin(dock_yaw)
print(f'DOCK_IK_GATE: can_forward={{can_forward:.3f}} m; requerido 0 < x <= 0.590 m; '
      f'base={{tuple(round(v, 3) for v in docking.pose)}}')
if not 0.0 < can_forward <= 0.59:
    print('ERROR: pose final de restaurant_docking fuera del alcance IK validado; no se intenta el agarre.')
    shutdown_node()
    sys.exit(2)

# Refresh the MoveIt world objects in the final base_footprint frame. The
# initial scene was expressed at the staging pose and would otherwise be stale.
def world_xy_in_base(wx, wy):
    dx, dy = wx - bx, wy - by
    return (math.cos(dock_yaw) * dx + math.sin(dock_yaw) * dy,
            -math.sin(dock_yaw) * dx + math.cos(dock_yaw) * dy)

counter_x, counter_y = world_xy_in_base(-2.0, 0.0)
c_pose.position.x, c_pose.position.y = counter_x, counter_y
c_pose.orientation.z = math.sin(-dock_yaw / 2.0)
c_pose.orientation.w = math.cos(-dock_yaw / 2.0)
can_x, can_y = world_xy_in_base(can_pose0[0], can_pose0[1])
can_pose.position.x, can_pose.position.y, can_pose.position.z = can_x, can_y, can_pose0[2]
scene_after_dock = PlanningScene(is_diff=True)
scene_after_dock.world.collision_objects = [counter, can]
try:
    scene_ok = node.moveit._call(node.moveit.cli_apply_scene,
                                 ApplyPlanningScene.Request(scene=scene_after_dock))
except Exception as exc:
    print(f'ERROR: no se pudo actualizar la escena MoveIt tras docking: {{exc}}')
    shutdown_node()
    sys.exit(2)
if not getattr(scene_ok, 'success', False):
    print('ERROR: MoveIt no acepto la escena del mostrador actualizada tras docking.')
    shutdown_node()
    sys.exit(2)
print(f'MOVEIT_SCENE_AFTER_DOCK: ok=True; counter=({{counter_x:.3f}},{{counter_y:.3f}}); '
      f'can=({{can_x:.3f}},{{can_y:.3f}},{{can_pose0[2]:.3f}})')
save_wrist_snapshot('despues_docking')
try:
    diagnose_can_framing()
except Exception as exc:
    print(f'Proyeccion TF tras docking no disponible: {{type(exc).__name__}}: {{exc}}')
print(f'Docking y alcance confirmados; avance={{docking.last_travel_m:.3f}} m, '
      f'can_antes={{early_can_seen}}; can_forward={{can_forward:.3f}} m.')
base_pose0 = tuple(docking.pose)
start_yaw = docking.pose[2]
camera_deadline = time.monotonic() + 3.0
while node._harness_image_stamp is None and time.monotonic() < camera_deadline:
    time.sleep(0.05)
save_wrist_snapshot('pre_barrido')
try:
    diagnose_can_framing()
except Exception as exc:
    print(f'Proyeccion TF no disponible: {{type(exc).__name__}}: {{exc}}')

def observe_can(timeout=0.55):
    # Measure directly from a snapshot of the displayed wrist frame. The
    # production tracker can retain a detection while its camera has advanced
    # to another frame; that is useful for recovery but unsafe for simulation
    # feedback while the arm/base is moving.
    deadline = time.monotonic() + timeout
    target, previous_box, previous_stamp, stable = None, None, None, 0
    global evidence_count
    while time.monotonic() < deadline:
        frame = current_wrist_frame()
        with node.lock:
            q_det = list(node.joints_deg)
        stamp = node._harness_image_stamp
        if frame is not None:
            evidence_count += 1
            save_wrist_snapshot(f'intento_{{evidence_count:02d}}', frame)
            detections = node.detector._detect_sim_can(frame, wrist_can_expected_size_px())
        else:
            detections = []
        bbox = detections[0][0] if detections else None
        with node.lock:
            q_det = list(node.joints_deg)
        error = can_distance_error(bbox, q_det) if bbox is not None else None
        accepted = error is not None and error <= 0.12
        if bbox is None:
            reason = f"size/shape gate: {{node.detector._sim_can_last_rejection}}"
        else:
            reason = (f"size={{bbox[2]-bbox[0]:.0f}}x{{bbox[3]-bbox[1]:.0f}}px; distancia gate="
                      f"{{error:.2f}}m" if error is not None else "distancia gate no disponible")
        publish_detection_image(frame, bbox, accepted=accepted, reason=reason)
        if bbox is not None and (error is None or error > 0.12):
            time.sleep(0.05)
            continue
        if detections:
            box, _, _ = detections[0]
            box = tuple(float(v) for v in box)
            if (previous_stamp is not None and stamp is not None
                    and (stamp.sec, stamp.nanosec) != previous_stamp
                    and previous_box is not None
                    and max(abs(a - b) for a, b in zip(box, previous_box)) <= 8.0):
                stable += 1
            else:
                stable = 1
            previous_box = box
            previous_stamp = (stamp.sec, stamp.nanosec) if stamp is not None else time.monotonic()
            if stable >= 3:
                x1, y1, x2, y2 = box
                if not getattr(node, '_harness_camera_info_logged', False):
                    print(f"Wrist CameraInfo fx Gazebo={{node._harness_camera_info_fx}}; "
                          f"focal configurada={{node.calib.focal_px}}; "
                          f"diametro sim={{node.calib.object_diameter_m}} m.")
                    node._harness_camera_info_logged = True
                z = wg.size_distance(x2 - x1, node.calib)
                if z is None or not math.isfinite(float(z)):
                    print('Deteccion descartada: ancho aparente de lata invalido.')
                    return None
                obj = node._object_point(box, z, q_det)
                target = dict(obj=obj, z=z, box=box, label='can', t=time.time(), q_deg=q_det)
                node.target, node.detection, node.obj_base = target, (box, z), obj
                break
        time.sleep(0.05)
    if target is not None:
        if frame is not None:
            import cv2
            cv2.imwrite('/tmp/wrist_counter_detection.png', frame)
        # Color-only sim detector can mistake the visitor's green shirt for a can.
        # Keep detections only when their base-frame measurement projects near the
        # independently spawned can odometry; this harness never acts on a mismatch.
        can_reference = node.restaurant_item_pose or can_pose0
        if can_reference is not None and docking.pose is not None:
            ox, oy = float(target['obj'][0]), float(target['obj'][1])
            bx, by, yaw = docking.pose
            wx = bx + ox * math.cos(yaw) - oy * math.sin(yaw)
            wy = by + ox * math.sin(yaw) + oy * math.cos(yaw)
            error = math.hypot(wx - can_reference[0], wy - can_reference[1])
            if error > 0.12:
                box = target.get('box')
                print(f"Deteccion descartada: diferencia {{error:.2f}} m; visual=({{wx:.2f}},{{wy:.2f}}), "
                      f"Gazebo=({{can_reference[0]:.2f}},{{can_reference[1]:.2f}}), "
                      f"local=({{ox:.3f}},{{oy:.3f}}), base=({{bx:.3f}},{{by:.3f}},{{math.degrees(yaw):.1f}}deg), "
                      f"cam_tf={{getattr(node, '_harness_camera_translation', None)}}, "
                      f"box={{box}}, z={{target.get('z')}}; "
                      f"TF frame stamp={{node._harness_image_stamp}}.")
                return None
        print(f"Can visible (3 cuadros estables): xyz={{[round(float(v), 3) for v in target['obj']]}}; "
              f"box={{tuple(round(float(v), 1) for v in target['box'])}}, "
              f"TF stamp={{node._harness_image_stamp}}")
    elif frame is not None:
        # Keep a concrete sensor sample even when the mask never finds the can;
        # this lets a failed sweep be inspected without relying on stale images.
        import cv2
        cv2.imwrite('/tmp/wrist_counter_no_detection.png', frame)
        stamp = node._harness_image_stamp
        print(f"Sin deteccion en frame={{frame.shape}}, stamp={{stamp}}; "
              "imagen guardada en /tmp/wrist_counter_no_detection.png")
    return target

def wait_arm_settled(timeout=12.0):
    # Wait for still feedback before pairing the camera image with TF.
    started = time.monotonic()
    stable_since = None
    with node.lock:
        previous = list(node.joints_deg)
    while time.monotonic() - started < timeout:
        time.sleep(0.10)
        with node.lock:
            current = list(node.joints_deg)
        delta = max((abs(a - b) for a, b in zip(current, previous)), default=0.0)
        if delta < 0.35:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= 0.65:
                print(f'Brazo asentado: max_delta_feedback={{delta:.3f}}°/0.10s')
                return True
        else:
            stable_since = None
        previous = current
    print('Timeout esperando feedback estable del brazo.')
    return False

print('Camara en rumbo actual; si no ve la lata, barrido 40° derecha y luego 40° izquierda...')
offsets = [0] + list(range(5, 41, 5)) + list(range(35, -41, -5))
target = None
for offset_deg in offsets:
    desired = start_yaw + math.radians(offset_deg)
    if offset_deg != 0 and not turn_to(desired):
        if node._visual_interrupt:
            target = observe_can(timeout=1.0)
            if target is not None:
                print(f'Deteccion valida interrumpio barrido a {{offset_deg:+d}} grados.')
                bearing = math.atan2(float(target['obj'][1]), float(target['obj'][0]))
                if not turn_to(docking.pose[2] + bearing, interrupt_on_detection=False):
                    print('No se pudo fijar el rumbo tras deteccion durante el barrido.')
                    target = None
                else:
                    target = observe_can(timeout=1.0) or target
                break
        print(f'Barrido detenido por falta de margen/lectura a {{offset_deg:+d}}°; base parada.')
        break
    target = observe_can()
    if target is not None:
        print(f'Objeto detectado a {{offset_deg:+d}}°; se fija el rumbo hacia el objeto.')
        bearing = math.atan2(float(target['obj'][1]), float(target['obj'][0]))
        if not turn_to(docking.pose[2] + bearing, interrupt_on_detection=False):
            print('No se pudo fijar el rumbo sin perder las guardas de escaneo.')
            target = None
        else:
            target = observe_can(timeout=1.0) or target
        break
    time.sleep(0.10)

print(f'Pregrasp, pose look y docking separados: arm_ok={{arm_ok}}, early_can_seen={{early_can_seen}}; yaw_start={{math.degrees(start_yaw):.1f}}°, yaw_actual={{math.degrees(docking.pose[2]):.1f}}°')
if target is not None:
    obj = target.get('obj')
    offset = base_link_offset_in_model_frame()
    wx = docking.pose[0] + math.cos(docking.pose[2]) * float(obj[0]) - math.sin(docking.pose[2]) * float(obj[1])
    wy = docking.pose[1] + math.sin(docking.pose[2]) * float(obj[0]) + math.cos(docking.pose[2]) * float(obj[1])
    wz = float(obj[2]) + float(offset[2])
    match_error = math.dist((wx, wy, wz), can_pose0)
    print(f'REDETECT_MATCH: measured_odom=({{wx:.3f}},{{wy:.3f}},{{wz:.3f}}) '
          f'reference_odom={{tuple(round(float(v),3) for v in can_pose0)}} '
          f'error={{match_error:.3f}} m; required <=0.030 m')
    if match_error > 0.03:
        print('ERROR: la deteccion visual posterior no coincide con referencia; no se intenta pick.')
        target = None
if target is None or (not arm_ok and not early_can_seen):
    print('RESULTADO: no hubo deteccion visual fija o pregrasp no termino; no se intenta pick.')
    if {args.hold} > 0:
        print(f'Conservando la escena visible {args.hold:.0f}s para inspeccion.')
        time.sleep({args.hold})
    shutdown_node()
    sys.exit(2)

state_pub.publish(String(data='{{"state":"PICKING_ITEM"}}'))
time.sleep(0.25)
settled = wait_arm_settled()
if node.frame is not None:
    import cv2
    cv2.imwrite('/tmp/wrist_counter_debug.png', node.frame)
if not settled:
    print('RESULTADO: brazo sin feedback estable; no se intenta pick_item.')
    shutdown_node()
    sys.exit(3)

# Reuse RestaurantArm's guarded pick_item path with the measured look-pose target.
time.sleep(1.2)
node.select_item(label)
det_ok = arm.detect_item('can', timeout_s=15.0)
print(f'detect_item can: {{det_ok}}; target={{getattr(node, "target", None)}}')
if det_ok and getattr(node, 'target', None):
    obj = node.target.get('obj')
    offset = base_link_offset_in_model_frame()
    wx = docking.pose[0] + math.cos(docking.pose[2]) * float(obj[0]) - math.sin(docking.pose[2]) * float(obj[1])
    wy = docking.pose[1] + math.sin(docking.pose[2]) * float(obj[0]) + math.cos(docking.pose[2]) * float(obj[1])
    wz = float(obj[2]) + float(offset[2])
    match_error = math.dist((wx, wy, wz), can_pose0)
    print(f'PICK_TARGET_MATCH: measured_odom=({{wx:.3f}},{{wy:.3f}},{{wz:.3f}}) '
          f'reference_odom={{tuple(round(float(v),3) for v in can_pose0)}} '
          f'error={{match_error:.3f}} m; required <=0.030 m')
    det_ok = match_error <= 0.03
    if det_ok:
        node.counter_look_pose_active = True
    else:
        print('ERROR: la deteccion de RestaurantArm no coincide con referencia; no se intenta pick.')
pick_ok = False
carry_ok = False
lifted = False
retreat_ok = False
if det_ok and getattr(node, 'target', None):
    print('Ejecutando flujo RestaurantArm.pick_item can...')
    pick_ok = arm.pick_item('can')
    print(f'pick_item can: {{pick_ok}}; failure={{arm.last_failure}}')
else:
    print('RESULTADO: detect_item no confirmó la lata; no se intenta pick_item.')

if pick_ok:
    from base_athome_sim.counter_carry import CounterCarry

    def attached_now():
        pose = node.restaurant_item_pose
        return bool(pose is not None and float(node.gripper_mm) < 100.0)

    def rear_clearance():
        _scan, rear = fresh_scan()
        if rear is None:
            return None
        vals = [r for a, r in zip(rear.angles, rear.ranges)
                if abs(a) <= math.radians(8) and math.isfinite(r)
                and rear.range_min <= r <= rear.range_max]
        return sorted(vals)[len(vals) // 2] if vals else None

    def publish_reverse(linear_x):
        twist = Twist()
        twist.linear.x = float(linear_x)
        base_pub.publish(twist)

    carry = CounterCarry(
        can_pose=lambda: node.restaurant_item_pose,
        base_pose=lambda: docking.pose,
        attached=attached_now,
        rear_clearance=rear_clearance,
        publish_twist=publish_reverse,
        move_holding=lambda name: node.go_to_holding(name),
        initial_can_pose=can_pose0,
        retreat_m={{args.retreat_m}},
        log=lambda msg: print(msg),
    )
    carry_ok, carry_reason, carry_evidence = carry.run()
    lifted = carry_evidence.lift_m >= 0.05
    retreat_ok = carry_evidence.retreat_m >= {{args.retreat_m}}
    pick_ok = bool(pick_ok and carry_ok)
    print(f'CARRY: ok={{carry_ok}}, motivo={{carry_reason}}, evidencia={{carry_evidence}}')
    print(f'Altura validada={{lifted}}; retroceso validado={{retreat_ok}}.')

if pick_ok:
    print(f'RESULTADO: PICK={{pick_ok}} CARRY={{pick_ok}} LIFTED={{lifted}} RETREAT={{retreat_ok}}.')
    print(f'Sosteniendo visualización durante {args.hold:.0f}s...')
    time.sleep({args.hold})
else:
    print('RESULTADO: PICK fallido; se conserva la escena para diagnóstico.')
    print(f'Sosteniendo escena para inspección durante {args.hold:.0f}s...')
    time.sleep({args.hold})

shutdown_node()
time.sleep(1.0)
sys.exit(0 if (det_ok and target is not None and pick_ok and lifted and retreat_ok) else 1)
"""
        py_cmd = ["python3", "-c", test_script]
        test_proc = subprocess.Popen(py_cmd, cwd=REPO, env=env, start_new_session=True)
        children.append(test_proc)
        try:
            result = test_proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"ERROR: rutina pick excedió {args.timeout:.0f}s; solicitando SIGTERM")
            stop_process(test_proc)
            return 124
        print(f"[5/5] Resultado final: exit_code={result} (t={time.time() - t0:.1f}s)")
        return result

    finally:
        print("\nLimpiando procesos hijos...")
        for c in children:
            stop_process(c)
        print("Fin de sesion de debug.")


if __name__ == "__main__":
    sys.exit(main())
