#!/usr/bin/env python3
"""Agrega a robot.urdf (sin modificarlo) los bloques <gazebo> para Gazebo Harmonic (gz sim):

- Base: VelocityControl (sigue /cmd_vel cinematicamente; las ruedas son 'fixed' en el URDF) +
  OdometryPublisher (odom -> base_footprint).
- Brazo, pinza, pan-tilt y elevador: JointPositionController por joint (los comanda sim_robot con
  el mismo modelo temporal que el controlador real) y gravedad apagada en todo el robot.
- Sensores: Hokuyo (gpu_lidar 270 grados, 5 m), RealSense (rgbd 640x480) y camara de muneca.
Asi Gazebo usa EXACTAMENTE la cinematica y las medidas del URDF del robot.
"""

import json
import math
import re
import yaml

MODEL = "base_athome"
POSITION_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8",
                   "pan_joint", "tilt_joint", "torso_lift_joint"]


BAG_MODEL, GRASP_LINK = "bolsa", "link6"
BAG_ATTACH_TOPIC, BAG_DETACH_TOPIC = "/sim/gz/bolsa/attach", "/sim/gz/bolsa/detach"
BAG_ODOM_TOPIC = "/sim/gz/bolsa/odometry"
BAG_SIDE_M = 0.28
HAND_M = 0.15
BAG_SOLID_BELOW_TOP_M = 0.12      # arriba de esto la bolsa es blanda: los dedos entran sin empujarla
GRIPPER_HALF_MAX_M = 0.0525       # limite URDF por dedo; apertura completa 105 mm
# Yema del Piper como caja en Gazebo: el contacto malla-cilindro de STL es inestable en DART.
FINGER_PAD_SIZE_M = (0.056, 0.0765, 0.0265)
FINGER_PAD_CENTER_M = (0.0, -0.03825, 0.01325)
FINGER_LINKS = ("link7", "link8")
RESTAURANT_COUNTER_CAN = "restaurant_counter_can"
RESTAURANT_TRAY_ITEM = "restaurant_tray_item"
RESTAURANT_TRAY_SIDE_M = 0.30
RESTAURANT_TRAY_LINK = "restaurant_tray_link"
RESTAURANT_TRAY_CENTER_X_M = 0.155
RESTAURANT_TRAY_TOP_Z_M = 1.4325  # base_footprint to raised torso platform plus tray floor


def restaurant_item_odometry_topic(name: str) -> str:
    """Gazebo odometry for a dynamically spawned Restaurant item model."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Gazebo model name must be a simple identifier")
    return f"/sim/gz/{name}/odometry"


def restaurant_counter_item_model_names():
    """Model IDs the two customer orders may create at the counter."""
    return tuple(f"{RESTAURANT_COUNTER_CAN}_{idx}" for idx in (1, 2))


def _restaurant_item_odom_plugin(name: str) -> str:
    topic = restaurant_item_odometry_topic(name)
    return f'''<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
    <odom_topic>{topic}</odom_topic><odom_frame>map</odom_frame><robot_base_frame>body</robot_base_frame>
    <odom_publish_frequency>10</odom_publish_frequency><dimensions>3</dimensions></plugin>'''


def restaurant_can_sdf(xyz, name: str = RESTAURANT_COUNTER_CAN) -> str:
    """Unbranded dynamic can sized for the Restaurant counter grasp."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Gazebo model name must be a simple identifier")
    radius, height, mass = 0.0375, 0.23, 0.20
    ixx_iyy = mass * (3 * radius**2 + height**2) / 12.0
    izz = mass * radius**2 / 2.0
    return f'''<?xml version="1.0"?><sdf version="1.9"><model name="{name}">
  <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose><link name="body">
  <inertial><mass>{mass:.6f}</mass><inertia><ixx>{ixx_iyy:.9f}</ixx><iyy>{ixx_iyy:.9f}</iyy><izz>{izz:.9f}</izz>
    <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
  <velocity_decay><linear>1.0</linear><angular>1.0</angular></velocity_decay>
  <collision name="can_collision"><geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
    <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2><slip1>0</slip1><slip2>0</slip2></ode>
      <torsional><ode><slip>0</slip><mu>1.0</mu><use_patch_radius>true</use_patch_radius><patch_radius>0.02</patch_radius></ode></torsional></friction>
      <contact><ode><kp>20000</kp><kd>100.0</kd><max_vel>0.01</max_vel><min_depth>0.001</min_depth></ode></contact></surface></collision>
  <visual name="can_visual"><geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
    <material><diffuse>0.18 0.48 0.32 1</diffuse><ambient>0.10 0.30 0.18 1</ambient><emissive>0.02 0.08 0.04 1</emissive></material></visual>
  </link>{_restaurant_item_odom_plugin(name)}</model></sdf>'''


def restaurant_tray_item_sdf(xyz, name: str = RESTAURANT_TRAY_ITEM) -> str:
    """Small dynamic drink can for loading/unloading on the physical simulation tray."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Gazebo model name must be a simple identifier")
    radius, height, mass = 0.033, 0.10, 0.15
    ixx_iyy = mass * (3 * radius**2 + height**2) / 12.0
    izz = mass * radius**2 / 2.0
    return f'''<?xml version="1.0"?><sdf version="1.9"><model name="{name}">
  <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose><link name="body">
  <inertial><mass>{mass:.6f}</mass><inertia><ixx>{ixx_iyy:.9f}</ixx><iyy>{ixx_iyy:.9f}</iyy><izz>{izz:.9f}</izz></inertia></inertial>
  <collision name="item_collision"><geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
    <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode><torsional><ode><mu>1.0</mu><use_patch_radius>true</use_patch_radius><patch_radius>0.02</patch_radius></ode></torsional></friction>
      <contact><ode><kp>50000</kp><kd>50.0</kd><max_vel>0.01</max_vel><min_depth>0.005</min_depth></ode></contact></surface></collision>
  <visual name="item_visual"><geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
    <material><diffuse>0.80 0.52 0.16 1</diffuse><ambient>0.62 0.36 0.08 1</ambient></material></visual>
  </link>{_restaurant_item_odom_plugin(name)}</model></sdf>'''


def _restaurant_tray_urdf() -> str:
    """Gazebo-only 30 cm square tray fixed to the simulated torso platform."""
    side, floor_h, rim_h, rim_w = RESTAURANT_TRAY_SIDE_M, 0.015, 0.025, 0.012
    inner = side - 2 * rim_w
    floor_box = f'<geometry><box size="{side} {side} {floor_h}"/></geometry>'
    floor_visual = (
        f'<visual name="floor">{floor_box}'
        '<material name="restaurant_tray_floor"><color rgba="0.25 0.28 0.30 1"/>'
        '</material></visual>'
    )
    floor_collision = f'<collision name="floor">{floor_box}</collision>'
    rim_visuals, rim_collisions = [], []
    for name, xyz, size in (
        ('front', (side / 2 - rim_w / 2, 0, floor_h / 2 + rim_h / 2), (rim_w, side, rim_h)),
        ('back', (-side / 2 + rim_w / 2, 0, floor_h / 2 + rim_h / 2), (rim_w, side, rim_h)),
        ('left', (0, side / 2 - rim_w / 2, floor_h / 2 + rim_h / 2), (inner, rim_w, rim_h)),
        ('right', (0, -side / 2 + rim_w / 2, floor_h / 2 + rim_h / 2), (inner, rim_w, rim_h)),
    ):
        pose = f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="0 0 0"/>'
        geom = f'<geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry>'
        rim_visuals.append(
            f'<visual name="rim_{name}">{pose}{geom}'
            f'<material name="restaurant_tray_rim_{name}"><color rgba="0.25 0.28 0.30 1"/>'
            '</material></visual>'
        )
        rim_collisions.append(f'<collision name="rim_{name}">{pose}{geom}</collision>')
    return f'''<link name="{RESTAURANT_TRAY_LINK}"><inertial><mass value="0.15"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  {floor_visual}{floor_collision}{''.join(rim_visuals)}{''.join(rim_collisions)}</link>
  <joint name="restaurant_tray_fixed_joint" type="fixed"><parent link="mounting_cube_link"/><child link="{RESTAURANT_TRAY_LINK}"/><origin xyz="0 0 0.19" rpy="0 0 0"/></joint>
  <gazebo reference="{RESTAURANT_TRAY_LINK}"><gravity>0</gravity><mu1>1.0</mu1><mu2>1.0</mu2><kp>1000000</kp><kd>1.0</kd><max_vel>0.01</max_vel><min_depth>0.001</min_depth></gazebo>
'''


def finger_pad_urdf(urdf: str) -> str:
    """Use box collisions for the Gazebo finger copies; leave the source robot URDF untouched."""
    size = " ".join(str(value) for value in FINGER_PAD_SIZE_M)
    center = " ".join(str(value) for value in FINGER_PAD_CENTER_M)
    pad = (f'<collision name="finger_pad"><origin rpy="0 0 0" xyz="{center}"/>'
           f'<geometry><box size="{size}"/></geometry></collision>')
    for link in FINGER_LINKS:
        block = re.search(rf'<link name="{link}">.*?</link>', urdf, re.S)
        if block is None:
            continue
        fixed, count = re.subn(r'<collision\b.*?</collision>', pad, block.group(0), count=1, flags=re.S)
        if count != 1:
            raise ValueError(f"Finger {link} has no collision to replace")
        urdf = urdf[:block.start()] + fixed + urdf[block.end():]
    return urdf


def _static_box(name, xyz, size, rgb, rpy=(0.0, 0.0, 0.0)):
    color = f"{rgb[0]} {rgb[1]} {rgb[2]} 1"
    return f"""<?xml version="1.0"?><sdf version="1.9"><model name="{name}"><static>true</static>
  <pose>{xyz[0]} {xyz[1]} {xyz[2]} {rpy[0]} {rpy[1]} {rpy[2]}</pose><link name="l"><visual name="v">
  <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
  <material><diffuse>{color}</diffuse><ambient>{color}</ambient></material></visual></link></model></sdf>"""


def bag_sdf(xy, top_z: float) -> str:
    """Bolsa con fisica. Magenta emisiva: es lo que segmenta el detector de simulacion, con cualquier luz."""
    h, side = float(top_z), BAG_SIDE_M
    solid = max(0.05, h - BAG_SOLID_BELOW_TOP_M)
    return f"""<?xml version="1.0"?><sdf version="1.9"><model name="{BAG_MODEL}">
  <pose>{xy[0]} {xy[1]} {h / 2.0 + 0.002} 0 0 0</pose><link name="l">
  <inertial><mass>0.4</mass><inertia><ixx>0.005</ixx><iyy>0.005</iyy><izz>0.005</izz></inertia></inertial>
  <collision name="c"><pose>0 0 {solid / 2.0 - h / 2.0} 0 0 0</pose><geometry><box><size>{side} {side} {solid}</size></box></geometry>
    <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface></collision>
  <visual name="v"><geometry><box><size>{side} {side} {h}</size></box></geometry>
    <material><diffuse>0.8 0.1 0.8 1</diffuse><ambient>0.8 0.1 0.8 1</ambient><emissive>0.78 0.12 0.78 1</emissive></material></visual>
  </link><plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
    <odom_topic>{BAG_ODOM_TOPIC}</odom_topic><odom_frame>map</odom_frame><robot_base_frame>{BAG_MODEL}</robot_base_frame>
    <odom_publish_frequency>10</odom_publish_frequency><dimensions>3</dimensions></plugin></model></sdf>"""


def table_object_sdf(xyz, length: float = 0.23, radius: float = 0.0375) -> str:
    """Botella sin marca (geometria propia), estable sobre la barra para practicar toma YOLO.

    La forma de cuerpo, cuello y tapa es original del simulador; no redistribuye el modelo Coke
    de Fuel, cuya licencia no se pudo verificar.
    """
    body_len = min(length * .78, .18)
    neck_len = length - body_len
    neck_z = body_len / 2 + neck_len / 2
    return f"""<?xml version="1.0"?><sdf version="1.9"><model name="{BAG_MODEL}">
  <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose><link name="l">
  <inertial><mass>0.15</mass><inertia><ixx>0.0007</ixx><iyy>0.0007</iyy><izz>0.0001</izz></inertia></inertial>
  <collision name="body_collision"><pose>0 0 {-neck_len / 2} 0 0 0</pose><geometry><cylinder><radius>{radius}</radius><length>{body_len}</length></cylinder></geometry></collision>
  <collision name="neck_collision"><pose>0 0 {neck_z - neck_len / 2} 0 0 0</pose><geometry><cylinder><radius>{radius * .48}</radius><length>{neck_len}</length></cylinder></geometry></collision>
  <visual name="body_visual"><pose>0 0 {-neck_len / 2} 0 0 0</pose><geometry><cylinder><radius>{radius}</radius><length>{body_len}</length></cylinder></geometry>
    <material><diffuse>0.12 0.42 0.20 1</diffuse><ambient>0.10 0.32 0.16 1</ambient></material></visual>
  <visual name="neck_visual"><pose>0 0 {neck_z - neck_len / 2} 0 0 0</pose><geometry><cylinder><radius>{radius * .48}</radius><length>{neck_len}</length></cylinder></geometry>
    <material><diffuse>0.12 0.42 0.20 1</diffuse><ambient>0.10 0.32 0.16 1</ambient></material></visual>
  <visual name="cap"><pose>0 0 {body_len / 2 + neck_len - .004} 0 0 0</pose><geometry><cylinder><radius>{radius * .50}</radius><length>0.008</length></cylinder></geometry>
    <material><diffuse>0.82 0.82 0.78 1</diffuse><ambient>0.7 0.7 0.66 1</ambient></material></visual>
  </link><plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
    <odom_topic>{BAG_ODOM_TOPIC}</odom_topic><odom_frame>map</odom_frame><robot_base_frame>{BAG_MODEL}</robot_base_frame>
    <odom_publish_frequency>10</odom_publish_frequency><dimensions>3</dimensions></plugin></model></sdf>"""


def person_sdf(xy, name: str = "persona") -> str:
    """Cuerpo de la persona: bloque visual, sin colision, de la altura que usa sim_scene.make_person."""
    return _static_box(name, (xy[0], xy[1], 0.81), (0.52, 0.52, 1.62), (0.2, 0.35, 0.6))


def receptionist_bag_sdf(xyz, name: str = "recepcionista_bolsa") -> str:
    """Small visible simulation prop used for Receptionist handover and delivery."""
    color = "0.45 0.20 0.08 1"
    return f"""<?xml version="1.0"?><sdf version="1.9"><model name="{name}">
  <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose><link name="l">
  <inertial><mass>0.4</mass><inertia><ixx>0.005</ixx><iyy>0.005</iyy><izz>0.005</izz></inertia></inertial>
  <collision name="c"><geometry><box><size>0.3 0.22 0.24</size></box></geometry>
    <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface></collision>
  <visual name="v"><geometry><box><size>0.3 0.22 0.24</size></box></geometry>
  <material><diffuse>{color}</diffuse><ambient>{color}</ambient></material></visual>
  </link><plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
    <odom_topic>/sim/gz/{name}/odometry</odom_topic><odom_frame>map</odom_frame><robot_base_frame>{name}</robot_base_frame>
    <odom_publish_frequency>10</odom_publish_frequency><dimensions>3</dimensions></plugin></model></sdf>"""


def receptionist_drop_sdf(xy, name: str = "recepcionista_drop") -> str:
    """Floor marker for the simulated receptionist bag drop area."""
    return _static_box(name, (xy[0], xy[1], 0.015), (0.65, 0.65, 0.03), (0.15, 0.65, 0.25))


def customer_sdf(name: str, xy, hand: str = "right", seated: bool = False) -> str:
    """Cuerpo visible en Gazebo de un cliente con mano levantada (Restaurant)."""
    h = 1.25 if seated else 1.62
    sh_z = 1.05 if seated else 1.40
    raised_z = h + 0.20
    cz = h / 2.0
    color = "0.2 0.35 0.6 1"
    arm_color = "0.3 0.45 0.7 1"

    arm_len = raised_z - sh_z
    arm_cz = (sh_z + raised_z) / 2.0

    visuals = [
        f'  <visual name="v_cuerpo"><pose>0 0 {cz:.3f} 0 0 0</pose>'
        f'<geometry><box><size>0.52 0.52 {h:.2f}</size></box></geometry>'
        f'<material><diffuse>{color}</diffuse><ambient>{color}</ambient></material></visual>'
    ]

    if hand in ("right", "both"):
        visuals.append(
            f'  <visual name="v_brazo_d"><pose>0 -0.20 {arm_cz:.3f} 0 0 0</pose>'
            f'<geometry><box><size>0.10 0.10 {arm_len:.3f}</size></box></geometry>'
            f'<material><diffuse>{arm_color}</diffuse><ambient>{arm_color}</ambient></material></visual>'
        )
    if hand in ("left", "both"):
        visuals.append(
            f'  <visual name="v_brazo_i"><pose>0 0.20 {arm_cz:.3f} 0 0 0</pose>'
            f'<geometry><box><size>0.10 0.10 {arm_len:.3f}</size></box></geometry>'
            f'<material><diffuse>{arm_color}</diffuse><ambient>{arm_color}</ambient></material></visual>'
        )

    v_str = "\n".join(visuals)
    return f"""<?xml version="1.0"?><sdf version="1.9"><model name="{name}"><static>true</static>
  <pose>{xy[0]} {xy[1]} 0 0 0 0</pose><link name="l">
{v_str}
</link></model></sdf>"""


def arm_pose(shoulder, wrist):
    """(centro, largo, cabeceo, guinada) de una barra entre hombro y muneca, con su eje X sobre el brazo."""
    dx, dy, dz = wrist[0] - shoulder[0], wrist[1] - shoulder[1], wrist[2] - shoulder[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    center = ((shoulder[0] + wrist[0]) / 2.0, (shoulder[1] + wrist[1]) / 2.0, (shoulder[2] + wrist[2]) / 2.0)
    return center, length, math.atan2(-dz, math.hypot(dx, dy)), math.atan2(dy, dx)


def arm_with_hand(shoulder, wrist):
    """Pose del brazo. La mano sigue mas alla de la muneca: sin ella, la profundidad en la muneca cae al fondo."""
    n = math.dist(shoulder, wrist)
    return arm_pose(shoulder, [w + HAND_M * (w - v) / n for v, w in zip(shoulder, wrist)])


def arm_sdf(name, shoulder, wrist) -> str:
    center, length, pitch, yaw = arm_with_hand(shoulder, wrist)
    return _static_box(name, center, (length, 0.10, 0.10), (0.3, 0.45, 0.7), (0.0, pitch, yaw))


def create_command(world: str, sdf: str) -> list:
    return ["gz", "service", "-s", f"/world/{world}/create", "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean", "--timeout", "5000", "--req", "sdf: " + json.dumps(sdf)]


def set_pose_command(world: str, name: str, xyz, pitch: float = 0.0, yaw: float = 0.0) -> list:
    cp, sp, cy, sy = math.cos(pitch / 2.0), math.sin(pitch / 2.0), math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    req = (f'name: "{name}" position: {{x: {xyz[0]} y: {xyz[1]} z: {xyz[2]}}} '
           f"orientation: {{x: {-sp * sy} y: {sp * cy} z: {cp * sy} w: {cp * cy}}}")
    return ["gz", "service", "-s", f"/world/{world}/set_pose", "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000", "--req", req]


def remove_command(world: str, name: str) -> list:
    return ["gz", "service", "-s", f"/world/{world}/remove", "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean", "--timeout", "2000", "--req", f'name: "{name}" type: MODEL']


def joint_cmd_topic(joint: str) -> str:
    return f"/sim/gz/{joint}/cmd_pos"


def _joint_controller(j: str, initial_position: float = 0.0) -> str:
    return f"""  <gazebo><plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
    <joint_name>{j}</joint_name><topic>{joint_cmd_topic(j)}</topic>
    <use_velocity_commands>true</use_velocity_commands><initial_position>{initial_position:.8f}</initial_position>
    <p_gain>8</p_gain><cmd_max>3</cmd_max><cmd_min>-3</cmd_min>
  </plugin></gazebo>
"""


def initial_joint_positions_from_pose_config(path: str, pose_name: str = "home") -> dict:
    """Convert a named, measured arm pose to the scalar initial targets used by Gazebo."""
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    pose = (config.get("named_poses") or {}).get(pose_name)
    joints = pose.get("joints_deg") if isinstance(pose, dict) else None
    width = pose.get("gripper_mm") if isinstance(pose, dict) else None
    if (not isinstance(joints, list) or len(joints) != 6 or width is None
            or not all(math.isfinite(float(value)) for value in joints)
            or not math.isfinite(float(width)) or not 0.0 <= float(width) <= 105.0):
        raise ValueError(f"named arm pose {pose_name!r} must contain six finite joints and gripper_mm")
    result = {f"joint{index}": math.radians(float(value)) for index, value in enumerate(joints, 1)}
    half_width_m = float(width) / 2000.0
    result.update(joint7=half_width_m, joint8=-half_width_m)
    return result


def gazebo_blocks(initial_joint_positions=None) -> str:
    initial_joint_positions = initial_joint_positions or {}
    out = [f"""  <!-- ===== Generado por base_athome_sim/gz_model.py (solo simulacion) ===== -->
  <gazebo><plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
    <topic>/model/{MODEL}/cmd_vel</topic></plugin></gazebo>
  <gazebo><plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
    <odom_frame>odom</odom_frame><robot_base_frame>base_footprint</robot_base_frame>
    <odom_publish_frequency>30</odom_publish_frequency><dimensions>2</dimensions></plugin></gazebo>
  <gazebo><plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
    <topic>/sim/gz/joint_states</topic><update_rate>50</update_rate></plugin></gazebo>
  <!-- Solo ruedas y caster reciben fricción para resistir pares del brazo mientras
       el controlador sostiene velocidad cero. El chasis queda sin fricción. -->
  <gazebo reference="left_wheel"><mu1>1.0</mu1><mu2>1.0</mu2></gazebo>
  <gazebo reference="right_wheel"><mu1>1.0</mu1><mu2>1.0</mu2></gazebo>
  <gazebo reference="rear_caster"><mu1>1.0</mu1><mu2>1.0</mu2></gazebo>
  <gazebo reference="base_link"><mu1>0.0</mu1><mu2>0.0</mu2></gazebo>
  <!-- Friccion de la pinza: necesaria para sostener objetos dinamicos con contacto, sin union cinematica. -->
  <gazebo reference="link7"><mu1>1.0</mu1><mu2>1.0</mu2><kp>1000000</kp><kd>1.0</kd><maxVel>0.01</maxVel><minDepth>0.001</minDepth></gazebo>
  <gazebo reference="link8"><mu1>1.0</mu1><mu2>1.0</mu2><kp>1000000</kp><kd>1.0</kd><maxVel>0.01</maxVel><minDepth>0.001</minDepth></gazebo>
  <gazebo reference="laser"><sensor name="hokuyo" type="gpu_lidar">
    <topic>/sim/gz/scan</topic><gz_frame_id>laser</gz_frame_id><update_rate>20</update_rate>
    <lidar><scan><horizontal><samples>541</samples><min_angle>-2.356</min_angle><max_angle>2.356</max_angle></horizontal></scan>
    <range><min>0.06</min><max>5.0</max></range></lidar><always_on>1</always_on></sensor></gazebo>
  <link name="restaurant_rear_laser"/>
  <joint name="restaurant_rear_laser_joint" type="fixed"><parent link="base_link"/><child link="restaurant_rear_laser"/>
    <origin xyz="-0.24 0 0.32" rpy="0 0 3.14159265359"/></joint>
  <!-- Sim-only rear lidar for the counter retreat: 20 Hz so a slow GUI sim (RTF<1) still beats the 0.3 s dock freshness guard. -->
  <gazebo reference="restaurant_rear_laser"><sensor name="restaurant_rear_lidar" type="gpu_lidar">
    <topic>/sim/gz/rear_scan</topic><gz_frame_id>restaurant_rear_laser</gz_frame_id><update_rate>20</update_rate>
    <lidar><scan><horizontal><samples>361</samples><min_angle>-1.5707963268</min_angle><max_angle>1.5707963268</max_angle></horizontal></scan>
    <range><min>0.04</min><max>8.0</max></range></lidar><always_on>1</always_on></sensor></gazebo>
  <gazebo reference="camera_link"><sensor name="realsense" type="rgbd_camera">
    <topic>/sim/gz/head</topic><gz_frame_id>camera_color_optical_frame</gz_frame_id><update_rate>15</update_rate>
    <camera><horizontal_fov>0.962</horizontal_fov><image><width>640</width><height>480</height></image>
    <clip><near>0.1</near><far>8</far></clip></camera><always_on>1</always_on></sensor></gazebo>
  <gazebo reference="arm_camera_link"><sensor name="wrist_camera" type="camera">
    <topic>/sim/gz/wrist</topic><gz_frame_id>arm_camera_optical_frame</gz_frame_id><update_rate>15</update_rate>
    <camera><horizontal_fov>1.165</horizontal_fov><image><width>640</width><height>480</height></image>
    <clip><near>0.03</near><far>5</far></clip></camera><always_on>1</always_on></sensor></gazebo>
"""]
    out += [_joint_controller(j, float(initial_joint_positions.get(j, 0.0))) for j in POSITION_JOINTS]
    # Agarrar la bolsa: una union que se crea al cerrar la pinza sobre ella y se suelta al abrir
    out.append(f"""  <gazebo><plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
    <parent_link>{GRASP_LINK}</parent_link><child_model>{BAG_MODEL}</child_model><child_link>l</child_link>
    <attach_topic>{BAG_ATTACH_TOPIC}</attach_topic><detach_topic>{BAG_DETACH_TOPIC}</detach_topic>
    <suppress_child_warning>true</suppress_child_warning></plugin></gazebo>
""")
    out.append(f"""  <gazebo><plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
    <parent_link>{GRASP_LINK}</parent_link><child_model>recepcionista_bolsa</child_model><child_link>l</child_link>
    <attach_topic>/sim/gz/recepcionista_bolsa/attach</attach_topic><detach_topic>/sim/gz/recepcionista_bolsa/detach</detach_topic>
    <suppress_child_warning>true</suppress_child_warning></plugin></gazebo>
""")
    for idx in (1, 2):
        can_name = f"{RESTAURANT_COUNTER_CAN}_{idx}"
        out.append(f"""  <gazebo><plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
    <parent_link>{GRASP_LINK}</parent_link><child_model>{can_name}</child_model><child_link>body</child_link>
    <attach_topic>/sim/gz/{can_name}/attach</attach_topic><detach_topic>/sim/gz/{can_name}/detach</detach_topic>
    <suppress_child_warning>true</suppress_child_warning></plugin></gazebo>
""")
    return "".join(out)


SIM_BASE_MASS_KG = 40.0  # real chassis ~12 kg; heavier and lower in sim so the arm reach cannot tip it


def _heavy_base(urdf: str, mass: float = SIM_BASE_MASS_KG) -> str:
    """Sim-only: heavier base_link with a low centre of mass (the real URDF is not changed)."""
    m = re.search(r'(<link name="base_link">.*?)<inertial>.*?</inertial>', urdf, flags=re.S)
    if not m:
        return urdf
    w = d = 0.48
    h = 0.22
    ixx = mass * (d * d + h * h) / 12.0
    izz = mass * (w * w + d * d) / 12.0
    inertial = (f'<inertial><origin rpy="0 0 0" xyz="0 0 0.05"/><mass value="{mass}"/>'
                f'<inertia ixx="{ixx:.4f}" ixy="0" ixz="0" iyy="{ixx:.4f}" iyz="0" izz="{izz:.4f}"/></inertial>')
    return urdf[:m.start()] + m.group(1) + inertial + urdf[m.end():]


def gazebo_urdf(urdf: str, initial_joint_positions=None) -> str:
    """Devuelve el URDF con los bloques de Gazebo insertados antes de </robot>."""
    # Todo el robot sin gravedad: la base se mueve por velocidad, sin ruedas ni rueda loca que la sostengan. Con gravedad
    # cabeceaba 12 grados al avanzar, y las camaras dejaban de estar donde dice el TF.
    urdf = _heavy_base(urdf)
    i = urdf.rfind("</robot>")
    if i < 0:
        raise ValueError("URDF sin </robot>")
    links = re.findall(r'<link name="([^"]+)"', urdf)
    gravity = "".join(f'  <gazebo reference="{ln}"><gravity>0</gravity></gazebo>\n' for ln in links)
    urdf = finger_pad_urdf(urdf)
    i = urdf.rfind("</robot>")
    return urdf[:i] + gazebo_blocks(initial_joint_positions) + gravity + _restaurant_tray_urdf() + urdf[i:]


def bridge_args() -> list:
    """Argumentos para ros_gz_bridge parameter_bridge (gz -> ROS con [, ROS -> gz con ])."""
    args = [
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        f"/model/{MODEL}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        f"/model/{MODEL}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/sim/gz/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        "/sim/gz/rear_scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        "/sim/gz/head/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "/sim/gz/head/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
        "/sim/gz/head/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        "/sim/gz/wrist@sensor_msgs/msg/Image[gz.msgs.Image",  # sensor "camera": la imagen va en el topico base
        "/sim/gz/wrist/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
    ]
    # Necesario para medir el ancho real entre joint7 y joint8 durante el agarre.
    args += ["/sim/gz/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model"]
    args += [f"{joint_cmd_topic(j)}@std_msgs/msg/Float64]gz.msgs.Double" for j in POSITION_JOINTS]
    args += [f"{BAG_ATTACH_TOPIC}@std_msgs/msg/Empty]gz.msgs.Empty", f"{BAG_DETACH_TOPIC}@std_msgs/msg/Empty]gz.msgs.Empty",
             f"{BAG_ODOM_TOPIC}@nav_msgs/msg/Odometry[gz.msgs.Odometry"]
    for idx in (1, 2):
        can_name = f"{RESTAURANT_COUNTER_CAN}_{idx}"
        args.append(f"/sim/gz/{can_name}/attach@std_msgs/msg/Empty]gz.msgs.Empty")
        args.append(f"/sim/gz/{can_name}/detach@std_msgs/msg/Empty]gz.msgs.Empty")
        for prefix in (RESTAURANT_COUNTER_CAN, RESTAURANT_TRAY_ITEM):
            topic = restaurant_item_odometry_topic(f"{prefix}_{idx}")
            args.append(f"{topic}@nav_msgs/msg/Odometry[gz.msgs.Odometry")
    return args
