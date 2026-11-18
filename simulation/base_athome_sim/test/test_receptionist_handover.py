#!/usr/bin/env python3
"""Pruebas unitarias para el Paso 5: Handover real de la bolsa con brazo Piper y DetachableJoint."""

import ast
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional
import pytest


ROOT = Path(__file__).resolve().parents[3]


class MockJointState:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None)
        self.name = []
        self.position = []
        self.velocity = []


class MockFloat32:
    def __init__(self):
        self.data = 0.0


def test_calibrated_piper_poses_exist_in_arm_gesture_poses():
    """Verifica que las 4 poses oficiales de manipulacion esten en ARM_GESTURE_POSES con sus angulos calibrados."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "ARM_GESTURE_POSES" for t in n.targets))
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assign], type_ignores=[])), str(source), "exec"), ns)
    poses = ns["ARM_GESTURE_POSES"]

    assert "restaurant_pregrasp" in poses
    assert "restaurant_postgrasp" in poses
    assert "restaurant_carry" in poses
    assert "restaurant_handover" in poses

    # Calibradas de restaurant_arm_poses.yaml
    assert poses["restaurant_pregrasp"] == [8.62, 163.36, -84.72, 19.05, -3.87, -99.57]
    assert poses["restaurant_postgrasp"] == [57.39, 168.91, -90.92, 59.24, -34.5, -101.95]
    assert poses["restaurant_carry"] == [42.87, 155.46, -93.71, 2.38, 21.87, -46.34]
    assert poses["restaurant_handover"] == [35.92, 162.44, -80.89, 96.59, -63.65, -168.54]


def test_move_arm_to_named_pose_publishes_joints_and_gripper_widths():
    """Verifica que move_arm_to_named_pose publique JointState y Float32 de pinza segun los parametros."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "move_arm_to_named_pose")
    assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "ARM_GESTURE_POSES" for t in n.targets))

    joint_cmds = []
    gripper_cmds = []

    ns = {
        "ARM_GESTURE_POSES": None,
        "JointState": MockJointState,
        "Float32": MockFloat32,
        "math": math,
        "Optional": Optional,
        "os": SimpleNamespace(environ={"ATHOME_SIM": "1"}),
        "rclpy": SimpleNamespace(ok=lambda: True),
        "time": SimpleNamespace(time=lambda: 100.0),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assign, method], type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace(
        arm_gestures=True,
        arm_feedback_ok=True,
        arm_enabled=True,
        latest_arm_status_time=100.0,
        current_arm_pose_name="home",
        arm_joints_deg=[15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
        pub_arm_joint=SimpleNamespace(publish=joint_cmds.append),
        pub_arm_gripper=SimpleNamespace(publish=gripper_cmds.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 100)),
        get_logger=lambda: SimpleNamespace(warn=lambda *a, **k: None, info=lambda *a, **k: None),
    )
    node.move_arm_to_named_pose = MethodType(ns["move_arm_to_named_pose"], node)

    # Pregrasp con apertura de 105 mm
    node.move_arm_to_named_pose("restaurant_pregrasp", duration_s=2.0, gripper_mm=105.0)
    assert len(joint_cmds) == 1
    assert len(gripper_cmds) == 1
    assert gripper_cmds[0].data == 105.0
    assert joint_cmds[0].position[0] == pytest.approx(math.radians(8.62), rel=1e-3)
    assert node.current_arm_pose_name == "restaurant_pregrasp"
    # El comando NO debe sobrescribir arm_joints_deg medido
    assert node.arm_joints_deg == [15.0, 25.0, 35.0, 45.0, 55.0, 65.0]

    # Postgrasp con cierre a 50.1 mm
    node.move_arm_to_named_pose("restaurant_postgrasp", duration_s=1.5, gripper_mm=50.1)
    assert len(joint_cmds) == 2
    assert len(gripper_cmds) == 2
    assert gripper_cmds[1].data == 50.1
    assert joint_cmds[1].position[0] == pytest.approx(math.radians(57.39), rel=1e-3)
    assert node.current_arm_pose_name == "restaurant_postgrasp"
    assert node.arm_joints_deg == [15.0, 25.0, 35.0, 45.0, 55.0, 65.0]


def test_move_arm_to_named_pose_enforces_feedback_guards_in_simulation():
    """Verifica que en simulacion (ATHOME_SIM=1) NO se omitan las guardas de feedback ni habilitacion."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "move_arm_to_named_pose")
    assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "ARM_GESTURE_POSES" for t in n.targets))

    joint_cmds = []
    gripper_cmds = []

    ns = {
        "ARM_GESTURE_POSES": None,
        "JointState": MockJointState,
        "Float32": MockFloat32,
        "math": math,
        "Optional": Optional,
        "os": SimpleNamespace(environ={"ATHOME_SIM": "1"}),
        "rclpy": SimpleNamespace(ok=lambda: True),
        "time": SimpleNamespace(time=lambda: 100.0),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assign, method], type_ignores=[])), str(source), "exec"), ns)

    # 1. Brazo sin feedback (arm_feedback_ok=False) -> comando omitido
    node = SimpleNamespace(
        arm_gestures=True,
        arm_feedback_ok=False,
        arm_enabled=True,
        latest_arm_status_time=100.0,
        current_arm_pose_name="home",
        arm_joints_deg=[0.0] * 6,
        pub_arm_joint=SimpleNamespace(publish=joint_cmds.append),
        pub_arm_gripper=SimpleNamespace(publish=gripper_cmds.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 100)),
        get_logger=lambda: SimpleNamespace(warn=lambda *a, **k: None, info=lambda *a, **k: None),
    )
    node.move_arm_to_named_pose = MethodType(ns["move_arm_to_named_pose"], node)
    node.move_arm_to_named_pose("restaurant_pregrasp", duration_s=2.0)
    assert len(joint_cmds) == 0, "No debe comandar brazo si feedback_ok es False"

    # 2. Brazo deshabilitado (arm_enabled=False) -> comando omitido
    node.arm_feedback_ok = True
    node.arm_enabled = False
    node.latest_arm_status_time = 100.0
    node.move_arm_to_named_pose("restaurant_pregrasp", duration_s=2.0)
    assert len(joint_cmds) == 0, "No debe comandar brazo si arm_enabled es False"

    # 3. Status vencido (>0.5 s: recibido hace 0.6 s) -> comando omitido
    node.arm_enabled = True
    node.latest_arm_status_time = 100.0 - 0.6  # 0.6 s atras
    node.move_arm_to_named_pose("restaurant_pregrasp", duration_s=2.0)
    assert len(joint_cmds) == 0, "No debe comandar brazo si status tiene mas de 0.5 s"

    # 4. Status fresco (hace 0.1 s) -> comando publicado
    node.latest_arm_status_time = 100.0 - 0.1
    node.move_arm_to_named_pose("restaurant_pregrasp", duration_s=2.0)
    assert len(joint_cmds) == 1, "Debe comandar brazo si status es fresco (<=0.5 s)"


def test_arm_status_cb_updates_timestamp_and_measured_joints():
    """Verifica que _arm_status_cb actualice latest_arm_status_time y solo joints validos."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_arm_status_cb")

    import json
    ns = {
        "json": json,
        "String": lambda **kw: SimpleNamespace(**kw),
        "time": SimpleNamespace(time=lambda: 123.45),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace(
        arm_feedback_ok=False,
        arm_enabled=False,
        latest_arm_status_time=0.0,
        arm_joints_deg=[0.0] * 6,
        current_arm_pose_name="home",
    )
    node._arm_status_cb = MethodType(ns["_arm_status_cb"], node)

    msg = SimpleNamespace(data=json.dumps({
        "feedback_ok": True,
        "enabled": True,
        "joints_deg": [11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
    }))
    node._arm_status_cb(msg)

    assert node.arm_feedback_ok is True
    assert node.arm_enabled is True
    assert node.latest_arm_status_time == 123.45
    assert node.arm_joints_deg == [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    assert node.current_arm_pose_name == "medida"


def test_after_guest_seated_mary_triggers_pregrasp_and_handover_command():
    """Al sentarse Mary (invitada 2), el robot ejecuta restaurant_pregrasp y pide la bolsa."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    methods = [next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
               for name in ("_after_guest_seated", "_sim_receptionist_task_command")]

    published = []
    subtitles = []
    arm_moves = []
    ns = {
        "os": SimpleNamespace(environ={"ATHOME_SIM": "1"}),
        "String": lambda **kw: SimpleNamespace(**kw),
        "time": SimpleNamespace(time=lambda: 10.0),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace(
        guest_index=2,
        current_name="Mary",
        state=15,
        STATE_SIM_BAG_HANDOVER=20,
        STATE_CHECK_NEXT_GUEST=12,
        pub_sim_receptionist_task=SimpleNamespace(publish=published.append),
        say=lambda text, **kw: subtitles.append(text),
        move_arm_to_named_pose=lambda pose, **kw: arm_moves.append((pose, kw)),
        _introduce_guests_to_each_other=lambda: subtitles.append("introductions"),
    )
    node._sim_receptionist_task_command = MethodType(ns["_sim_receptionist_task_command"], node)
    ns["_after_guest_seated"](node)

    assert "Mary, may I take your bag, please?" in subtitles
    assert arm_moves == [("restaurant_pregrasp", {"duration_s": 2.0, "gripper_mm": 105.0})]
    assert published[0].data == "handover"
    assert node.state == 20
    assert node.bag_handover_announced is False


def test_sim_robot_gz_handover_attaches_detachable_joint_without_move_bag_loop():
    """El simulador posiciona la manija entre los dedos y acopla la bolsa con DetachableJoint."""
    source = ROOT / "simulation/base_athome_sim/base_athome_sim/sim_robot.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimRobot")
    methods = [next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
               for name in ("_gz_receptionist_handover", "_gz_move_receptionist_bag_to_gripper")]

    calls = []
    attached = []
    ns = {
        "time": SimpleNamespace(sleep=lambda _: None, time=lambda: 15.0),
        "gz_model": SimpleNamespace(set_pose_command=lambda world, name, xyz: (name, xyz)),
        "Empty": lambda: None,
        "GZ_WORLD": "casa_simple",
        "math": math,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace(
        receptionist_bag_state="ready",
        receptionist_bag_held=False,
        receptionist_bag_model="recepcionista_bolsa",
        receptionist_seated_guests=[{"guest_index": 2, "xy": [3.5, -1.5]}],
        person_xy=[0.5, 1.8],
        pub_bag_attach=SimpleNamespace(publish=lambda msg: attached.append("bolsa")),
        pub_recep_bag_attach=SimpleNamespace(publish=lambda msg: attached.append("recepcionista_bolsa")),
        _publish_receptionist_task_state=lambda: None,
        _gz_run=calls.append,
        _lookup=lambda *args: ("rotation", [0.4, 0.1, 0.8]),
        get_logger=lambda: SimpleNamespace(info=lambda *args: None),
    )
    node._gz_move_receptionist_bag_to_gripper = MethodType(ns["_gz_move_receptionist_bag_to_gripper"], node)
    ns["_gz_receptionist_handover"](node)

    # Se posiciona una vez cerca de la persona y luego entre los dedos (tcp_link z - 0.05)
    assert calls[0] == ("recepcionista_bolsa", [3.15, -1.5, 0.95])
    assert calls[1] == ("recepcionista_bolsa", (0.4, 0.1, 0.75))
    # No hay mas llamadas set_pose repetidas (cero loop)
    assert len(calls) == 2
    assert node.receptionist_bag_held is True
    assert node.obj_attached is True
    assert node.receptionist_bag_state == "received"
    assert "recepcionista_bolsa" in attached
