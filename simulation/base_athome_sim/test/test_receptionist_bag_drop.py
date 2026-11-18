#!/usr/bin/env python3
"""Pruebas unitarias para el Paso 7: Depósito de la bolsa con brazo Piper y anuncio final."""

import ast
import math
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace
import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_sim_robot_gz_receptionist_drop_bag_detaches_and_places():
    """El simulador desacopla la bolsa mecánicamente mediante DetachableJoint y la ubica en el área de depósito."""
    source = ROOT / "simulation/base_athome_sim/base_athome_sim/sim_robot.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimRobot")
    drop_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_gz_receptionist_drop_bag")

    calls = []
    detached = []
    ns = {
        "gz_model": SimpleNamespace(set_pose_command=lambda world, name, xyz: (name, xyz)),
        "GZ_WORLD": "casa_simple",
        "Empty": lambda: None,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[drop_method], type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace(
        receptionist_bag_held=True,
        obj_attached=True,
        receptionist_bag_model="recepcionista_bolsa",
        receptionist_drop_xy=[4.35, 0.55],
        receptionist_bag_state="host_at_drop",
        pub_bag_detach=SimpleNamespace(publish=lambda msg: detached.append("bolsa")),
        pub_recep_bag_detach=SimpleNamespace(publish=lambda msg: detached.append("recepcionista_bolsa")),
        _publish_receptionist_task_state=lambda: None,
        _gz_run=calls.append,
        get_logger=lambda: SimpleNamespace(info=lambda *args: None),
    )
    ns["_gz_receptionist_drop_bag"](node)

    assert node.receptionist_bag_held is False
    assert node.obj_attached is False
    assert "recepcionista_bolsa" in detached
    assert node.receptionist_bag_state == "dropped"
    assert calls[0] == ("recepcionista_bolsa", (4.35, 0.55, 0.15))


def test_hri_bag_drop_arm_trajectory_and_announcement():
    """Verifica que el robot oriente el brazo, abra la pinza, retorne a home y anuncie el fin de la entrega."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    step_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")

    arm_moves = []
    subtitles = []
    task_cmds = []

    ns = {
        "time": SimpleNamespace(time=lambda: 100.0),
        "math": math,
        "np": SimpleNamespace(clip=lambda v, a, b: max(a, min(b, v))),
        "Twist": lambda: SimpleNamespace(linear=SimpleNamespace(x=0.0), angular=SimpleNamespace(z=0.0)),
        "Vector3": lambda **kw: SimpleNamespace(**kw),
        "String": lambda **kw: SimpleNamespace(**kw),
    }

    node = SimpleNamespace(
        frame_lock=threading.Lock(),
        face_lock=threading.Lock(),
        clear_memory_requested=False,
        latest_frame=None,
        state=21,  # STATE_SIM_BAG_FOLLOW
        STATE_SIM_BAG_FOLLOW=21,
        STATE_SIM_BAG_DROP=22,
        STATE_CHECK_NEXT_GUEST=12,
        STATE_CHALLENGE_COMPLETE=13,
        sim_receptionist_bag_state="host_at_drop",
        sim_receptionist_task_started=95.0,
        state_start_time=95.0,
        pub_cmd_vel=SimpleNamespace(publish=lambda _: None),
        move_arm_to_named_pose=lambda pose, **kw: arm_moves.append((pose, kw)),
        say=lambda text, **kw: subtitles.append(text),
        _sim_receptionist_task_command=lambda cmd: task_cmds.append(cmd),
        get_logger=lambda: SimpleNamespace(error=lambda *a: None),
    )

    for n in cls.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith("STATE_"):
                    val = n.value.value if isinstance(n.value, ast.Constant) else 0
                    setattr(node, t.id, val)

    exec(compile(ast.fix_missing_locations(ast.Module(body=[step_method], type_ignores=[])), str(source), "exec"), ns)
    node.step = MethodType(ns["step"], node)

    # 1. Al llegar el anfitrión, robot orienta brazo a restaurant_handover, abre pinza a 105 mm y comanda drop
    node.step()

    assert ("restaurant_handover", {"duration_s": 2.0, "gripper_mm": 50.1}) in arm_moves
    assert ("restaurant_handover", {"duration_s": 1.0, "gripper_mm": 105.0}) in arm_moves
    assert task_cmds == ["drop"]
    assert node.state == 22  # Transicionó a STATE_SIM_BAG_DROP

    # 2. Bolsa confirmada como soltada ('dropped'): retornar brazo a home y anunciar
    node.sim_receptionist_bag_state = "dropped"
    node.step()

    assert ("home", {"duration_s": 2.0, "gripper_mm": 25.0}) in arm_moves
    assert "The bag is placed. Enjoy the party!" in subtitles
    assert node.state == node.STATE_CHECK_NEXT_GUEST
