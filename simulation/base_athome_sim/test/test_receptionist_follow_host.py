#!/usr/bin/env python3
"""Pruebas unitarias para el Paso 6: Seguimiento reactivo del anfitrión sin teletransporte."""

import ast
import math
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace
import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_follow_host_walks_charlie_without_base_teleport():
    """Verifica que follow_host mueva a Charlie hacia la zona de entrega con cero teletransportes del robot."""
    source = ROOT / "simulation/base_athome_sim/base_athome_sim/sim_robot.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimRobot")
    methods = [next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
               for name in ("_gz_receptionist_follow_host", "_gz_move_receptionist_base")]

    calls = []
    voices = []
    ns = {
        "time": SimpleNamespace(sleep=lambda _: None),
        "math": math,
        "gz_model": SimpleNamespace(
            MODEL="base_athome",
            set_pose_command=lambda world, name, xyz, yaw: (name, xyz, yaw),
        ),
        "GZ_WORLD": "casa_simple",
        "String": lambda data: SimpleNamespace(data=data),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace(
        receptionist_bag_state="received",
        receptionist_follow_active=False,
        receptionist_host_path_done=False,
        receptionist_host_xy=[2.0, 1.9],
        receptionist_host_yaw=0.0,
        receptionist_drop_xy=[4.15, 0.55],
        base=SimpleNamespace(goal=(1.0, 1.0), x=0.0, y=0.0, yaw=0.0),
        pub_voice=SimpleNamespace(publish=lambda msg: voices.append(msg.data)),
        pub_receptionist_task=SimpleNamespace(publish=lambda msg: None),
        _publish_receptionist_task_state=lambda: None,
        _gz_run=calls.append,
        get_logger=lambda: SimpleNamespace(info=lambda *args: None),
    )
    node._gz_move_receptionist_base = MethodType(ns["_gz_move_receptionist_base"], node)
    ns["_gz_receptionist_follow_host"](node)

    assert "Follow me" in voices
    assert node.receptionist_host_path_done is True
    assert node.receptionist_host_xy == [4.15, 0.55]
    assert node.receptionist_bag_state == "host_at_drop"

    # Verificar que NINGUNA llamada a set_pose apunte a la base del robot (cero teletransporte)
    robot_teleports = [call for call in calls if call[0] == "base_athome"]
    assert len(robot_teleports) == 0

    # Charlie camina por la sala hasta la meta
    host_calls = [call for call in calls if call[0] == "recep_host"]
    assert len(host_calls) > 0
    assert host_calls[-1][1][:2] == (4.15, 0.55)


def test_hri_follow_host_reactive_cmd_vel_and_arrival():
    """Verifica que HRI publique cmd_vel reactivo para mantener distancia entre 0.8 y 1.4 m y frene al llegar."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    step_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")

    cmd_vels = []
    subtitles = []
    task_cmds = []
    head_moves = []

    class MockTwist:
        def __init__(self, linear=None, angular=None):
            self.linear = linear or SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = angular or SimpleNamespace(x=0.0, y=0.0, z=0.0)

    ns = {
        "time": SimpleNamespace(time=lambda: 100.0),
        "math": math,
        "np": SimpleNamespace(clip=lambda v, a, b: max(a, min(b, v))),
        "Twist": MockTwist,
        "Vector3": lambda **kw: SimpleNamespace(**kw),
        "String": lambda **kw: SimpleNamespace(**kw),
    }

    node = SimpleNamespace(
        frame_lock=threading.Lock(),
        state=21,  # STATE_SIM_BAG_FOLLOW
        STATE_SIM_BAG_FOLLOW=21,
        STATE_SIM_BAG_DROP=22,
        STATE_CHALLENGE_COMPLETE=13,
        sim_receptionist_bag_state="following",
        sim_receptionist_host_xy=[3.0, 1.0],
        sim_receptionist_host_xy_time=100.0,
        sim_receptionist_drop_xy=[4.35, 0.55],
        robot_pose=(1.5, 1.0, 0.0),  # A 1.5 m de distancia (dist_err = 1.5 - 1.1 = 0.4 m > 0)
        robot_pose_time=100.0,
        sim_receptionist_task_started=90.0,
        state_start_time=90.0,
        pub_cmd_vel=SimpleNamespace(publish=cmd_vels.append),
        say=lambda text, **kw: subtitles.append(text),
        _sim_receptionist_task_command=lambda cmd: task_cmds.append(cmd),
        move_head_to=lambda pan, tilt: head_moves.append((pan, tilt)),
        get_logger=lambda: SimpleNamespace(error=lambda *a, **k: None, warn=lambda *a, **k: None, info=lambda *a, **k: None),
    )

    # 1. Ejecutar ciclo de seguimiento reactivo en marcha
    # Compilar solo el bloque de STATE_SIM_BAG_FOLLOW
    target_block = None
    for n in ast.walk(step_method):
        if (isinstance(n, ast.Compare) and isinstance(n.left, ast.Attribute)
                and n.left.attr == "state" and any(isinstance(c, ast.Attribute)
                and c.attr == "STATE_SIM_BAG_FOLLOW" for c in n.comparators)):
            # Encontrar el If que contiene esta comparacion
            pass

    # Usar evaluacion directa de la funcion completa con mocks seguros
    # Copiar constantes de estados de la clase
    for n in cls.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith("STATE_"):
                    val = n.value.value if isinstance(n.value, ast.Constant) else 0
                    setattr(node, t.id, val)

    node.state = node.STATE_SIM_BAG_FOLLOW

    node.current_name = "Mary"
    node.latest_frame = None
    node.clear_memory_requested = False
    node.face_lock = threading.Lock()
    node.force_advance = False
    node.gentle_tracker = None
    node.pan_tilt_tracking = False
    node.move_arm_to_named_pose = lambda *a, **k: None
    node.current_poses = []
    node.voice = SimpleNamespace(cancel=lambda _: None)
    node.listen_id = None
    node.get_forward_obstacle_distance = lambda: float("inf")

    exec(compile(ast.fix_missing_locations(ast.Module(body=[step_method], type_ignores=[])), str(source), "exec"), ns)
    node.step = MethodType(ns["step"], node)

    node.step()
    # Verifica que se envio velocidad hacia adelante para acortar distancia a 1.10 m
    assert len(cmd_vels) > 0
    assert cmd_vels[-1].linear.x > 0.0
    assert node.state == 21  # Sigue en STATE_SIM_BAG_FOLLOW

    # 2. Obstaculo frontal detectado por /scan (< 0.55 m): el avance debe frenar inmediatamente a 0.0
    node.get_forward_obstacle_distance = lambda: 0.42
    node.step()
    assert cmd_vels[-1].linear.x == 0.0, "El avance lineal debe detenerse ante obstáculo frontal detectado por scan"

    # 2b. Scan ausente, vencido (>0.3s) o sin lecturas validas (retorna 0.0): cmd_vel completamente 0 (linear y angular)
    node.get_forward_obstacle_distance = lambda: 0.0
    node.step()
    assert cmd_vels[-1].linear.x == 0.0, "linear.x debe ser 0 ante scan 0.0"
    assert cmd_vels[-1].angular.z == 0.0, "angular.z debe ser 0 ante scan 0.0 (cmd_vel 0 completo)"
    node.get_forward_obstacle_distance = lambda: float("inf")

    # 3. Host llega a la zona de deposito: el robot se detiene y avisa
    node.sim_receptionist_bag_state = "host_at_drop"
    node.step()

    assert cmd_vels[-1].linear.x == 0.0
    assert cmd_vels[-1].angular.z == 0.0
    assert "We have arrived. I will place your bag in the drop area." in subtitles
    assert task_cmds == ["drop"]
    assert node.state == 22  # Transiciono a STATE_SIM_BAG_DROP

    # 4. Parada segura por timeout (> 90.0 s): publica 0 y transiciona a STATE_CHALLENGE_COMPLETE
    node.state = 21  # Retornar a STATE_SIM_BAG_FOLLOW para prueba de timeout
    node.sim_receptionist_bag_state = "following"
    node.sim_receptionist_task_started = 0.0  # now (100.0) - started (0.0) = 100s > 90s
    node.step()

    assert cmd_vels[-1].linear.x == 0.0
    assert cmd_vels[-1].angular.z == 0.0
    assert node.state == node.STATE_CHALLENGE_COMPLETE


def test_get_forward_obstacle_distance_safety():
    """Verifica que el chequeo de obstaculo frontal trate scan ausente, vencido (>0.3s) o sin lecturas como obstaculo (0.0)."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_forward_obstacle_distance")

    ns = {"time": SimpleNamespace(time=lambda: 100.0), "float": float}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace()
    node.get_forward_obstacle_distance = MethodType(ns["get_forward_obstacle_distance"], node)

    # 1. Scan ausente (None) -> 0.0
    node.latest_scan = None
    node.latest_scan_time = 100.0
    assert node.get_forward_obstacle_distance() == 0.0

    # 2. Scan sin ranges -> 0.0
    node.latest_scan = SimpleNamespace(ranges=[])
    node.latest_scan_time = 100.0
    assert node.get_forward_obstacle_distance() == 0.0

    # 3. Scan vencido (> 0.3 s de antigüedad) -> 0.0
    scan_ok = SimpleNamespace(
        ranges=[2.0] * 100,
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 100,
        range_min=0.1,
        range_max=10.0,
    )
    node.latest_scan = scan_ok
    node.latest_scan_time = 100.0 - 0.35  # 0.35 s atras > 0.30 s
    assert node.get_forward_obstacle_distance() == 0.0

    # 4. Scan fresco pero sin lecturas validas en el arco frontal (todas inf / nan fuera de rango) -> 0.0
    scan_invalid = SimpleNamespace(
        ranges=[float("inf")] * 100,
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 100,
        range_min=0.1,
        range_max=10.0,
    )
    node.latest_scan = scan_invalid
    node.latest_scan_time = 100.0 - 0.1  # Fresco (0.1 s)
    assert node.get_forward_obstacle_distance() == 0.0

    # 5. Scan fresco con obstaculo a 0.40 m en el arco frontal -> retorna 0.40
    scan_with_obstacle = SimpleNamespace(
        ranges=[5.0] * 40 + [0.40] * 20 + [5.0] * 40,
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 100,
        range_min=0.1,
        range_max=10.0,
    )
    node.latest_scan = scan_with_obstacle
    node.latest_scan_time = 100.0 - 0.1  # Fresco
    assert node.get_forward_obstacle_distance() == pytest.approx(0.40)


def test_hri_detect_and_offer_seat_all_occupied_requests_verbal_help():
    """Fail-closed: si todos los asientos estan ocupados o reservados, no ofrece ninguno y pide ayuda verbal."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    step_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")

    from base_athome_perception import seat_occupancy as seatocc
    from base_athome_perception import scene_analysis as scene

    subtitles = []
    arm_moves = []
    after_guest_called = []

    class MockTwist:
        def __init__(self, linear=None, angular=None):
            self.linear = linear or SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = angular or SimpleNamespace(x=0.0, y=0.0, z=0.0)

    ns = {
        "time": SimpleNamespace(time=lambda: 100.0),
        "math": math,
        "np": SimpleNamespace(clip=lambda v, a, b: max(a, min(b, v))),
        "Twist": MockTwist,
        "Vector3": lambda **kw: SimpleNamespace(**kw),
        "String": lambda **kw: SimpleNamespace(**kw),
        "seatocc": seatocc,
        "scene": scene,
    }

    # Asientos físicos donde couch está ocupado y chair está reservada
    seats = {
        "couch": {"name": "Sofa", "capacity": 1, "x": 3.5, "y": 0.0, "occupied": True},
        "chair": {"name": "Chair", "capacity": 1, "x": 3.5, "y": 1.5, "reserved": True},
    }

    node = SimpleNamespace(
        frame_lock=threading.Lock(),
        face_lock=threading.Lock(),
        latest_frame=None,
        clear_memory_requested=False,
        force_advance=False,
        gentle_tracker=None,
        pan_tilt_tracking=False,
        current_poses=[],
        voice=SimpleNamespace(cancel=lambda _: None),
        listen_id=None,
        state=9,  # STATE_DETECT_AND_OFFER_SEAT
        STATE_DETECT_AND_OFFER_SEAT=9,
        guest_index=2,
        current_name="Bob",
        current_drink="Tea",
        seats=seats,
        scene_objects=[],
        head_pan_sign=1.0,
        registered_guests=[
            {"name": "Alice", "drink": "Water", "seat": "chair"},
            {"name": "Bob", "drink": "Tea", "seat": "provisional_chair"},
        ],
        recognizer=SimpleNamespace(profiles={}, save_metadata_only=lambda: None),
        receptionist_host_name="Charlie",
        scan_seats=lambda: ({"couch": 1, "chair": 1}, True),
        smart_seat_detector=lambda **kw: {"free_seats": [], "chosen_seat": None, "occupied_seats": {"couch": 1, "chair": 1}},
        _head_pan_to_seat=lambda s: 0.0 if s else None,
        move_head_to=lambda *a, **k: None,
        move_arm_to_named_pose=lambda pose, **kw: arm_moves.append(pose),
        sleep_gui=lambda *a, **k: None,
        say=lambda text, **kw: subtitles.append(text),
        _after_guest_seated=lambda: after_guest_called.append(True),
        get_logger=lambda: SimpleNamespace(warn=lambda *a, **k: None, info=lambda *a, **k: None, error=lambda *a, **k: None),
    )

    for n in cls.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith("STATE_"):
                    val = n.value.value if isinstance(n.value, ast.Constant) else 0
                    setattr(node, t.id, val)

    node.state = node.STATE_DETECT_AND_OFFER_SEAT

    exec(compile(ast.fix_missing_locations(ast.Module(body=[step_method], type_ignores=[])), str(source), "exec"), ns)
    node.step = MethodType(ns["step"], node)

    node.step()

    # 1. El robot no debe asignar ningún asiento ocupado o reservado a Bob
    assert node.registered_guests[-1]["seat"] is None

    # 2. No debe ejecutar gestos de señalamiento de asiento con el brazo
    assert "point_couch" not in arm_moves
    assert "point_chair" not in arm_moves
    assert "cube_stow" in arm_moves

    # 3. Debe pedir ayuda verbal al anfitrión (Charlie)
    assert len(subtitles) > 0
    help_text = subtitles[0]
    assert "Charlie" in help_text
    assert "occupied" in help_text.lower()
    assert "Bob" in help_text
    assert "help" in help_text.lower()

    # 4. Debe avanzar limpiamente llamando a _after_guest_seated()
    assert after_guest_called == [True]


def test_hri_visual_seat_path_rejects_host_sofa_covered_by_virtual_chair():
    """Verifica que la vía de asiento VISUAL respete VirtualChairs y no ofrezca el sofá de Charlie."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    step_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")

    from base_athome_perception import seat_occupancy as seatocc
    from base_athome_perception import scene_analysis as scene
    from base_athome_perception.seat_occupancy import VirtualChair

    subtitles = []
    arm_moves = []
    after_guest_called = []

    class MockTwist:
        def __init__(self, linear=None, angular=None):
            self.linear = linear or SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = angular or SimpleNamespace(x=0.0, y=0.0, z=0.0)

    ns = {
        "time": SimpleNamespace(time=lambda: 100.0),
        "math": math,
        "np": SimpleNamespace(clip=lambda v, a, b: max(a, min(b, v))),
        "Twist": MockTwist,
        "Vector3": lambda **kw: SimpleNamespace(**kw),
        "String": lambda **kw: SimpleNamespace(**kw),
        "seatocc": seatocc,
        "scene": scene,
    }

    # Asientos: couch (sofa de Charlie) y chair (silla de invitado)
    seats = {
        "couch": {"name": "Sofa", "capacity": 1, "x": 3.5, "y": 0.0},
        "chair": {"name": "Chair", "capacity": 1, "x": 3.5, "y": 1.5, "reserved": True},
    }

    # Objeto visual detectado por YOLO (un sofa libre aparentemente en imagen)
    visual_couch = SimpleNamespace(kind="couch", bearing_deg=5.0, capacity=1, free=1, seated=False)

    node = SimpleNamespace(
        frame_lock=threading.Lock(),
        face_lock=threading.Lock(),
        latest_frame=None,
        clear_memory_requested=False,
        force_advance=False,
        gentle_tracker=None,
        pan_tilt_tracking=False,
        current_poses=[],
        voice=SimpleNamespace(cancel=lambda _: None),
        listen_id=None,
        state=9,
        guest_index=2,
        current_name="Bob",
        current_drink="Tea",
        seats=seats,
        scene_objects=[visual_couch],
        head_pan_sign=1.0,
        registered_guests=[
            {"name": "Alice", "drink": "Water", "seat": "chair"},
            {"name": "Bob", "drink": "Tea", "seat": "provisional_chair"},
        ],
        recognizer=SimpleNamespace(profiles={}, save_metadata_only=lambda: None),
        receptionist_host_name="Charlie",
        # scan_seats no vio a nadie en el sofa (0) y vio la silla ocupada (1)
        scan_seats=lambda: ({"couch": 0, "chair": 1}, True),
        # smart_seat_detector devuelve couch ocupado virtualmente por Charlie y chair ocupada/reservada
        smart_seat_detector=lambda **kw: {
            "free_seats": [],
            "chosen_seat": None,
            "virtual_chairs": [VirtualChair(x=3.5, y=0.0, occupied=True, occupant_name="Charlie", source="host_default")],
        },
        _head_pan_to_seat=lambda s: 0.0 if s else None,
        move_head_to=lambda *a, **k: None,
        move_arm_to_named_pose=lambda pose, **kw: arm_moves.append(pose),
        sleep_gui=lambda *a, **k: None,
        say=lambda text, **kw: subtitles.append(text),
        _after_guest_seated=lambda: after_guest_called.append(True),
        get_logger=lambda: SimpleNamespace(warn=lambda *a, **k: None, info=lambda *a, **k: None, error=lambda *a, **k: None),
    )

    for n in cls.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith("STATE_"):
                    val = n.value.value if isinstance(n.value, ast.Constant) else 0
                    setattr(node, t.id, val)

    node.state = node.STATE_DETECT_AND_OFFER_SEAT

    exec(compile(ast.fix_missing_locations(ast.Module(body=[step_method], type_ignores=[])), str(source), "exec"), ns)
    node.step = MethodType(ns["step"], node)

    node.step()

    # La via visual NO debe ofrecer el sofa del anfitrion
    assert node.registered_guests[-1]["seat"] is None
    assert "point_couch" not in arm_moves

    # Debe pedir ayuda verbal al anfitrion
    assert len(subtitles) > 0
    assert "Charlie" in subtitles[0]
    assert "occupied" in subtitles[0].lower()
    assert after_guest_called == [True]


def test_validate_sim_domain_isolation_fail_closed():
    """Verifica fail-closed si ATHOME_SIM=1 y ROS_DOMAIN_ID es 10, 0 o vacio/ausente."""
    import os
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_sim_domain_isolation")
    from typing import Optional, Any
    ns = {"os": os, "Optional": Optional, "Any": Any, "RuntimeError": RuntimeError}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(source), "exec"), ns)
    fn = ns["validate_sim_domain_isolation"]

    # Inseguros con ATHOME_SIM=1: deben abortar con RuntimeError claro
    for unsafe_domain in ["10", "010", "0", "00", "", "   ", "abc", "-1", "233", None]:
        env = {"ATHOME_SIM": "1"}
        if unsafe_domain is not None:
            env["ROS_DOMAIN_ID"] = unsafe_domain
        with pytest.raises(RuntimeError, match="FAIL-CLOSED.*ROS_DOMAIN_ID"):
            fn(env)

    # Seguros en simulacion: dominios aislados
    for safe_domain in ["77", "42", "23"]:
        fn({"ATHOME_SIM": "1", "ROS_DOMAIN_ID": safe_domain})

    # Robot real (ATHOME_SIM != 1): dominio 10 o 0 esta permitido
    fn({"ATHOME_SIM": "0", "ROS_DOMAIN_ID": "10"})
    fn({"ROS_DOMAIN_ID": "10"})
    fn({"ATHOME_SIM": "", "ROS_DOMAIN_ID": "0"})


def test_hri_follow_host_stale_telemetry_commands_zero_cmd_vel():
    """Verifica que el seguimiento reactivo publique cmd_vel 0 si host_xy o robot_pose tienen >0.5 s."""
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    step_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")

    cmd_vels = []

    class MockTwist:
        def __init__(self, linear=None, angular=None):
            self.linear = linear or SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = angular or SimpleNamespace(x=0.0, y=0.0, z=0.0)

    import numpy as np
    ns = {
        "time": SimpleNamespace(time=lambda: 100.0),
        "math": math,
        "np": np,
        "Twist": MockTwist,
        "Vector3": lambda **kw: SimpleNamespace(**kw),
        "String": lambda **kw: SimpleNamespace(**kw),
    }

    node = SimpleNamespace(
        frame_lock=threading.Lock(),
        state=21,
        STATE_SIM_BAG_FOLLOW=21,
        STATE_SIM_BAG_DROP=22,
        STATE_CHALLENGE_COMPLETE=13,
        sim_receptionist_bag_state="following",
        sim_receptionist_host_xy=[3.0, 1.0],
        sim_receptionist_host_xy_time=100.0,
        sim_receptionist_drop_xy=[4.35, 0.55],
        robot_pose=(1.5, 1.0, 0.0),
        robot_pose_time=100.0,
        sim_receptionist_task_started=90.0,
        state_start_time=90.0,
        pub_cmd_vel=SimpleNamespace(publish=cmd_vels.append),
        say=lambda text, **kw: None,
        _sim_receptionist_task_command=lambda cmd: None,
        move_head_to=lambda pan, tilt: None,
        get_logger=lambda: SimpleNamespace(error=lambda *a, **k: None, warn=lambda *a, **k: None, info=lambda *a, **k: None),
        current_name="Mary",
        latest_frame=None,
        clear_memory_requested=False,
        face_lock=threading.Lock(),
        force_advance=False,
        gentle_tracker=None,
        pan_tilt_tracking=False,
        move_arm_to_named_pose=lambda *a, **k: None,
        current_poses=[],
        voice=SimpleNamespace(cancel=lambda _: None),
        listen_id=None,
        get_forward_obstacle_distance=lambda: float("inf"),
    )

    for n in cls.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith("STATE_"):
                    val = n.value.value if isinstance(n.value, ast.Constant) else 0
                    setattr(node, t.id, val)

    exec(compile(ast.fix_missing_locations(ast.Module(body=[step_method], type_ignores=[])), str(source), "exec"), ns)
    node.step = MethodType(ns["step"], node)

    # 1. Ambos frescos (0.1 s atras): publica avance hacia el host
    node.sim_receptionist_host_xy_time = 100.0 - 0.1
    node.robot_pose_time = 100.0 - 0.1
    node.step()
    assert len(cmd_vels) == 1
    assert cmd_vels[-1].linear.x > 0.0

    # 2. host_xy vencido (>0.5 s: 0.6 s atras): publica cmd_vel 0
    node.sim_receptionist_host_xy_time = 100.0 - 0.6
    node.robot_pose_time = 100.0 - 0.1
    node.step()
    assert cmd_vels[-1].linear.x == 0.0
    assert cmd_vels[-1].angular.z == 0.0

    # 3. robot_pose vencido (>0.5 s: 0.8 s atras): publica cmd_vel 0
    node.sim_receptionist_host_xy_time = 100.0 - 0.1
    node.robot_pose_time = 100.0 - 0.8
    node.step()
    assert cmd_vels[-1].linear.x == 0.0
    assert cmd_vels[-1].angular.z == 0.0

    # 4. Ambos vencidos: publica cmd_vel 0
    node.sim_receptionist_host_xy_time = 100.0 - 1.2
    node.robot_pose_time = 100.0 - 1.2
    node.step()
    assert cmd_vels[-1].linear.x == 0.0
    assert cmd_vels[-1].angular.z == 0.0


def test_follow_telemetry_callbacks_update_timestamps():
    """Verifica que _amcl_pose_cb y _sim_receptionist_task_state_cb actualicen sus marcas de tiempo."""
    import json
    source = ROOT / "src/base_athome_perception/scripts/hri_receptionist_challenge.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriReceptionistChallengeNode")
    amcl_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_amcl_pose_cb")
    task_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sim_receptionist_task_state_cb")

    ns = {
        "time": SimpleNamespace(time=lambda: 1234.5),
        "math": math,
        "json": json,
        "PoseWithCovarianceStamped": lambda **kw: None,
        "String": lambda **kw: None,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[amcl_method, task_method], type_ignores=[])), str(source), "exec"), ns)

    node = SimpleNamespace(
        robot_pose=None,
        robot_pose_time=0.0,
        sim_receptionist_host_xy=None,
        sim_receptionist_host_xy_time=0.0,
        sim_receptionist_drop_xy=None,
        sim_receptionist_bag_state=None,
    )
    node._amcl_pose_cb = MethodType(ns["_amcl_pose_cb"], node)
    node._sim_receptionist_task_state_cb = MethodType(ns["_sim_receptionist_task_state_cb"], node)

    # Pose AMCL
    msg_pose = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=1.2, y=3.4),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )))
    node._amcl_pose_cb(msg_pose)
    assert node.robot_pose == (1.2, 3.4, 0.0)
    assert node.robot_pose_time == 1234.5

    # Task state con host_xy
    msg_task = SimpleNamespace(data=json.dumps({"host_xy": [5.6, 7.8], "bag_state": "following"}))
    node._sim_receptionist_task_state_cb(msg_task)
    assert node.sim_receptionist_host_xy == [5.6, 7.8]
    assert node.sim_receptionist_host_xy_time == 1234.5
    assert node.sim_receptionist_bag_state == "following"
