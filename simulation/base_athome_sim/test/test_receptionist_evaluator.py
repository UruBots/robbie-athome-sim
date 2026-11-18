#!/usr/bin/env python3
"""Pruebas unitarias para el Paso 8: Evaluador Oficial de Puntaje y Campaña Automatizada.

Verifica la matriz oficial RuleBook 2026.2 (1450 pts máx, tiempo límite 420 s),
desglose por ítem, penalizaciones, detección en logs y registro para ledger.tsv.
"""

import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation/scripts"))

from receptionist_evaluator import (
    evaluate_receptionist_run,
    audit_events_from_logs,
    SCORE_SPEC,
    PENALTY_SPEC,
    MAX_TOTAL_POINTS,
    TIME_LIMIT_SECONDS,
)


def test_evaluator_nominal_run_scores_above_1000_points():
    """Una corrida completa nominal (sin el bonus opcional de abrir puerta) debe superar 1000 puntos."""
    # En la corrida nominal se completan todas las tareas regulares excepto open_door (opcional)
    nominal_events = {k: v["max_units"] for k, v in SCORE_SPEC.items() if k != "open_door"}
    nominal_events["open_door"] = 0

    res = evaluate_receptionist_run(
        events=nominal_events,
        penalties={k: 0 for k in PENALTY_SPEC},
        duration_s=324.5,
    )

    assert res["total_score"] >= 1000, f"Expected >= 1000 pts, got {res['total_score']}"
    assert res["total_score"] == 1050
    assert res["max_possible"] == 1450
    assert res["time_limit_exceeded"] is False
    assert res["items"]["doorbell_detected"]["points"] == 60
    assert res["items"]["gaze_tracking_at_door"]["points"] == 100
    assert res["items"]["offer_free_seat"]["points"] == 200
    assert res["items"]["look_navigation_direction"]["points"] == 30
    assert res["items"]["visual_attributes"]["points"] == 80
    assert res["items"]["no_redundant_questions"]["points"] == 60
    assert res["items"]["intro_name_drink"]["points"] == 120
    assert res["items"]["intro_cross_gaze"]["points"] == 100
    assert res["items"]["bag_handover"]["points"] == 50
    assert res["items"]["follow_host"]["points"] == 200
    assert res["items"]["bag_drop"]["points"] == 50
    assert "penalties=0" in res["summary_note"]


def test_evaluator_perfect_run_with_optional_door_scores_1450_points():
    """Si además se abre la puerta de entrada para ambos invitados, el puntaje alcanza el máximo de 1450."""
    all_events = {k: v["max_units"] for k, v in SCORE_SPEC.items()}
    res = evaluate_receptionist_run(
        events=all_events,
        penalties={k: 0 for k in PENALTY_SPEC},
        duration_s=360.0,
    )

    assert res["total_score"] == 1450
    assert res["items"]["open_door"]["points"] == 400


def test_evaluator_partial_run_scores_accurately():
    """Verifica que una corrida parcial que solo atiende al primer invitado sume exactamente los puntos correspondientes."""
    partial_events = {
        "doorbell_detected": 1,          # 1 x 30 = 30
        "open_door": 0,
        "gaze_tracking_at_door": 1,      # 1 x 50 = 50
        "offer_free_seat": 1,            # 1 x 100 = 100
        "look_navigation_direction": 1,  # 1 x 15 = 15
        "visual_attributes": 0,          # Solo se dicen al segundo invitado
        "no_redundant_questions": 2,     # Name1 + drink1 = 2 x 15 = 30
        "intro_name_drink": 2,           # Name1 + drink1 = 2 x 30 = 60
        "intro_cross_gaze": 1,           # 1 x 50 = 50
        "bag_handover": 0,
        "follow_host": 0,
        "bag_drop": 0,
    }
    # Expected: 30 + 50 + 100 + 15 + 30 + 60 + 50 = 335
    res = evaluate_receptionist_run(
        events=partial_events,
        penalties={k: 0 for k in PENALTY_SPEC},
        duration_s=180.0,
    )
    assert res["total_score"] == 335


def test_evaluator_penalties_deductions():
    """Verifica que las penalizaciones reglamentarias descuenten correctamente del puntaje bruto."""
    nominal_events = {k: v["max_units"] for k, v in SCORE_SPEC.items() if k != "open_door"}
    nominal_events["open_door"] = 0  # 1050 gross

    test_penalties = {
        "misremembered_info": 2,     # 2 x -40 = -80
        "lost_gaze": 1,              # 1 x -200 = -200
        "human_help_handover": 1,    # 1 x -25 = -25
        "dropped_bag": 1,            # 1 x -50 = -50
        "lost_host": 1,              # 1 x -50 = -50
    }
    # Total deductions = -80 - 200 - 25 - 50 - 50 = -405
    # Net score = 1050 - 405 = 645

    res = evaluate_receptionist_run(
        events=nominal_events,
        penalties=test_penalties,
        duration_s=340.0,
    )
    assert res["gross_score"] == 1050
    assert res["total_score"] == 645
    assert res["penalties"]["lost_gaze"]["deduction"] == -200
    assert res["penalties"]["misremembered_info"]["deduction"] == -80
    assert res["penalties"]["human_help_handover"]["deduction"] == -25
    assert res["penalties"]["dropped_bag"]["deduction"] == -50
    assert res["penalties"]["lost_host"]["deduction"] == -50


def test_evaluator_timeout_exceeding_7_minutes_zeroes_score():
    """Si el tiempo total supera 7 minutos (420 s), el evaluador declara timeout y anula el puntaje."""
    nominal_events = {k: v["max_units"] for k, v in SCORE_SPEC.items() if k != "open_door"}
    res = evaluate_receptionist_run(
        events=nominal_events,
        duration_s=435.0,  # > 420.0s
    )

    assert res["time_limit_exceeded"] is True
    assert res["total_score"] == 0
    assert res["net_score"] == 1050  # El puntaje bruto antes del corte
    assert "TIMEOUT_DISQUALIFIED" in res["summary_note"]


def test_evaluator_audits_realistic_nominal_log():
    """Verifica que el parser de logs extraiga correctamente los eventos de un log completo con los 7 pasos."""
    log_text = """
    [INFO] [hri_receptionist_challenge_node]: 🚀 INICIANDO ROBOCUP @HOME HRI RECEPTIONIST CHALLENGE
    SIM_SUBTITLE: Navigating to the entrance door.
    [INFO] [hri_receptionist_challenge_node]: [SIM NAV]: Llegada confirmada a 'A'.
    [INFO] [hri_receptionist_challenge_node]: Timbre detectado: voy a recibir al invitado.
    [INFO] [hri_receptionist_challenge_node]: 👀 [Puerta]: Rostro visible (64x72 px). Confirmando presencia...
    [INFO] [hri_receptionist_challenge_node]: ✔ [Puerta]: Invitado confirmado. Iniciando saludo y bienvenida.
    [INFO] [hri_receptionist_challenge_node]: Nose-Only pan-tilt centrado en nariz activado.
    SIM_SUBTITLE: Hello! Welcome to our home. I am Robot, the receptionist of this house.
    SIM_SUBTITLE: After the beep, please say: My name is, your name, and my favorite drink is, your drink.
    [INFO] [hri_receptionist_challenge_node]: 🎙️ [Whisper ASR]: "My name is John and my favorite drink is cider"
    [INFO] [hri_receptionist_challenge_node]: NLP Parser: Name=John, Drink=cider
    SIM_SUBTITLE: Thank you, John! I noted that your favorite drink is cider.
    SIM_SUBTITLE: John, please follow me to the living room.
    [INFO] [hri_receptionist_challenge_node]: pan_tilt centrado en navegacion look_navigation_direction
    [INFO] [hri_receptionist_challenge_node]: [SIM NAV]: Llegada confirmada a 'B'.
    SIM_SUBTITLE: Charlie, this is our new guest, John. His favorite drink is cider. Charlie, this is our host, Charlie.
    [INFO] [hri_receptionist_challenge_node]: mirar al anfitrion cross_gaze
    [INFO] [hri_receptionist_challenge_node]: 🪑 Ocupacion de asientos (smart_seat_detector): {'chair_north': 0, 'chair_south': 0}
    SIM_SUBTITLE: John, the chair on the north is free. Please take a seat there.
    SIM_SUBTITLE: Wonderful, John. Enjoy your stay.

    SIM_SUBTITLE: Navigating to the entrance door.
    [INFO] [hri_receptionist_challenge_node]: [SIM NAV]: Llegada confirmada a 'A'.
    [INFO] [hri_receptionist_challenge_node]: Timbre detectado: voy a recibir al invitado.
    [INFO] [hri_receptionist_challenge_node]: ✔ [Puerta]: Invitado confirmado. Iniciando saludo y bienvenida.
    [INFO] [hri_receptionist_challenge_node]: Nose-Only pan-tilt centrado en nariz activado.
    SIM_SUBTITLE: Hello! Welcome to our home. I am Robot, the receptionist of this house.
    SIM_SUBTITLE: After the beep, please say: My name is, your name, and my favorite drink is, your drink.
    [INFO] [hri_receptionist_challenge_node]: 🎙️ [Whisper ASR]: "My name is Mary and my favorite drink is red wine"
    [INFO] [hri_receptionist_challenge_node]: NLP Parser: Name=Mary, Drink=red wine
    SIM_SUBTITLE: Thank you, Mary! I noted that your favorite drink is red wine.
    [INFO] [hri_receptionist_challenge_node]: [Brazo] Ejecutando pose 'restaurant_pregrasp' (gripper 105.0 mm)...
    SIM_SUBTITLE: I detect the bag in my gripper, closing in 3 seconds, please release.
    [INFO] [hri_receptionist_challenge_node]: [Brazo] Agarre completado (gripper 50.1 mm). Pose 'restaurant_postgrasp'.
    SIM_SUBTITLE: Mary, please follow me to the living room.
    [INFO] [hri_receptionist_challenge_node]: pan_tilt centrado en navegacion look_navigation_direction
    [INFO] [hri_receptionist_challenge_node]: [SIM NAV]: Llegada confirmada a 'B'.
    SIM_SUBTITLE: Mary, John is sitting on the north chair. He is an adult male with short brown hair wearing a dark shirt.
    SIM_SUBTITLE: John, this is Mary. Her favorite drink is red wine.
    [INFO] [hri_receptionist_challenge_node]: mirar al invitado cross_gaze
    [INFO] [hri_receptionist_challenge_node]: 🪑 Ocupacion de asientos (smart_seat_detector): {'chair_north': 1, 'chair_south': 0}
    SIM_SUBTITLE: Mary, the chair on the south is free. Please have a seat there.
    [INFO] [hri_receptionist_challenge_node]: Charlie dice: "Follow me"
    [INFO] [hri_receptionist_challenge_node]: pub_cmd_vel seguidor reactivo activado siguiendo al anfitrion
    SIM_SUBTITLE: We have arrived. I will place your bag in the drop area.
    [INFO] [hri_receptionist_challenge_node]: [Brazo] Ejecutando pose 'restaurant_handover' (gripper 50.1 mm)...
    [INFO] [hri_receptionist_challenge_node]: [Brazo] Abriendo pinza (105.0 mm). pub_bag_detach ejecutado.
    SIM_SUBTITLE: The bag is placed. Enjoy the party!
    [INFO] [hri_receptionist_challenge_node]: 🎉 CHALLENGE ROBOCUP @HOME RECEPTIONIST COMPLETADA CON EXITO!
    """

    sim_log = "stage=at_door\nstage=at_door\n"
    exp_res = {"completed": True, "unique_seats": True}

    res = evaluate_receptionist_run(
        log_text=log_text,
        sim_log=sim_log,
        duration_s=318.0,
        experiment_result=exp_res,
    )

    assert res["total_score"] >= 1000
    assert res["total_score"] == 1050
    assert res["items"]["doorbell_detected"]["points"] == 60
    assert res["items"]["gaze_tracking_at_door"]["points"] == 100
    assert res["items"]["offer_free_seat"]["points"] == 200
    assert res["items"]["visual_attributes"]["points"] == 80
    assert res["items"]["no_redundant_questions"]["points"] == 60
    assert res["items"]["intro_name_drink"]["points"] == 120
    assert res["items"]["bag_handover"]["points"] == 50
    assert res["items"]["follow_host"]["points"] == 200
    assert res["items"]["bag_drop"]["points"] == 50


def test_evaluator_cli_json():
    """Verifica que la ejecución CLI con --json retorne salida parseable con campos oficiales."""
    cmd = [sys.executable, str(ROOT / "simulation/scripts/receptionist_evaluator.py"), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "total_score" in data
    assert "max_possible" in data
    assert data["max_possible"] == 1450
    assert data["total_score"] == 1050
    assert data["is_estimated"] is True
    assert data["score_status"] == "estimated"


def test_evaluator_strict_probative_auditing_no_unsupported_points():
    """Verifica que el evaluador NO otorgue puntos sin evidencia probatoria estricta."""
    # Log sin evento acústico/simulado de timbre, sin comandos pan_tilt de cabeceo ni cross-gaze,
    # y solo 2 atributos mencionados
    unsupported_log = """
    [INFO] [hri_receptionist_challenge_node]: Llegada confirmada a 'A'.
    SIM_SUBTITLE: Navigating to the entrance door.
    SIM_SUBTITLE: Hello! Welcome to our home.
    [INFO] [hri_receptionist_challenge_node]: NLP Parser: Name=John, Drink=cider
    SIM_SUBTITLE: Navigating to the living room.
    [INFO] [hri_receptionist_challenge_node]: Llegada confirmada a 'B'.
    SIM_SUBTITLE: Charlie, this is John. His favorite drink is cider.
    SIM_SUBTITLE: John is an adult male.
    """
    events, penalties = audit_events_from_logs(log_text=unsupported_log)

    # 1. Sin señal de timbre -> 0 puntos (no inferir por mera llegada a waypoint 'A')
    assert events["doorbell_detected"] == 0, "No debe otorgar timbre solo por llegar a 'A'"

    # 2. Sin pan_tilt de cabeceo ni look_navigation_direction -> 0 puntos (no inferir por subtítulo 'Navigating')
    assert events["look_navigation_direction"] == 0, "No debe otorgar mirada en navegación solo por subtítulo"

    # 3. Sin comando de pan_tilt de cross-gaze -> 0 puntos (no inferir por decir 'Charlie, this is')
    assert events["intro_cross_gaze"] == 0, "No debe otorgar cross-gaze solo por diálogo hablado"

    # 4. Solo 2 atributos (gender='male', age='adult') -> exactamente 2 unidades (40 pts), no inflar a 4
    assert events["visual_attributes"] == 2, f"Esperado 2 atributos, obtenido {events['visual_attributes']}"


def test_evaluator_marks_score_as_estimated():
    """Verifica que el evaluador marque explícitamente el resultado como estimado en simulation."""
    res = evaluate_receptionist_run(
        events={k: v["max_units"] for k, v in SCORE_SPEC.items() if k != "open_door"},
        duration_s=300.0,
    )
    assert res["is_estimated"] is True
    assert res["score_status"] == "estimated"
    assert "status=estimated;" in res["summary_note"]
    assert res["verified_hardware_score"] == 0
    assert "no constituye certificacion oficial" in res["limits"]
