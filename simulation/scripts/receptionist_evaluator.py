#!/usr/bin/env python3
"""Official RuleBook 2026.2 Score Evaluator for RoboCup@Home Receptionist.

Evaluates execution logs (hri.log, sim.log, metrics.json, result.json)
against the official RuleBook 2026.2 scoring matrix (1450 max regular points,
7 minute / 420 s time limit) and penalizes rule violations.
"""

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple


MAX_TOTAL_POINTS = 1450
TIME_LIMIT_SECONDS = 420.0  # 7 minutes

SCORE_SPEC: Dict[str, Dict[str, Any]] = {
    # Greeting (Recepción de Invitados) - max 930 pts
    "doorbell_detected": {
        "description": "Detectar el sonido del timbre como señal de llegada",
        "category": "greeting",
        "points_per_unit": 30,
        "max_units": 2,
        "max_points": 60,
    },
    "open_door": {
        "description": "Abrir la puerta de entrada para el invitado (opcional)",
        "category": "greeting",
        "points_per_unit": 200,
        "max_units": 2,
        "max_points": 400,
    },
    "gaze_tracking_at_door": {
        "description": "Mirar a la persona que habla al recibirla (Gaze tracking)",
        "category": "greeting",
        "points_per_unit": 50,
        "max_units": 2,
        "max_points": 100,
    },
    "offer_free_seat": {
        "description": "Ofrecer un asiento libre al nuevo invitado",
        "category": "greeting",
        "points_per_unit": 100,
        "max_units": 2,
        "max_points": 200,
    },
    "look_navigation_direction": {
        "description": "Mirar en la dirección de navegación o hacia la meta",
        "category": "greeting",
        "points_per_unit": 15,
        "max_units": 2,
        "max_points": 30,
    },
    "visual_attributes": {
        "description": "Decir atributo visual del primer invitado al segundo",
        "category": "greeting",
        "points_per_unit": 20,
        "max_units": 4,  # gender, age, hair, clothing
        "max_points": 80,
    },
    "no_redundant_questions": {
        "description": "No formular preguntas no esenciales para confirmar",
        "category": "greeting",
        "points_per_unit": 15,
        "max_units": 4,  # name1, drink1, name2, drink2
        "max_points": 60,
    },
    # Introduction (Presentación de Invitados) - max 220 pts
    "intro_name_drink": {
        "description": "Decir nombre y bebida favorita de cada invitado",
        "category": "introduction",
        "points_per_unit": 30,
        "max_units": 4,  # name1, drink1, name2, drink2
        "max_points": 120,
    },
    "intro_cross_gaze": {
        "description": "Mirar al invitado correcto mientras se habla del otro",
        "category": "introduction",
        "points_per_unit": 50,
        "max_units": 2,  # intro 1, intro 2
        "max_points": 100,
    },
    # Guiding & C4 (Guía y Manipulación) - max 300 pts
    "bag_handover": {
        "description": "Recibir la bolsa del invitado mediante handover seguro",
        "category": "guiding_c4",
        "points_per_unit": 50,
        "max_units": 1,
        "max_points": 50,
    },
    "follow_host": {
        "description": "Seguir al anfitrión hasta el área de depósito de bolsa",
        "category": "guiding_c4",
        "points_per_unit": 200,
        "max_units": 1,
        "max_points": 200,
    },
    "bag_drop": {
        "description": "Depositar la bolsa en el área correcta",
        "category": "guiding_c4",
        "points_per_unit": 50,
        "max_units": 1,
        "max_points": 50,
    },
}

PENALTY_SPEC: Dict[str, Dict[str, Any]] = {
    "misremembered_info": {
        "description": "Información errónea memorizada (-40 pts c/u, max -160)",
        "penalty_per_unit": -40,
        "max_penalty": -160,
    },
    "lost_gaze": {
        "description": "Ignorar o perder contacto visual con personas (-200 pts c/u)",
        "penalty_per_unit": -200,
        "max_penalty": -1000,
    },
    "human_help_handover": {
        "description": "Ayuda humana en handover (-25 pts)",
        "penalty_per_unit": -25,
        "max_penalty": -100,
    },
    "dropped_bag": {
        "description": "Caída de bolsa (-50 pts)",
        "penalty_per_unit": -50,
        "max_penalty": -200,
    },
    "lost_host": {
        "description": "Pérdida de anfitrión durante seguimiento (-50 pts)",
        "penalty_per_unit": -50,
        "max_penalty": -200,
    },
}


def audit_events_from_logs(
    log_text: str = "",
    sim_log: str = "",
    experiment_result: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Audits raw log texts and experiment results to extract event and penalty counts."""
    events: Dict[str, int] = {k: 0 for k in SCORE_SPEC}
    penalties: Dict[str, int] = {k: 0 for k in PENALTY_SPEC}
    exp = experiment_result or {}

    combined_text = f"{log_text}\n{sim_log}"
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]

    # 1. Doorbell detection (max 2)
    # Exige señal probatoria acústica o evento explícito de timbre, no simple navegación a puerta
    doorbell_matches = len(re.findall(
        r"(Timbre detectado|doorbell_detected|\[Inicio\]:\s*Timbre|DoorbellDetector\.process.*detected|doorbell_audio.*detected)",
        combined_text,
        re.IGNORECASE,
    ))
    events["doorbell_detected"] = min(2, doorbell_matches)

    # 2. Open door (optional bonus, max 2)
    # Requiere acción explícita de manipulación de puerta por el robot
    open_door_matches = len(re.findall(
        r"(door_opened_by_robot|abrir_puerta_robot|robot opened the entrance door)",
        combined_text,
        re.IGNORECASE,
    ))
    events["open_door"] = min(2, open_door_matches)

    # 3. Gaze tracking at door (max 2)
    # Requiere seguimiento activo de cabeza/nariz en puerta, no mera detección facial estática
    gaze_door_matches = len(re.findall(
        r"(Nose-Only pan-tilt|gaze tracking at door|mirando a la persona|pan_tilt.*tracking)",
        log_text,
        re.IGNORECASE,
    ))
    events["gaze_tracking_at_door"] = min(2, gaze_door_matches)

    # 4. Offer free seat (max 2)
    # Asignación verificada de asiento libre (smart seat detector con sillas virtuales)
    seat_offers = len(re.findall(
        r"(smart_seat_detector|is free\. Please|the chair on the (?:left|right|north|south) .* is free|Please (?:take|have) a seat)",
        log_text,
        re.IGNORECASE,
    ))
    if exp.get("unique_seats"):
        events["offer_free_seat"] = 2
    else:
        events["offer_free_seat"] = min(2, seat_offers)

    # 5. Look in navigation direction (max 2)
    # Exige evidencia en logger/pan_tilt de cabeceo/guiñada alineados a la trayectoria
    look_nav_matches = len(re.findall(
        r"(look_navigation_direction|pan_tilt.*(?:pitch=0|centrado en navegacion))",
        log_text,
        re.IGNORECASE,
    ))
    events["look_navigation_direction"] = min(2, look_nav_matches)

    # 6. Visual attributes (max 4: gender, age, hair, clothing)
    # Conteo estricto y no inflado por cada categoría extraída y anunciada
    attr_categories = {"gender": False, "age": False, "hair": False, "clothing": False}
    for line in lines:
        if any(kw in line.lower() for kw in ("sitting on the", "guest is", "attributes", "he is", "she is", "is an", "is a")):
            lower = line.lower()
            if any(w in lower for w in ("male", "female", "man", "woman", "gentleman", "lady")):
                attr_categories["gender"] = True
            if any(w in lower for w in ("adult", "young", "elderly", "teen")):
                attr_categories["age"] = True
            if any(w in lower for w in ("hair", "brown", "black", "blonde", "short", "long")):
                attr_categories["hair"] = True
            if any(w in lower for w in ("shirt", "t-shirt", "jacket", "pants", "wearing", "dark")):
                attr_categories["clothing"] = True
    events["visual_attributes"] = sum(1 for v in attr_categories.values() if v)

    # 7. No redundant questions (max 4: name1, drink1, name2, drink2)
    # Check clean direct NLP parsing without repeat requests
    has_nlp = any("NLP Parser:" in line or "I noted that your favorite drink is" in line or "favorite drink is" in line for line in lines)
    redundant_repeats = len(re.findall(
        r"(could you repeat|please say again|answer yes or no to confirm your name|pardon me|I did not catch)",
        log_text,
        re.IGNORECASE,
    ))
    if has_nlp and redundant_repeats == 0:
        events["no_redundant_questions"] = 4
    elif has_nlp:
        events["no_redundant_questions"] = max(0, 4 - redundant_repeats)

    # 8. Introduction of name & drink (max 4: name1, drink1, name2, drink2)
    intro_items = 0
    # Check guest 1 intro (John / Robin) + drink (cider / Iced Tea)
    if re.search(r"(this is (?:our (?:new )?guest, )?(?:John|Robin)|Charlie, this is (?:John|Robin))", log_text, re.IGNORECASE):
        intro_items += 1
    if re.search(r"((?:His|Their|Robin's|John's) favorite drink is (?:cider|Iced Tea))", log_text, re.IGNORECASE):
        intro_items += 1
    # Check guest 2 intro (Mary / Morgan) + drink (red wine / Apple Juice)
    if re.search(r"((?:this is (?:our (?:new )?guest, )?|Charlie, Robin, this is (?:our (?:new )?guest, )?)(?:Mary|Morgan)|John, this is Mary)", log_text, re.IGNORECASE):
        intro_items += 1
    if re.search(r"((?:Her|Their|Mary's|Morgan's) favorite drink is (?:red wine|Apple Juice))", log_text, re.IGNORECASE):
        intro_items += 1
    events["intro_name_drink"] = min(4, intro_items)

    # 9. Cross gaze during introductions (max 2)
    # Exige comandos explícitos de orientación de pan_tilt hacia la otra persona
    intro_gaze_matches = len(re.findall(
        r"(mirar al anfitrion|mirar al invitado|cross_gaze|pan_tilt.*(?:cross_intro|intro_target))",
        log_text,
        re.IGNORECASE,
    ))
    events["intro_cross_gaze"] = min(2, intro_gaze_matches)

    # 10. Bag handover (Piper arm) (max 1)
    if re.search(
        r"(restaurant_pregrasp|restaurant_postgrasp|closing in 3 seconds|detect the bag in my gripper|pub_bag_attach|gripper_mm.*50\.1)",
        combined_text,
        re.IGNORECASE,
    ):
        events["bag_handover"] = 1

    # 11. Follow host reactively via cmd_vel (max 1)
    if re.search(
        r"(Follow me|pub_cmd_vel.*following|reactive_follow_host|following host.*cmd_vel|We have arrived\. I will (?:leave|place) your bag in the (?:marked )?drop area)",
        combined_text,
        re.IGNORECASE,
    ):
        events["follow_host"] = 1

    # 12. Bag drop on marked zone (Piper arm) (max 1)
    if re.search(
        r"(restaurant_handover|The bag is placed\. Enjoy the party!|pub_bag_detach|pub_recep_bag_detach|_gz_receptionist_drop_bag|gripper_mm.*105\.0)",
        combined_text,
        re.IGNORECASE,
    ):
        events["bag_drop"] = 1

    # Auditing Penalties
    # Lost gaze
    lost_gaze_matches = len(re.findall(r"(lost gaze contact|perdio contacto visual|face tracking lost for >5s)", combined_text, re.IGNORECASE))
    penalties["lost_gaze"] = lost_gaze_matches

    # Misremembered info
    misinfo_matches = len(re.findall(r"(misremembered|wrong name announced|wrong drink announced|hallucinated profile)", combined_text, re.IGNORECASE))
    penalties["misremembered_info"] = min(4, misinfo_matches)

    # Human help in handover
    human_help = len(re.findall(r"(human_help_handover|manual push on gripper|operator intervention in handover)", combined_text, re.IGNORECASE))
    penalties["human_help_handover"] = human_help

    # Dropped bag
    dropped_bag = len(re.findall(r"(bag dropped outside zone|dropped_bag_penalty|premature bag detach)", combined_text, re.IGNORECASE))
    penalties["dropped_bag"] = dropped_bag

    # Lost host
    lost_host = len(re.findall(r"(lost host during follow|anfitrion perdido|host lost timeout)", combined_text, re.IGNORECASE))
    penalties["lost_host"] = lost_host

    return events, penalties


def evaluate_receptionist_run(
    log_text: str = "",
    sim_log: str = "",
    duration_s: float = 0.0,
    experiment_result: Optional[Dict[str, Any]] = None,
    events: Optional[Dict[str, int]] = None,
    penalties: Optional[Dict[str, int]] = None,
    allow_overtime: bool = False,
) -> Dict[str, Any]:
    """Computes official RuleBook 2026.2 score for a Receptionist mission.

    Args:
        log_text: Content of hri.log.
        sim_log: Content of sim.log.
        duration_s: Elapsed runtime in seconds.
        experiment_result: Parsed dictionary from RENDERED_EXPERIMENT_RESULT.
        events: Optional explicit event counts to override or provide directly.
        penalties: Optional explicit penalty counts to override or provide directly.
        allow_overtime: If True, do not zero score on >420s timeout.

    Returns:
        Structured evaluation dictionary containing:
          - total_score: Final calculated score (0..1450).
          - max_possible: 1450.
          - duration_s: Execution duration.
          - time_limit_exceeded: Boolean indicating if time > 420.0s.
          - items: Per-item breakdown with points and units.
          - penalties: Per-penalty breakdown with deductions.
          - summary_note: Compact string suitable for ledger.tsv.
    """
    if events is None or penalties is None:
        audited_events, audited_penalties = audit_events_from_logs(log_text, sim_log, experiment_result)
        final_events = {**audited_events, **(events or {})}
        final_penalties = {**audited_penalties, **(penalties or {})}
    else:
        final_events = events
        final_penalties = penalties

    time_limit_exceeded = (duration_s > TIME_LIMIT_SECONDS) and not allow_overtime

    items_breakdown: Dict[str, Dict[str, Any]] = {}
    gross_score = 0

    for rule_name, rule in SCORE_SPEC.items():
        unit_count = min(rule["max_units"], max(0, final_events.get(rule_name, 0)))
        earned_points = unit_count * rule["points_per_unit"]
        items_breakdown[rule_name] = {
            "units": unit_count,
            "max_units": rule["max_units"],
            "points": earned_points,
            "max_points": rule["max_points"],
            "description": rule["description"],
            "category": rule["category"],
        }
        gross_score += earned_points

    penalties_breakdown: Dict[str, Dict[str, Any]] = {}
    total_deductions = 0

    for penalty_name, rule in PENALTY_SPEC.items():
        count = max(0, final_penalties.get(penalty_name, 0))
        raw_deduction = count * rule["penalty_per_unit"]
        if "max_penalty" in rule:
            raw_deduction = max(rule["max_penalty"], raw_deduction)
        penalties_breakdown[penalty_name] = {
            "count": count,
            "deduction": raw_deduction,
            "description": rule["description"],
        }
        total_deductions += raw_deduction

    net_score = max(0, gross_score + total_deductions)
    total_score = 0 if time_limit_exceeded else net_score

    # Compact summary string for ledger.tsv
    # E.g. doorbell=60/60,gaze_door=100/100,seats=200/200,nav_gaze=30/30,attributes=80/80,no_redundant=60/60,intro=120/120,intro_gaze=100/100,handover=50/50,follow=200/200,drop=50/50;penalties=0
    summary_parts = []
    for k in (
        "doorbell_detected",
        "gaze_tracking_at_door",
        "offer_free_seat",
        "look_navigation_direction",
        "visual_attributes",
        "no_redundant_questions",
        "intro_name_drink",
        "intro_cross_gaze",
        "bag_handover",
        "follow_host",
        "bag_drop",
    ):
        it = items_breakdown[k]
        abbr = k.replace("_detected", "").replace("tracking_at_", "").replace("look_", "")
        summary_parts.append(f"{abbr}={it['points']}/{it['max_points']}")
    if items_breakdown["open_door"]["points"] > 0:
        summary_parts.append(f"open_door={items_breakdown['open_door']['points']}/400")

    summary_note = "status=estimated;" + ",".join(summary_parts) + f";penalties={abs(total_deductions)}"
    if time_limit_exceeded:
        summary_note += ";TIMEOUT_DISQUALIFIED"

    return {
        "total_score": total_score,
        "estimated_score": total_score,
        "verified_hardware_score": 0,
        "is_estimated": True,
        "score_status": "estimated",
        "net_score": net_score,
        "gross_score": gross_score,
        "max_possible": MAX_TOTAL_POINTS,
        "duration_s": round(duration_s, 2),
        "time_limit_exceeded": time_limit_exceeded,
        "items": items_breakdown,
        "penalties": penalties_breakdown,
        "summary_note": summary_note,
        "limits": "Puntaje estimado en simulador mediante fixtures de HRI/speech; no constituye certificacion oficial con hardware real",
    }


def evaluate_folder(folder_path: Path) -> Dict[str, Any]:
    """Evaluates a completed campaign execution folder."""
    folder = Path(folder_path)
    hri_log = (folder / "hri.log").read_text(errors="replace") if (folder / "hri.log").is_file() else ""
    sim_log = (folder / "sim.log").read_text(errors="replace") if (folder / "sim.log").is_file() else ""

    result_json = {}
    if (folder / "result.json").is_file():
        try:
            result_json = json.loads((folder / "result.json").read_text())
        except ValueError:
            pass

    duration_s = float(result_json.get("elapsed_s", 0.0))
    exp_result = {}
    for line in hri_log.splitlines():
        if line.startswith("RENDERED_EXPERIMENT_RESULT "):
            try:
                exp_result = json.loads(line.split("RENDERED_EXPERIMENT_RESULT ", 1)[1])
                break
            except ValueError:
                pass

    return evaluate_receptionist_run(
        log_text=hri_log,
        sim_log=sim_log,
        duration_s=duration_s,
        experiment_result=exp_result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="Directory of run (containing hri.log / result.json) or file")
    parser.add_argument("--log", type=Path, help="Path to hri.log")
    parser.add_argument("--sim-log", type=Path, help="Path to sim.log")
    parser.add_argument("--duration", type=float, default=0.0, help="Elapsed runtime in seconds")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    if args.target:
        p = Path(args.target)
        if p.is_dir():
            eval_res = evaluate_folder(p)
        else:
            eval_res = evaluate_receptionist_run(log_text=p.read_text(errors="replace"), duration_s=args.duration)
    elif args.log:
        hri_text = args.log.read_text(errors="replace")
        sim_text = args.sim_log.read_text(errors="replace") if args.sim_log and args.sim_log.is_file() else ""
        eval_res = evaluate_receptionist_run(log_text=hri_text, sim_log=sim_text, duration_s=args.duration)
    else:
        # Default dry nominal evaluation
        eval_res = evaluate_receptionist_run(
            events={k: v["max_units"] for k, v in SCORE_SPEC.items() if k != "open_door"},
            duration_s=320.0,
        )

    if args.json:
        print(json.dumps(eval_res, indent=2))
    else:
        print(f"==================================================")
        print(f"ROBOCUP@HOME RECEPTIONIST RULEBOOK 2026.2 SCORE")
        print(f"==================================================")
        print(f"Total Score : {eval_res['total_score']} / {eval_res['max_possible']} pts")
        print(f"Duration    : {eval_res['duration_s']} s (Limit: {TIME_LIMIT_SECONDS} s)")
        print(f"Timeout     : {eval_res['time_limit_exceeded']}")
        print(f"Summary     : {eval_res['summary_note']}")
        print(f"--------------------------------------------------")
        print(f"ITEM BREAKDOWN:")
        for k, it in eval_res["items"].items():
            print(f"  - {it['description'][:48]:<48}: {it['points']:>4} / {it['max_points']:>4} pts ({it['units']}/{it['max_units']})")
        print(f"PENALTIES:")
        for k, pen in eval_res["penalties"].items():
            if pen["deduction"] != 0:
                print(f"  - {pen['description'][:48]:<48}: {pen['deduction']:>4} pts (count={pen['count']})")
        print(f"==================================================")


if __name__ == "__main__":
    main()
