"""Fase posterior al cierre de pinza en la escena del mostrador (solo simulacion).

Despues de que la pinza se cierra y Gazebo acopla la lata mediante
DetachableJoint, esta fase comprueba el acople, levanta a restaurant_postgrasp,
retrocede la base sin perder la lata ni chocar el mostrador y adopta la pose de
transporte restaurant_carry.

Todas las interacciones con ROS, Gazebo y el reloj se inyectan, de modo que la
logica se pueda verificar con fakes, sin simulacion y sin hardware.
"""

import math
import time
from dataclasses import dataclass

POSTGRASP_POSE = "restaurant_postgrasp"
CARRY_POSE = "restaurant_carry"

MIN_LIFT_M = 0.05
MIN_RETREAT_M = 0.15
MAX_RETREAT_M = 0.30
DEFAULT_RETREAT_M = 0.20
REAR_STOP_M = 0.18
CAN_FOLLOW_TOL_M = 0.10
CAN_SEATED_Z = 1.015
STEP_DT_S = 0.25


@dataclass
class CarryEvidence:
    """Mediciones de la fase, pensadas para el log de la corrida."""

    attached: bool = False
    lift_m: float = 0.0
    retreat_m: float = 0.0
    can_follow_error_m: float = 0.0
    rear_clearance_m: float = float("inf")
    can_z_m: float = 0.0
    pose: str = ""
    reason: str = ""


def can_is_seated(z, *, counter_z=CAN_SEATED_Z, tol=MIN_LIFT_M):
    """Anti-flotacion: la lata sin agarrar debe reposar sobre la mesada."""
    if z is None or not math.isfinite(float(z)):
        return False
    return abs(float(z) - counter_z) <= tol


class CounterCarry:
    """Secuencia attach -> lift -> retreat -> carry con guardas verificables."""

    def __init__(self, *, can_pose, base_pose, attached, rear_clearance,
                 publish_twist, move_holding, sleep=time.sleep,
                 lift_pose=POSTGRASP_POSE, carry_pose=CARRY_POSE,
                 initial_can_pose=None,
                 retreat_m=DEFAULT_RETREAT_M, rear_stop_m=REAR_STOP_M,
                 min_lift_m=MIN_LIFT_M, speed_mps=0.04, timeout_s=12.0,
                 step_dt_s=STEP_DT_S, monotonic=time.monotonic, log=None):
        if not MIN_RETREAT_M - 1e-9 <= retreat_m <= MAX_RETREAT_M + 1e-9:
            raise ValueError(
                f"retreat_m={retreat_m} fuera de rango [{MIN_RETREAT_M}, {MAX_RETREAT_M}]")
        if min_lift_m <= 0 or speed_mps <= 0 or step_dt_s <= 0 or timeout_s <= 0:
            raise ValueError("los parametros de la fase deben ser positivos")
        self.can_pose = can_pose
        self.base_pose = base_pose
        self.attached = attached
        self.rear_clearance = rear_clearance
        self.publish_twist = publish_twist
        self.move_holding = move_holding
        self.sleep = sleep
        self.lift_pose = lift_pose
        self.carry_pose = carry_pose
        self.initial_can_pose = initial_can_pose
        self.retreat_m = float(retreat_m)
        self.rear_stop_m = float(rear_stop_m)
        self.min_lift_m = float(min_lift_m)
        self.speed_mps = float(speed_mps)
        self.timeout_s = float(timeout_s)
        self.step_dt_s = float(step_dt_s)
        self.monotonic = monotonic
        self.log = log if log is not None else (lambda _msg: None)

    def _fail(self, evidence, reason):
        evidence.reason = reason
        self.log(f"COUNTER_CARRY FAIL: {reason}")
        return False, reason, evidence

    def run(self):
        """Devuelve (ok, motivo, evidencia); nunca deja un twist de retroceso activo."""
        evidence = CarryEvidence()
        evidence.attached = bool(self.attached())
        if not evidence.attached:
            return self._fail(evidence, "la lata no esta acoplada; no se levanta ni retrocede")

        # pick_item() may already have lifted the can to postgrasp; retain its
        # measured counter pose so the lift check still compares against ground.
        can0 = self.initial_can_pose if self.initial_can_pose is not None else self.can_pose()
        base0 = self.base_pose()
        if can0 is None or base0 is None:
            return self._fail(evidence, "sin odometria de lata o de base; fase abortada")

        ok, msg = self.move_holding(self.lift_pose)
        evidence.pose = self.lift_pose
        if not ok:
            return self._fail(evidence, f"no se alcanzo {self.lift_pose}: {msg}")

        can_lift = self.can_pose()
        if can_lift is None:
            return self._fail(evidence, "sin odometria de lata tras el levantamiento")
        evidence.lift_m = float(can_lift[2]) - float(can0[2])
        if evidence.lift_m < self.min_lift_m:
            return self._fail(
                evidence,
                f"levantamiento insuficiente ({evidence.lift_m:.3f} m < {self.min_lift_m:.3f} m)")

        origin = (float(base0[0]), float(base0[1]))
        deadline = self.monotonic() + self.timeout_s
        try:
            while True:
                if not self.attached():
                    return self._fail(evidence, "la lata se desacoplo antes de completar el retroceso")
                rear = self.rear_clearance()
                if rear is None or not math.isfinite(float(rear)):
                    return self._fail(evidence, "lidar trasero sin lectura fresca")
                evidence.rear_clearance_m = float(rear)
                if rear <= self.rear_stop_m:
                    return self._fail(
                        evidence,
                        f"sin margen trasero ({rear:.3f} m <= {self.rear_stop_m:.3f} m)")
                now = self.base_pose()
                if now is None:
                    return self._fail(evidence, "odometria de base perdida durante el retroceso")
                evidence.retreat_m = math.hypot(float(now[0]) - origin[0], float(now[1]) - origin[1])
                if evidence.retreat_m >= self.retreat_m:
                    break
                if self.monotonic() >= deadline:
                    return self._fail(
                        evidence, f"timeout de retroceso ({evidence.retreat_m:.3f} m de {self.retreat_m:.3f} m)")
                can_now = self.can_pose()
                if can_now is None or float(can_now[2]) < float(can_lift[2]) - self.min_lift_m / 2.0:
                    return self._fail(evidence, "la lata cayo durante el retroceso")
                self.publish_twist(-self.speed_mps)
                self.sleep(self.step_dt_s)
                self.publish_twist(0.0)
        finally:
            self.publish_twist(0.0)

        base1 = self.base_pose()
        can1 = self.can_pose()
        if base1 is None or can1 is None:
            return self._fail(evidence, "sin odometria al verificar el retroceso")
        evidence.can_z_m = float(can1[2])
        if evidence.can_z_m < float(can_lift[2]) - self.min_lift_m / 2.0:
            return self._fail(evidence, "la lata no conserva la altura de carry tras retroceder")
        if evidence.retreat_m < MIN_RETREAT_M - 1e-9:
            return self._fail(evidence, f"retroceso insuficiente ({evidence.retreat_m:.3f} m)")
        expected_x = float(can0[0]) + float(base1[0]) - origin[0]
        expected_y = float(can0[1]) + float(base1[1]) - origin[1]
        evidence.can_follow_error_m = math.hypot(float(can1[0]) - expected_x, float(can1[1]) - expected_y)
        if evidence.can_follow_error_m > CAN_FOLLOW_TOL_M:
            return self._fail(
                evidence,
                f"la lata no acompano a la base ({evidence.can_follow_error_m:.3f} m)")

        ok, msg = self.move_holding(self.carry_pose)
        evidence.pose = self.carry_pose
        if not ok:
            return self._fail(evidence, f"no se alcanzo {self.carry_pose}: {msg}")
        self.log(
            f"COUNTER_CARRY OK: lift={evidence.lift_m:.3f} m retreat={evidence.retreat_m:.3f} m "
            f"follow_err={evidence.can_follow_error_m:.3f} m pose={self.carry_pose}")
        return True, "carry listo", evidence
