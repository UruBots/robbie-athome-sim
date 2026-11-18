#!/usr/bin/env python3
"""Modelos puros (sin ROS) de la simulacion: brazo, pan-tilt, base movil y microfono.

Se testean sin ROS; sim_robot.py los conecta a los mismos topicos/servicios que el robot real.
"""
import math
import time
from typing import List, Optional, Sequence, Tuple

REST_DEG = [86.97, 172.87, -162.63, -99.89, -0.68, 44.98]


def sim_arm_enable_requested(environment):
    """Allow arm motion only when the launcher explicitly selects a sim arm run."""
    return (environment.get("SIM") == "1" or environment.get("ATHOME_SIM") == "1") \
        and environment.get("ARM") == "1" \
        and environment.get("ATHOME_SIM_ARM_MOTION", "1") == "1"


def restaurant_base_hold_requested(state, info=None):
    """Hold the simulated base from docked service through all arm manipulation states."""
    if state in {"ORDERING_BARMAN", "DETECTING_ITEM", "PICKING_ITEM", "HANDING_OVER",
                 "RELEASING_ITEM", "STOWING_ARM"}:
        return True
    return state == "WAITING_BARMAN"


def restaurant_base_hold_angular_command(current_yaw, anchor_yaw):
    """Apply a bounded sim-only yaw correction toward the docked hold heading."""
    try:
        current_yaw, anchor_yaw = float(current_yaw), float(anchor_yaw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(current_yaw) or not math.isfinite(anchor_yaw):
        return 0.0
    error = math.atan2(math.sin(anchor_yaw - current_yaw),
                       math.cos(anchor_yaw - current_yaw))
    return max(-0.04, min(0.04, 0.8 * error))


def restaurant_handoff_confirmed(text):
    """Recognize only explicit affirmative voice replies for the simulated barman."""
    try:
        normalized = " ".join(str(text).lower().strip().rstrip(".!?").split())
    except (TypeError, ValueError):
        return False
    return normalized in {"yes", "robot yes", "yeah", "robot yeah"}


class SimArm:
    """Piper simulado: sigue joint_cmd (coseno) y trayectorias temporizadas como el controlador real."""

    def __init__(self, joints_deg: Optional[Sequence[float]] = None, *, enabled: bool = False):
        self.q = list(joints_deg or REST_DEG)
        self.gripper_mm = 60.0
        self.gripper_cmd_mm = 60.0  # apertura ordenada que persiguen los dedos fisicos
        self.enabled = bool(enabled)
        self._seg = None   # (t0, dur, q0, q1)
        self._traj = None  # (t0, times, points)
        self._grip = None  # (t0, dur, g0, g1)
        self.gripper_target_mm = 60.0
        self.grip_block_mm: Optional[float] = None  # objeto entre los dedos: la pinza no cierra mas alla


    def set_enable(self, on: bool) -> bool:
        self.enabled = bool(on)
        if not on:
            self._seg = self._traj = None
        return True

    def joint_cmd(self, target_deg: Sequence[float], duration: float, now: Optional[float] = None):
        now = time.time() if now is None else now
        self._traj = None
        self._seg = (now, max(0.6, min(10.0, duration)), list(self.q), list(target_deg))

    def trajectory(self, times: Sequence[float], points: Sequence[Sequence[float]], now: Optional[float] = None):
        now = time.time() if now is None else now
        self._seg = None
        self._traj = (now, [0.0] + list(times), [list(self.q)] + [list(p) for p in points])

    def gripper(self, mm: float, now: Optional[float] = None):
        now = time.time() if now is None else now
        self._grip = (now, 1.2, self.gripper_mm, float(max(0.0, min(105.0, mm))))
        self.gripper_target_mm = self._grip[3]

    @property
    def moving(self) -> bool:
        return self._seg is not None or self._traj is not None

    def update(self, now: Optional[float] = None):
        now = time.time() if now is None else now
        if self.enabled and self._seg:
            t0, dur, q0, q1 = self._seg
            a = min(1.0, (now - t0) / dur)
            s = (1 - math.cos(math.pi * a)) / 2
            self.q = [x + s * (y - x) for x, y in zip(q0, q1)]
            if a >= 1.0:
                self._seg = None
        elif self.enabled and self._traj:
            t0, times, pts = self._traj
            el = now - t0
            if el >= times[-1]:
                self.q, self._traj = list(pts[-1]), None
            else:
                k = max(0, next(i for i, t in enumerate(times) if t > el) - 1)
                a = (el - times[k]) / max(1e-3, times[k + 1] - times[k])
                self.q = [x + a * (y - x) for x, y in zip(pts[k], pts[k + 1])]
        if self.enabled and self._grip:
            t0, dur, g0, g1 = self._grip
            a = min(1.0, (now - t0) / dur)
            self.gripper_mm = self.gripper_cmd_mm = g0 + a * (g1 - g0)
            if self.grip_block_mm is not None and g1 < g0:
                self.gripper_mm = max(self.gripper_mm, self.grip_block_mm)
            if a >= 1.0:
                self._grip = None


def sim_arm_start_state(initial_joint_positions):
    """Return commanded Piper joints and gripper width from Gazebo initial coordinates."""
    joints_deg = [math.degrees(float(initial_joint_positions[f"joint{index}"])) for index in range(1, 7)]
    width_mm = 1000.0 * (float(initial_joint_positions["joint7"])
                         - float(initial_joint_positions["joint8"]))
    if not all(math.isfinite(value) for value in (*joints_deg, width_mm)) or not 0.0 <= width_mm <= 105.0:
        raise ValueError("invalid initial Gazebo arm joint positions")
    return joints_deg, width_mm


def sim_arm_start_state_matches(measured_positions, target_positions, tolerance_rad=0.05):
    """Whether Gazebo feedback has settled at the requested initial arm pose."""
    try:
        measured = [float(value) for value in measured_positions]
        target = [float(value) for value in target_positions]
        tolerance = float(tolerance_rad)
        return (len(measured) == len(target) == 8
                and all(math.isfinite(value) for value in measured + target)
                and math.isfinite(tolerance) and tolerance >= 0.0
                and max(abs(a - b) for a, b in zip(measured, target)) <= tolerance)
    except (TypeError, ValueError, OverflowError):
        return False


class SimPanTilt:
    """Pan-tilt simulado: va al objetivo a velocidad limitada (grados/s) si tiene torque."""

    def __init__(self, speed_dps: float = 90.0):
        self.pan = self.tilt = 0.0
        self.target = (0.0, 0.0)
        self.torque = False
        self.speed = speed_dps

    def update(self, dt: float):
        if not self.torque:
            return
        step = self.speed * dt
        tp, tt = self.target
        self.pan += max(-step, min(step, tp - self.pan))
        self.tilt += max(-step, min(step, tt - self.tilt))


class SimBase:
    """Base movil simulada: Nav2 'teletransporta' suavemente hacia el objetivo (0.5 m/s)."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, speed=0.5):
        self.x, self.y, self.yaw = x, y, yaw
        self.speed = speed
        self.goal: Optional[Tuple[float, float, float]] = None

    def set_goal(self, x, y, yaw):
        self.goal = (x, y, yaw)

    def update(self, dt: float) -> bool:
        """Avanza; devuelve True cuando llego."""
        if self.goal is None:
            return True
        gx, gy, gyaw = self.goal
        dx, dy = gx - self.x, gy - self.y
        d = math.hypot(dx, dy)
        step = self.speed * dt
        if d < 0.02 or step >= d:
            self.x, self.y, self.yaw, self.goal = gx, gy, gyaw, None
            return True
        self.x += step * dx / d
        self.y += step * dy / d
        self.yaw = math.atan2(dy, dx)
        return False


class SimMicrophone:
    """Microfono simulado con la MISMA semantica que whisper_node (escucha por pedido con id):

    - request(id) abre una escucha: espera a que la persona EMPIECE a hablar hasta start_timeout
      (el reloj se detiene mientras el robot habla: mute) y corta end_silence despues de que termina.
    - Lo que se dice sin escucha abierta o con el robot hablando se pierde (como en el real).
    - due() devuelve los resultados (id, status, texto): 'ok' o 'no_speech'. Siempre hay resultado.
    """

    WORD_SEC = 0.35  # duracion aproximada de cada palabra hablada

    def __init__(self, start_timeout: float = 8.0, end_silence: float = 1.1, asr_s: float = 1.3):
        self.start_timeout = start_timeout
        self.end_silence = end_silence
        self.asr_s = asr_s
        self.muted = False
        self.active: Optional[dict] = None          # {'id', 'deadline'}
        self.pending: List[Tuple[float, str, str, str]] = []  # (t_publicar, id, status, texto)
        self._last = None

    def request(self, rid: str, start_timeout: Optional[float] = None, end_silence: Optional[float] = None,
                now: Optional[float] = None) -> Optional[str]:
        """Abre una escucha. Devuelve el id reemplazado (que debe terminar como 'cancelled'), si habia."""
        now = time.time() if now is None else now
        old = self.active["id"] if self.active else None
        self.active = {"id": rid, "deadline": now + (start_timeout or self.start_timeout),
                       "end_silence": end_silence or self.end_silence}
        self._last = now
        return old

    def cancel(self, rid: str = "") -> Optional[str]:
        if self.active and rid in ("", "*", self.active["id"]):
            old, self.active = self.active["id"], None
            return old
        return None

    @property
    def listening(self) -> bool:
        return self.active is not None and not self.muted

    def hear(self, text: str, now: Optional[float] = None) -> bool:
        """El 'invitado' habla. Devuelve False si el robot no estaba escuchando (se pierde, como en el real)."""
        now = time.time() if now is None else now
        if self.active is None or self.muted:
            return False
        dur = max(0.5, len(text.split()) * self.WORD_SEC)
        self.pending.append((now + dur + self.active["end_silence"] + self.asr_s, self.active["id"], "ok", text))
        self.active = None
        return True

    def due(self, now: Optional[float] = None) -> List[Tuple[str, str, str]]:
        now = time.time() if now is None else now
        if self.active is not None:
            if self.muted and self._last is not None:
                self.active["deadline"] += now - self._last   # mientras el robot habla no corre la espera
            if now >= self.active["deadline"]:
                self.pending.append((now, self.active["id"], "no_speech", ""))
                self.active = None
        self._last = now
        out = [(rid, st, t) for tp, rid, st, t in self.pending if tp <= now]
        self.pending = [x for x in self.pending if x[0] > now]
        return out


class GripperStall:
    """Infer a contact stop from fresh measured Gazebo finger widths."""

    def __init__(self, still_mm: float = 0.6, still_s: float = 0.35, gap_mm: float = 3.0, fresh_s: float = 0.5):
        self.still_mm, self.still_s, self.gap_mm, self.fresh_s = still_mm, still_s, gap_mm, fresh_s
        self._anchor = None
        self._sample = None

    def measure(self, width_mm: float, now: float):
        if not math.isfinite(width_mm):
            return
        self._sample = (now, float(width_mm))
        if self._anchor is None or abs(width_mm - self._anchor[1]) > self.still_mm:
            self._anchor = (now, float(width_mm))

    def clear(self):
        """Forget stale finger contact after the simulated object detaches."""
        self._anchor = self._sample = None

    def measured_mm(self, now: float) -> Optional[float]:
        if self._sample is None or now - self._sample[0] > self.fresh_s:
            return None
        return self._sample[1]

    def block_mm(self, target_mm: float, now: float) -> Optional[float]:
        """Return the measured width only when the fingers stalled wider than their target."""
        width = self.measured_mm(now)
        if width is None or self._anchor is None:
            return None
        if width - target_mm < self.gap_mm or now - self._anchor[0] < self.still_s:
            return None
        return width


def gripper_stall_enabled_after_detach(detach_pending_release, target_mm, open_threshold_mm=95.0):
    """Keep a detached can from reappearing as finger contact until the open command arrives."""
    if not detach_pending_release:
        return True
    try:
        target, threshold = float(target_mm), float(open_threshold_mm)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(target) and math.isfinite(threshold) and target >= threshold


def restaurant_gripper_contact_block(target_mm, contact_mm, *, item_present,
                                     detach_pending_release, open_threshold_mm=95.0):
    """Avoid recreating a can contact after actor detach until the gripper opens."""
    try:
        target = float(target_mm)
        threshold = float(open_threshold_mm)
        contact = None if contact_mm is None else float(contact_mm)
    except (TypeError, ValueError, OverflowError):
        return None, bool(detach_pending_release)
    if not math.isfinite(target) or not math.isfinite(threshold):
        return None, bool(detach_pending_release)
    if detach_pending_release:
        if target < threshold:
            return None, True
        return None, False
    if not item_present or contact is None or not math.isfinite(contact):
        return None, False
    return contact, False


def map_odom_relay_stamp(now_s, received_s, *, max_age_s=0.5, max_future_s=0.25,
                         lead_s=0.15, max_lead_s=0.15):
    """Future-date a fresh simulated map->odom estimate by a small bounded margin."""
    try:
        now, received, max_age, max_future, lead, max_lead = map(
            float, (now_s, received_s, max_age_s, max_future_s, lead_s, max_lead_s))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in
               (now, received, max_age, max_future, lead, max_lead)):
        return None
    age = now - received
    if age < -max_future or age > max_age or max_age < 0.0 or max_future < 0.0 or max_lead < 0.0:
        return None
    return now + min(max(0.0, lead), max_lead)


def gripper_width_from_joint_states(names: Sequence[str], positions: Sequence[float]) -> Optional[float]:
    """Read the Piper finger opening from JointState, including Gazebo scoped names.

    Gazebo may publish joint names as ``model::joint7`` rather than the bare URDF
    names.  Only accept an unambiguous pair, since an arbitrary suffix match could
    combine fingers from another model.
    """
    if len(names) != len(positions):
        return None
    selected = {}
    for name, position in zip(names, positions):
        joint = str(name).rsplit("::", 1)[-1]
        if joint not in ("joint7", "joint8"):
            continue
        if joint in selected:
            return None
        try:
            value = float(position)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        selected[joint] = value
    if set(selected) != {"joint7", "joint8"}:
        return None
    return 1000.0 * (selected["joint7"] - selected["joint8"])
