"""Tests de la fase post-grasp del mostrador, sin ROS ni Gazebo."""

import ast
import math
import re
import sys
from pathlib import Path

import pytest

SIM_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SIM_DIR.parent / "scripts"
sys.path.insert(0, str(SIM_DIR))

from base_athome_sim.counter_carry import (  # noqa: E402
    CAN_SEATED_Z, CARRY_POSE, DEFAULT_RETREAT_M, POSTGRASP_POSE,
    CounterCarry, can_is_seated,
)


class World:
    """Base y lata con acople simulado; la lata sigue a la base si esta asida."""

    def __init__(self, *, attached=True, lift_m=0.08, rear=0.8, follow=True,
                 detach_after=0, drop_after=0, can_follow_error=0.0,
                 pose_ok=True, carry_pose_ok=True, steps_per_call=None):
        self.attached_state = attached
        self.lift_m = lift_m
        self.rear = rear
        self.follow = follow
        self.detach_after = detach_after
        self.drop_after = drop_after
        self.can_follow_error = can_follow_error
        self.pose_ok = pose_ok
        self.carry_pose_ok = carry_pose_ok
        self.steps_per_call = steps_per_call
        self.base = [0.0, 0.0]
        self.can = [-1.78, 0.0, CAN_SEATED_Z]
        self.can_lifted_z = CAN_SEATED_Z + lift_m
        self.moves = []
        self.twists = []
        self.steps = 0
        self.lifted = False
        self.now = 0.0

    def attached(self):
        if self.detach_after and self.steps >= self.detach_after:
            return False
        return self.attached_state

    def base_pose(self):
        return (self.base[0], self.base[1], 0.0)

    def can_pose(self):
        drift = self.base[0] if self.follow else 0.0
        error = self.can_follow_error if self.steps else 0.0
        if self.drop_after and self.steps >= self.drop_after:
            z = CAN_SEATED_Z
        elif self.lifted:
            z = self.can_lifted_z
        else:
            z = self.can[2]
        return (self.can[0] + drift + error, self.can[1] + self.base[1], z)

    def rear_clearance(self):
        return self.rear

    def publish_twist(self, linear_x):
        self.twists.append(linear_x)
        if linear_x < 0:
            step = self.steps_per_call or (0.04 * 0.25)
            self.base[0] += linear_x * step * 20.0
            self.steps += 1

    def move_holding(self, name):
        self.moves.append(name)
        if name == POSTGRASP_POSE and self.lift_m > 0:
            self.lifted = True
        if name == CARRY_POSE:
            return self.carry_pose_ok, "pose"
        return self.pose_ok, "pose"

    def step(self, dt):
        self.now += dt

    def clock(self):
        return self.now


def carry(world, **kwargs):
    return CounterCarry(
        can_pose=world.can_pose,
        base_pose=world.base_pose,
        attached=world.attached,
        rear_clearance=world.rear_clearance,
        publish_twist=world.publish_twist,
        move_holding=world.move_holding,
        sleep=world.step,
        monotonic=world.clock,
        **kwargs)


def test_retreat_distance_is_bounded_to_15_30_cm():
    for invalid in (0.10, 0.31, -0.20):
        with pytest.raises(ValueError):
            carry(World(), retreat_m=invalid)
    assert carry(World(), retreat_m=0.15).retreat_m == pytest.approx(0.15)
    assert carry(World(), retreat_m=0.30).retreat_m == pytest.approx(0.30)


def test_requires_attach_before_any_motion():
    world = World(attached=False)
    ok, reason, evidence = carry(world).run()
    assert not ok
    assert "acoplada" in reason
    assert world.moves == []
    assert world.twists == []


def test_lift_and_carry_use_holding_poses():
    world = World(steps_per_call=0.0125)
    ok, _reason, evidence = carry(world).run()
    assert ok
    assert world.moves == [POSTGRASP_POSE, CARRY_POSE]
    assert evidence.pose == CARRY_POSE


def test_carry_can_validate_lift_after_pick_item_already_reached_postgrasp():
    world = World(steps_per_call=0.0125)
    initial_can_pose = world.can_pose()
    world.lifted = True  # RestaurantArm.pick_item already moved through postgrasp.

    ok, _reason, evidence = carry(
        world, initial_can_pose=initial_can_pose, retreat_m=0.15).run()

    assert ok
    assert evidence.lift_m == pytest.approx(world.lift_m)


def test_successful_carry_reports_evidence():
    world = World(steps_per_call=0.0125)
    ok, _reason, evidence = carry(world).run()
    assert ok
    assert evidence.attached is True
    assert evidence.lift_m >= 0.05
    assert DEFAULT_RETREAT_M - 0.02 <= evidence.retreat_m <= DEFAULT_RETREAT_M + 0.02
    assert evidence.can_follow_error_m <= 0.10
    assert evidence.can_z_m >= CAN_SEATED_Z + 0.05
    assert evidence.reason == ""


def test_twist_is_zeroed_after_success_and_failure():
    for world in (World(steps_per_call=0.0125), World(rear=0.05)):
        carry(world).run()
        assert world.twists
        assert world.twists[-1] == 0.0


def test_aborts_when_lift_is_insufficient():
    world = World(lift_m=0.01)
    ok, reason, _evidence = carry(world).run()
    assert not ok
    assert "levantamiento" in reason
    assert world.twists == []


def test_aborts_when_can_drops_during_retreat():
    world = World(steps_per_call=0.0125, drop_after=1)
    ok, reason, _evidence = carry(world).run()
    assert not ok
    assert "cayo" in reason


def test_aborts_when_can_does_not_follow_base():
    world = World(steps_per_call=0.0125, follow=False)
    ok, reason, _evidence = carry(world).run()
    assert not ok
    assert "no acompano" in reason


def test_aborts_when_can_drifts_beyond_tolerance():
    world = World(steps_per_call=0.0125, can_follow_error=0.25)
    ok, reason, _evidence = carry(world).run()
    assert not ok
    assert "no acompano" in reason


def test_aborts_when_rear_clearance_is_low():
    world = World(rear=0.10)
    ok, reason, evidence = carry(world).run()
    assert not ok
    assert "margen trasero" in reason
    assert evidence.rear_clearance_m == pytest.approx(0.10)


def test_aborts_when_detached_mid_retreat():
    world = World(steps_per_call=0.0125, detach_after=2)
    ok, reason, _evidence = carry(world).run()
    assert not ok
    assert "desacoplo" in reason


def test_aborts_when_carry_pose_fails():
    world = World(steps_per_call=0.0125, carry_pose_ok=False)
    ok, reason, evidence = carry(world).run()
    assert not ok
    assert CARRY_POSE in reason
    assert evidence.pose == CARRY_POSE


def test_aborts_on_retreat_timeout():
    world = World(steps_per_call=0.0001)
    ok, reason, _evidence = carry(world, timeout_s=0.05).run()
    assert not ok
    assert "timeout" in reason
    assert math.hypot(*world.base) < 0.15


def test_anti_floating_seated_check():
    assert can_is_seated(CAN_SEATED_Z)
    assert can_is_seated(1.03)
    assert not can_is_seated(1.48)
    assert not can_is_seated(None)
    assert not can_is_seated(float("nan"))


def test_evidence_is_finite_for_success():
    world = World(steps_per_call=0.0125)
    ok, _reason, evidence = carry(world).run()
    assert ok
    for value in (evidence.lift_m, evidence.retreat_m, evidence.can_follow_error_m, evidence.can_z_m):
        assert math.isfinite(value)


def _embedded_debug_body():
    src = (SCRIPTS_DIR / "sim_counter_debug.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "test_script" for t in node.targets)):
            assert isinstance(node.value, ast.JoinedStr)
            body = "".join(part.value if isinstance(part, ast.Constant) else "0"
                           for part in node.value.values)
            return re.sub(r"\{args\.[a-z_]*(?::[^}]*)?\}", "0", body).replace("{{", "{").replace("}}", "}")
    raise AssertionError("sim_counter_debug.py no define test_script")


def test_debug_scene_wires_bounded_postgrasp_helper():
    body = _embedded_debug_body()
    compile(body, "<sim_counter_debug embedded>", "exec")
    assert "CounterCarry(" in body
    assert "move_holding=lambda name: node.go_to_holding(name)" in body
    assert "node.go_to_blocking('restaurant_look')" in body
    assert "node.counter_look_pose_active = True" in body
    assert "memory_xyz" not in body  # RestaurantArm retains the measured target itself.
    assert "CARRY: ok=" in body


def test_debug_scene_defaults_put_can_in_front_and_retreat_in_range():
    src = (SCRIPTS_DIR / "sim_counter_debug.py").read_text()
    assert 'default=0.0' in src
    assert 'default=-0.90' in src
    assert 'default=180.0' in src
    assert '--retreat-m' in src
    assert 'default=0.20' in src
    assert '0.15 <= args.retreat_m <= 0.30' in src
