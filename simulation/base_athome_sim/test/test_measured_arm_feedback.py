"""Regression tests for measured Gazebo arm feedback and status reporting."""
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))
from base_athome_sim.measured_arm_feedback import (
    GZ_ARM_SETTLE_TIMEOUT_S,
    arm_command_state_from_measured_sample,
    arm_status_from_measured_sample,
    measured_goal_deviation_deg,
    measured_goal_is_settled,
    measured_sample_stamp_advances,
)


def sample(joints=None, gripper=72.0, sec=10, nanosec=20):
    return {
        "joints_deg": list(joints if joints is not None else [1, 2, 3, 4, 5, 6]),
        "gripper_mm": gripper,
        "stamp_sec": sec,
        "stamp_nanosec": nanosec,
    }


def test_arm_status_feedback_freshness_boundary_is_half_second():
    measured = sample()
    assert arm_status_from_measured_sample(
        measured, 0.5, enabled=True, is_moving=False)["feedback_ok"] is True
    assert arm_status_from_measured_sample(
        measured, 0.5001, enabled=True, is_moving=False)["feedback_ok"] is False


def test_measured_source_timestamps_must_advance():
    previous = sample(sec=10, nanosec=200)
    repeated = sample(sec=10, nanosec=200)
    advanced = sample(sec=10, nanosec=201)
    assert not measured_sample_stamp_advances(previous, repeated)
    assert measured_sample_stamp_advances(previous, advanced)
    assert not measured_sample_stamp_advances(
        previous, sample(sec=10, nanosec=1_000_000_000))
    assert measured_sample_stamp_advances(None, previous)


def test_goal_settle_requires_fresh_measurement_within_tolerance_and_stationary():
    measured = sample([10.0] * 6)
    assert measured_goal_deviation_deg(measured, 0.1, [11.0] * 6) == 1.0
    assert measured_goal_is_settled(measured, 0.1, [11.0] * 6, is_moving=False)
    assert not measured_goal_is_settled(measured, 0.1, [13.0] * 6, is_moving=False)
    assert not measured_goal_is_settled(measured, 0.1, [11.0] * 6, is_moving=True)
    assert not measured_goal_is_settled(measured, 0.51, [11.0] * 6, is_moving=False)


def test_arm_command_initialization_uses_first_valid_measurement():
    measured = sample([1, 2, 3, 4, 5, 6], gripper=55.0)
    assert arm_command_state_from_measured_sample(measured) == ([1, 2, 3, 4, 5, 6], 55.0)
    assert arm_command_state_from_measured_sample(sample([0] * 5)) is None
    assert GZ_ARM_SETTLE_TIMEOUT_S == 10.0


def test_arm_status_reports_measured_values_when_command_differs():
    commanded_joints = [90.0] * 6
    commanded_gripper = 200.0
    measured = sample([1, 2, 3, 4, 5, 6], gripper=72.0)

    status = arm_status_from_measured_sample(
        measured, 0.1, enabled=True, is_moving=False)

    assert status["feedback_ok"] is True
    assert status["joints_deg"] == measured["joints_deg"]
    assert status["joints_deg"] != commanded_joints
    assert status["gripper_mm"] == measured["gripper_mm"]
    assert status["gripper_mm"] != commanded_gripper


def test_arm_status_rejects_old_or_missing_feedback():
    measured = sample()
    stale = arm_status_from_measured_sample(
        measured, 0.51, enabled=True, is_moving=False)
    missing = arm_status_from_measured_sample(
        None, float("inf"), enabled=True, is_moving=False)

    assert stale["feedback_ok"] is False
    assert missing["feedback_ok"] is False
    assert missing["joints_deg"] == []
    assert missing["gripper_mm"] is None
