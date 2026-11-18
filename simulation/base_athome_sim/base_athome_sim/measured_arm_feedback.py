"""Helpers for reporting measured Gazebo arm feedback without inventing fresh state."""
import math


MAX_ARM_FEEDBACK_AGE_S = 0.5
GZ_ARM_SETTLE_TIMEOUT_S = 10.0


def arm_command_state_from_measured_sample(sample):
    """Return measured arm joint/gripper positions for one-time Gazebo initialization."""
    try:
        joints = [float(value) for value in sample["joints_deg"]]
        gripper = float(sample["gripper_mm"])
        if (len(joints) != 6 or not all(math.isfinite(value) for value in joints)
                or not math.isfinite(gripper)):
            return None
        return joints, gripper
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def measured_sample_stamp_ns(sample):
    """Return a valid source timestamp in nanoseconds, or None for malformed samples."""
    try:
        sec = int(sample["stamp_sec"])
        nanosec = int(sample["stamp_nanosec"])
        if sec < 0 or not 0 <= nanosec < 1_000_000_000:
            return None
        return sec * 1_000_000_000 + nanosec
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def measured_sample_stamp_advances(previous, current):
    """Accept only Gazebo samples whose source timestamp advances monotonically."""
    current_stamp = measured_sample_stamp_ns(current)
    if current_stamp is None:
        return False
    if previous is None:
        return True
    previous_stamp = measured_sample_stamp_ns(previous)
    return previous_stamp is not None and current_stamp > previous_stamp


def measured_goal_deviation_deg(sample, age_s, goal_joints_deg, *, max_age_s=MAX_ARM_FEEDBACK_AGE_S):
    """Return maximum measured joint error for a fresh six-joint Gazebo sample."""
    try:
        age = float(age_s)
        joints = [float(value) for value in sample["joints_deg"]]
        goal = [float(value) for value in goal_joints_deg]
        if (not math.isfinite(age) or not 0.0 <= age <= float(max_age_s)
                or len(joints) != 6 or len(goal) != 6
                or not all(math.isfinite(value) for value in joints + goal)):
            return None
        return max(abs(actual - target) for actual, target in zip(joints, goal))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def measured_goal_is_settled(sample, age_s, goal_joints_deg, *, is_moving, tolerance_deg=2.0):
    """Require fresh measured joints within tolerance and no recent Gazebo movement."""
    try:
        tolerance = float(tolerance_deg)
        deviation = measured_goal_deviation_deg(sample, age_s, goal_joints_deg)
        return (deviation is not None and math.isfinite(tolerance) and tolerance > 0.0
                and deviation <= tolerance and is_moving is False)
    except (TypeError, ValueError, OverflowError):
        return False


def arm_status_from_measured_sample(sample, age_s, *, enabled, is_moving,
                                    max_age_s=MAX_ARM_FEEDBACK_AGE_S):
    """Build the arm status contract from one measured sample and its receipt age."""
    age = float(age_s)
    fresh = math.isfinite(age) and 0.0 <= age <= float(max_age_s)
    valid_sample = isinstance(sample, dict)
    if valid_sample:
        try:
            joints = [float(value) for value in sample["joints_deg"]]
            gripper_mm = float(sample["gripper_mm"])
            stamp_sec = int(sample["stamp_sec"])
            stamp_nanosec = int(sample["stamp_nanosec"])
            valid_sample = (len(joints) == 6 and all(math.isfinite(value) for value in joints)
                            and math.isfinite(gripper_mm)
                            and 0 <= stamp_nanosec < 1_000_000_000)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid_sample = False
    else:
        joints, gripper_mm, stamp_sec, stamp_nanosec = [], None, 0, 0
    if not valid_sample:
        joints, gripper_mm, stamp_sec, stamp_nanosec = [], None, 0, 0
    return {
        "is_moving": bool(is_moving),
        "enabled": bool(enabled),
        "feedback_ok": bool(fresh and valid_sample),
        "gripper_mm": gripper_mm,
        "joints_deg": joints,
        "sim": True,
        "measured_stamp": {"sec": stamp_sec, "nanosec": stamp_nanosec},
        "feedback_age_s": round(age, 3) if math.isfinite(age) else None,
    }


def apply_attached_object_gripper_width(status, *, attached, diameter_mm):
    """Model the can as blocking the simulated gripper after Gazebo attach succeeds."""
    if not attached or not isinstance(status, dict) or status.get("feedback_ok") is not True:
        return status
    try:
        measured = float(status["gripper_mm"])
        diameter = float(diameter_mm)
    except (KeyError, TypeError, ValueError, OverflowError):
        return status
    if not math.isfinite(measured) or not math.isfinite(diameter) or diameter <= 2.0:
        return status
    return dict(status, gripper_mm=diameter)
