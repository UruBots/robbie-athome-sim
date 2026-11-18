"""Match Restaurant actor replies to the ASR listening window that requested them."""
import math
import threading
import time


def counter_can_pose_world(robot_pose, arm_base_y_m, *, counter_front_x_m=-1.80,
                           height_m=1.015, robot_right_offset_m=0.04):
    """Place the can 4 cm to the barman's left (the robot's right) of arm center."""
    try:
        x, y, yaw = (float(value) for value in robot_pose)
        arm_y = float(arm_base_y_m)
        edge_x, height = float(counter_front_x_m), float(height_m)
        right_offset = float(robot_right_offset_m)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x, y, yaw, arm_y, edge_x, height, right_offset)):
        return None
    c, s = math.cos(yaw), math.sin(yaw)
    if abs(c) < 0.5:
        return None
    # Local -Y is robot-right; solve world_x == bar edge after this lateral shift.
    can_y = arm_y - right_offset
    base_x = (edge_x - x + s * can_y) / c
    return (x + c * base_x - s * can_y,
            y + s * base_x + c * can_y,
            height)


def customer_order_key(snapshot):
    """Return the global delivery slot shared by barman staging and diner pickup.

    ``customer_idx`` is local to a search round and may change or reset during
    a delivery. ``served`` advances only after the diner takes their order, so
    it is the stable identity throughout staging and pickup.
    """
    try:
        served = int(snapshot.get("served", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    return (served,)


def actor_freshness_now(node):
    """Clock shared with ROS timestamps in simulation; wall monotonic on hardware."""
    try:
        if bool(node.get_parameter("use_sim_time").value):
            return node.get_clock().now().nanoseconds * 1e-9
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return time.monotonic()


def fresh_tf_receipt_age(now, received_at, max_age=1.0):
    """Validate transform freshness using ROS-clock receipt time, not TF header time."""
    try:
        age, limit = float(now) - float(received_at), float(max_age)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(age) or not math.isfinite(limit) or age < 0.0 or age > limit:
        return None
    return age


def find_staged_tray_order(staged, snapshot, items):
    """Find a staged record using the same logical customer key as the mission."""
    key = customer_order_key(snapshot)
    if key is None:
        return None
    wanted = set(items)
    return next((record for staged_key, record in staged.items()
                 if staged_key[:1] == key and wanted.issubset(set(record.get("tray_items", ())))), None)


class CustomerPositionCache:
    """Carry an observed approach target into Restaurant's later delivery state."""

    def __init__(self):
        self._positions = {}

    def observe(self, state):
        snapshot = dict(state)
        try:
            index = int(snapshot.get("customer_idx", 0))
        except (TypeError, ValueError):
            return snapshot
        info = snapshot.get("info") or {}
        info = dict(info) if isinstance(info, dict) else {}
        point = info.get("customer_xy")
        if snapshot.get("state") == "APPROACHING_CUSTOMER":
            try:
                xy = tuple(float(value) for value in point)
            except (TypeError, ValueError, OverflowError):
                xy = ()
            if len(xy) == 2 and all(math.isfinite(value) for value in xy):
                self._positions[index] = xy
        elif index in self._positions:
            info["customer_xy"] = list(self._positions[index])
        snapshot["info"] = info
        return snapshot


class DynamicCanFallTracker:
    """Report a dynamic can only after it falls below the counter-level threshold."""

    def __init__(self, floor_threshold_m=0.3):
        self.floor_threshold_m = float(floor_threshold_m)
        self._above_floor = set()
        self._fallen = set()

    def observe(self, model, x, y, z):
        try:
            x, y, z = float(x), float(y), float(z)
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(value) for value in (x, y, z)):
            return False
        model = str(model)
        if z >= self.floor_threshold_m:
            self._above_floor.add(model)
            self._fallen.discard(model)
            return False
        if model not in self._above_floor or model in self._fallen:
            return False
        self._fallen.add(model)
        return True


class SingleFlight:
    """Run one side effect per key while concurrent callers await its result."""

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = {}
        self._completed = {}

    def run(self, key, action):
        with self._condition:
            if key in self._completed:
                return self._completed[key]
            attempt = self._pending.get(key)
            if attempt is None:
                attempt = {"done": False, "result": False}
                self._pending[key] = attempt
                owner = True
            else:
                owner = False

            if not owner:
                while not attempt["done"]:
                    self._condition.wait()
                return attempt["result"]

        try:
            result = bool(action())
        except BaseException:
            with self._condition:
                attempt["done"] = True
                self._pending.pop(key, None)
                self._condition.notify_all()
            raise

        with self._condition:
            attempt["result"] = result
            attempt["done"] = True
            self._pending.pop(key, None)
            if result:
                self._completed[key] = True
            self._condition.notify_all()
        return result


def barman_counter_item_for_request(state, request):
    """Return the single item to transfer from the counter after a failed pick."""
    if state.get("state") != "WAITING_BARMAN":
        return ""
    item = str((state.get("info") or {}).get("item", "")).strip()
    # The mission state names the item explicitly; the listen request may carry no prompt
    # text (opus_nuc_04: transfer never ran because the prompt check never matched).
    return item


def barman_gripper_handoff_ready(state, arm_status, arm_age_s, facing_barman):
    """Allow the sim barman handoff only for a fresh, open gripper facing the bar."""
    if state.get("state") != "WAITING_BARMAN":
        return False
    info = state.get("info") or {}
    if info.get("placement") != "gripper" or not str(info.get("item", "")).strip():
        return False
    try:
        age = float(arm_age_s)
        width = float(arm_status["gripper_mm"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (arm_status.get("feedback_ok") is True and math.isfinite(age)
            and 0.0 <= age < 1.0 and math.isfinite(width) and width >= 90.0
            and bool(facing_barman))


def barman_gripper_handoff_item(state, staged_record=None):
    """Return requested item for a gripper handoff even if counter staging was skipped."""
    if state.get("state") != "WAITING_BARMAN":
        return None
    info = state.get("info") or {}
    item = str(info.get("item", "")).strip()
    if info.get("placement") != "gripper" or not item:
        return None
    if staged_record is not None and item in staged_record.get("tray_items", ()):
        return None
    return item


def barman_gripper_confirmation_action(state, placed_handoffs):
    """Place once on confirmation one; answer confirmation two only if placement succeeded."""
    if state.get("state") != "WAITING_BARMAN":
        return "reject", None
    info = state.get("info") or {}
    item = str(info.get("item", "")).strip()
    if info.get("placement") != "gripper" or not item:
        return "reject", None
    try:
        step = int(info.get("confirmation_step", 1))
    except (TypeError, ValueError, OverflowError):
        return "reject", None
    key = (customer_order_key(state), item)
    if step == 1:
        return "place", key
    if step == 2 and key in placed_handoffs:
        return "confirm", key
    return "reject", key


def customer_gripper_take_ready(state, robot_pose, customer_xy, can_name, speech):
    """Allow a spoken handover while the robot is physically within diner reach."""
    if state.get("state") not in {"RELEASING_ITEM", "WAITING_CUSTOMER"}:
        return False
    if "one. two. three." not in str(speech).lower():
        return False
    if not can_name:
        return False
    try:
        pose = tuple(float(value) for value in robot_pose)
        target = tuple(float(value) for value in customer_xy)
    except (TypeError, ValueError, OverflowError):
        return False
    if (len(pose) != 3 or len(target) != 2
            or not all(math.isfinite(v) for v in (*pose, *target))):
        return False
    return math.dist(pose[:2], target) < 1.35


def tray_model_names(items, slot, order=()):
    """Map requested tray contents to deterministic Restaurant Gazebo models."""
    names = tuple(str(item).strip() for item in items if str(item).strip())
    try:
        slot = max(1, int(slot))
    except (TypeError, ValueError, OverflowError):
        return ()
    full_order = tuple(str(item).strip() for item in order if str(item).strip())
    if len(full_order) != 2 or any(item not in full_order for item in names):
        return ()
    models = (f"restaurant_counter_can_{slot}", f"restaurant_tray_item_{slot}")
    model_by_item = dict(zip(full_order, models))
    return tuple((item, model_by_item[item]) for item in names)


def record_gripper_handoff(staged, key, item, can_name):
    """Record a direct Plan B can so a later failed grasp can move it to the tray."""
    item = str(item).strip()
    can_name = str(can_name).strip()
    if not isinstance(staged, dict) or key is None or not item or not can_name:
        return False
    record = staged.get(key)
    if record is None:
        record = {"can": can_name, "tray": f"{can_name}_on_tray", "items": (),
                  "tray_items": (), "tray_names": ()}
    if record.get("can") != can_name:
        return False
    record = dict(record)
    items = tuple(record.get("items", ()))
    if item not in items:
        record["items"] = items + (item,)
    record.setdefault("tray_items", ())
    record.setdefault("tray_names", ())
    staged[key] = record
    return True


def can_pose_matches_tcp(can_xyz, tcp_xyz, max_error_m=0.08):
    """Require a handoff can's measured Gazebo world pose to match the TCP pose."""
    try:
        can_xyz = tuple(float(value) for value in can_xyz)
        tcp_xyz = tuple(float(value) for value in tcp_xyz)
        max_error_m = float(max_error_m)
    except (TypeError, ValueError, OverflowError):
        return False
    if (len(can_xyz) != 3 or len(tcp_xyz) != 3 or not math.isfinite(max_error_m)
            or max_error_m <= 0.0
            or not all(math.isfinite(value) for value in (*can_xyz, *tcp_xyz))):
        return False
    return math.dist(can_xyz, tcp_xyz) <= max_error_m


def transfer_staged_item_to_tray(record, item, tray_name, create_item, remove_counter_item,
                                 remove_tray_item):
    """Move a failed-pick counter item to the tray, updating the matching staged record."""
    if not isinstance(record, dict):
        return None
    item = str(item).strip()
    items = tuple(record.get("items", ()))
    tray_items = tuple(record.get("tray_items", ()))
    tray_names = tuple(record.get("tray_names", ()))
    if not item or item not in items or len(tray_items) != len(tray_names):
        return None
    if item in tray_items:
        return dict(record)
    if not create_item():
        return None
    if not remove_counter_item():
        remove_tray_item()
        return None
    updated = dict(record)
    updated["tray_items"] = tray_items + (item,)
    updated["tray_names"] = tray_names + (str(tray_name),)
    return updated


class ListenRequestGate:
    """Release an actor reply only for an active, matching VoiceClient request."""

    def __init__(self, *, clock=time.monotonic, default_start_timeout=8.0, early_event_ttl=2.0):
        self._clock = clock
        self._default_start_timeout = float(default_start_timeout)
        self._early_event_ttl = float(early_event_ttl)
        self._condition = threading.Condition()
        self._requests = {}
        self._opened = {}

    def register_request(self, request):
        """Remember a request id and its speech-start window; reject uncorrelated requests."""
        request_id = str(request.get("id", "")).strip()
        if not request_id:
            return ""
        try:
            start_timeout = float(request.get("start_timeout", self._default_start_timeout))
        except (TypeError, ValueError):
            start_timeout = self._default_start_timeout
        if start_timeout <= 0:
            start_timeout = self._default_start_timeout
        now = self._clock()
        with self._condition:
            self._prune(now)
            self._requests[request_id] = {"start_timeout": start_timeout, "registered_at": now}
            self._condition.notify_all()
        return request_id

    def mark_listening(self, event):
        """Record the matching VoiceClient listening acknowledgement, including early delivery."""
        if event.get("event") != "listening":
            return False
        request_id = str(event.get("id", "")).strip()
        if not request_id:
            return False
        now = self._clock()
        with self._condition:
            self._prune(now)
            request = self._requests.get(request_id)
            self._opened[request_id] = now
            self._condition.notify_all()
        return True

    def response_ready(self, request_id, *, now=None):
        """Return true only while the same request's speech-start window remains open."""
        now = self._clock() if now is None else now
        with self._condition:
            request = self._requests.get(request_id)
            opened_at = self._opened.get(request_id)
            return (request is not None and opened_at is not None
                    and 0 <= now - opened_at <= request["start_timeout"])

    def wait_until_listening(self, request_id, *, open_timeout=2.0):
        """Wait briefly for the ASR open event, then require an unexpired speech-start window."""
        deadline = self._clock() + max(0.0, float(open_timeout))
        with self._condition:
            while request_id not in self._opened:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            opened_at = self._opened[request_id]
            request = self._requests.get(request_id)
            now = self._clock()
            return (request is not None and 0 <= now - opened_at <= request["start_timeout"])

    def _prune(self, now):
        for request_id, opened_at in tuple(self._opened.items()):
            request = self._requests.get(request_id)
            ttl = request["start_timeout"] if request is not None else self._early_event_ttl
            if now - opened_at > ttl:
                self._opened.pop(request_id, None)
        for request_id, request in tuple(self._requests.items()):
            if now - request["registered_at"] > request["start_timeout"] + self._early_event_ttl:
                self._requests.pop(request_id, None)
                self._opened.pop(request_id, None)
