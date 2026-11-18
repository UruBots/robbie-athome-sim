"""Restaurant's fixed rendered diners and barman, aligned with restaurant.sh.

Geometry comes from OpenRobotics Male visitor v2 (CC0-1.0), with static poses
built by models/humans/download_restaurant.py. All coordinates are world metres.
"""
import math

CUSTOMERS = ((2.0, 2.8, 'right', 'seated'), (3.0, -2.9, 'left', 'seated'),
             (-2.6, 0.0, 'right', 'standing'))
RENDERED_MODELS = (
    ('human_diner_north', (2.0, 2.8, 0.0, -1.57079633), 'male_visitor', 'seated_down'),
    ('human_diner_south', (3.0, -2.9, 0.0, 1.57079633), 'female_visitor', 'seated_down'),
    ('human_barman', (-2.6, 0.0, 0.0, 0.0), 'nurse_female', 'standing_right'),
)
BARMAN_XY = (-2.6, 0.0)
LOWERED_RENDERED_MODELS = ('seated_down', 'seated_down', 'standing_down')
RAISED_RENDERED_MODELS = ('seated_right', 'seated_left', 'standing_right')


class ParticipantGestures:
    """Drive visible restaurant call gestures from the mission dialogue state."""
    def __init__(self):
        self.customer_by_mission_index = {}
        self.lowered = set()
        self.raised = {2}  # the barman alone signals while Robbie searches for the bar
        self.served = 0

    def observe(self, state, customers=CUSTOMERS):
        name = state.get('state', '')
        info = state.get('info') or {}
        mission_idx = int(state.get('customer_idx', 0))
        xy = info.get('customer_xy')
        if name == 'APPROACHING_CUSTOMER' and isinstance(xy, (list, tuple)) and len(xy) == 2:
            nearest = min(range(min(2, len(customers))),
                          key=lambda i: math.dist(xy, customers[i][:2]), default=None)
            if nearest is not None and math.dist(xy, customers[nearest][:2]) < .8:
                self.customer_by_mission_index[mission_idx] = nearest
        served = int(state.get('served', self.served))
        if name not in ('STARTING', 'FINDING_BARMAN', 'CONFIRMING_BARMAN'):
            self.raised.discard(2)
        if served > self.served:
            physical = self.customer_by_mission_index.get(mission_idx)
            if physical is not None:
                self.lowered.add(physical)
                self.raised.discard(physical)
            self.served = served
        if name == 'RETURNING_TO_BAR' and served > 0:
            self.raised.update(i for i in range(2) if i not in self.lowered)
        if name == 'DETECTING_CUSTOMERS' and served == 0:
            self.raised.update(i for i in range(2) if i not in self.lowered)
        if name == 'DETECTING_CUSTOMERS' and served > 0:
            # Mission customer_idx is local to each search round and resets to
            # zero. After serving diner one, the remaining unserved diner must
            # signal again when the mission starts another detection round.
            self.raised.update(i for i in range(2) if i not in self.lowered)
        return frozenset(self.raised)


def gesture_model_sdf(index, raised):
    """Return the licensed static mesh replacement for one visible call gesture."""
    import xml.sax.saxutils as xml
    name, (x, y, z, yaw), model, _ = RENDERED_MODELS[index]
    mesh = RAISED_RENDERED_MODELS[index] if raised else LOWERED_RENDERED_MODELS[index]
    uri = f'model://{model}/meshes/{mesh}.dae'
    return (f'<sdf version="1.9"><model name="{xml.escape(name)}"><static>true</static>'
            f'<pose>{x} {y} {z} 0 0 {yaw}</pose><link name="body"><visual name="textured_human">'
            f'<geometry><mesh><uri>{uri}</uri></mesh></geometry></visual></link></model></sdf>')


def lowered_model_sdf(index):
    """Compatibility helper for the served-customer replacement path."""
    return gesture_model_sdf(index, False)


def rendered_restaurant():
    import os
    return os.environ.get('ATHOME_SIM_WORLD') == 'restaurant.sdf'


def reply_for_state(state):
    """A diner must reject the barman role; never reward a wrong visual identity."""
    name = state.get('state', '')
    if name == 'CONFIRMING_BARMAN':
        xy = state.get('info', {}).get('xy')
        if xy is None or len(xy) != 2:
            return 'robot no'
        return 'robot yes' if math.hypot(xy[0] + 2.6, xy[1]) < .8 else 'robot no'
    if name == 'TAKING_ORDER':
        orders = ('coke and water', 'juice and tea')
        # Some mission cycles restart customer_idx at zero after a delivery.
        # The served count still identifies the next diner in the campaign.
        index = max(int(state.get('customer_idx', 0)), int(state.get('served', 0)))
        index = max(0, min(index, 1))
        return 'my order is ' + orders[index]
    return 'robot yes'


def ordered_items_for_state(state):
    """Return the two requested Restaurant items only at the barman placement event."""
    if state.get('state') != 'ORDERING_BARMAN':
        return ()
    info = state.get('info') or {}
    items = info.get('items')
    if info.get('use_tray') is not True or not isinstance(items, (list, tuple)) or len(items) < 2:
        return ()
    return tuple(str(item).strip() for item in items[:2] if str(item).strip())


def barman_tray_items_for_request(state, request):
    """Return a full order only for the explicit fallback request to use the tray."""
    if state.get('state') != 'WAITING_BARMAN':
        return ()
    prompt = str((request or {}).get('prompt', '')).casefold()
    if 'please place the' not in prompt or 'on my tray' not in prompt:
        return ()
    items = (state.get('info') or {}).get('items')
    if not isinstance(items, (list, tuple)) or len(items) != 2:
        return ()
    normalized = tuple(str(item).strip() for item in items)
    return normalized if all(normalized) else ()


def tray_items_for_state(state):
    """Return the items the customer has been asked to take from the tray."""
    if state.get('state') != 'WAITING_CUSTOMER':
        return ()
    items = (state.get('info') or {}).get('items')
    if not isinstance(items, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in items if str(item).strip())


def pose_faces_target(pose, target_xy, *, max_distance=1.5, max_heading_error=math.radians(35)):
    """Check the simulated robot is close to and facing a Restaurant participant."""
    if pose is None or len(pose) != 3 or target_xy is None or len(target_xy) != 2:
        return False
    x, y, yaw = (float(value) for value in pose)
    tx, ty = (float(value) for value in target_xy)
    if not all(math.isfinite(value) for value in (x, y, yaw, tx, ty)):
        return False
    if math.hypot(tx - x, ty - y) >= max_distance:
        return False
    bearing = math.atan2(ty - y, tx - x)
    heading_error = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
    return abs(heading_error) <= max_heading_error


def customer_xy_for_state(state):
    """Return the explicitly observed pose; mission indexes are not physical table ids."""
    info = state.get('info') or {}
    point = info.get('customer_xy')
    if isinstance(point, (list, tuple)) and len(point) == 2:
        try:
            xy = tuple(float(value) for value in point)
        except (TypeError, ValueError, OverflowError):
            return None
        return xy if all(math.isfinite(value) for value in xy) else None
    return None


def setup_measured_head(node):
    """Real-render mode must report Gazebo head joints, not their moving targets."""
    import time
    import math
    from sensor_msgs.msg import JointState
    from rclpy.qos import qos_profile_sensor_data
    node.rendered_head = None
    def measured(msg):
        positions = dict(zip(msg.name, msg.position))
        names = ('pan_joint', 'tilt_joint')
        if all(name in positions and math.isfinite(positions[name]) for name in names):
            node.rendered_head = (time.monotonic(), {name: positions[name] for name in names})
    node.create_subscription(JointState, '/sim/gz/joint_states', measured, qos_profile_sensor_data,
                             callback_group=node.cb)


def apply_measured_head(node, full, head):
    """No fresh sample means absent measured joints, never synthetic zero feedback."""
    import time
    sample = node.rendered_head
    measured = sample[1] if sample is not None and time.monotonic() - sample[0] < 1.0 else {}
    pairs = [(n, measured.get(n, p)) for n, p in zip(full.name, full.position)
             if n not in ('pan_joint', 'tilt_joint') or n in measured]
    full.name = [n for n, _ in pairs]
    full.position = [float(p) for _, p in pairs]
    head.name = list(measured)
    head.position = [float(v) for v in measured.values()]
