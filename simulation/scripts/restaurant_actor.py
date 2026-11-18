#!/usr/bin/env python3
"""Restaurant participants answer requests and stage physical Gazebo items."""
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import Empty, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener
from sim_domain import validate_lab_domain

BAR_FRONT_XY = (-1.70, 0.0)


# A seated diner reaches a tray ~0.6 m beyond the chair edge; approach poses go up to 1.2 m.
CUSTOMER_REACH_M = 1.35


def main():
    try:
        validate_lab_domain(os.environ.get("ROS_DOMAIN_ID"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rclpy.init()
    node = rclpy.create_node(
        "restaurant_actor", parameter_overrides=[Parameter("use_sim_time", value=True)])
    pub = node.create_publisher(String, "/sim/speech", 10)
    from base_athome_sim import gz_model, real_gap
    from base_athome_sim.restaurant_actor_protocol import (
        CustomerPositionCache, DynamicCanFallTracker, ListenRequestGate, SingleFlight, actor_freshness_now,
        fresh_tf_receipt_age,
        barman_gripper_handoff_item, barman_gripper_handoff_ready,
        barman_gripper_confirmation_action,
        barman_counter_item_for_request, customer_order_key, find_staged_tray_order,
        transfer_staged_item_to_tray, record_gripper_handoff, counter_can_pose_world,
        can_pose_matches_tcp,
        customer_gripper_take_ready, tray_model_names,
    )
    from base_athome_sim.restaurant_humans import (
        customer_xy_for_state, ordered_items_for_state, pose_faces_target,
        barman_tray_items_for_request, reply_for_state, tray_items_for_state,
    )
    mission = {}
    lock = threading.Lock()
    arm_status = {"feedback_ok": False, "gripper_mm": 0.0}
    arm_status_at = 0.0
    tcp_seen_at = 0.0
    map_odom_seen_at = 0.0
    odom_base_seen_at = 0.0
    map_base_seen_at = 0.0
    listen_gate = ListenRequestGate()
    asr_noise = real_gap.asr_noise_enabled(os.environ)
    last_robot_say = {"text": ""}
    barman_silent_budget = real_gap.barman_silent_windows(os.environ)
    barman_silent_seen = {}
    noise_rng = random.Random(os.environ.get("ATHOME_SIM_REAL_GAP_SEED") or None)
    if asr_noise:
        node.get_logger().info("ATHOME_SIM_REAL_GAP=1: ASR ruidoso (Road yes, A Coke., silencios, eco)")
    customer_positions = CustomerPositionCache()
    can_fall_tracker = DynamicCanFallTracker()
    can_odom_subscriptions = {}
    can_odom_samples = {}
    stage_flight = SingleFlight()
    tray_transfer_flight = SingleFlight()
    sim_items_reset = False
    staged = {}
    delivered_gripper_cans = set()
    placed_barman_handoffs = set()
    taken_tray_models = set()
    tray_take_lock = threading.Lock()
    world = os.path.splitext(os.path.basename(os.environ.get("ATHOME_SIM_WORLD") or "restaurant.sdf"))[0]
    attach_pub = {
        f"restaurant_counter_can_{idx}": node.create_publisher(
            Empty, f"/sim/gz/restaurant_counter_can_{idx}/attach", 10)
        for idx in (1, 2)
    }
    detach_pub = {
        f"restaurant_counter_can_{idx}": node.create_publisher(
            Empty, f"/sim/gz/restaurant_counter_can_{idx}/detach", 10)
        for idx in (1, 2)
    }
    tf_buffer = Buffer(node=node)
    tf_listener = TransformListener(tf_buffer, node, spin_thread=False)
    node.get_logger().info(
        f"Actor pose clock use_sim_time={node.get_parameter('use_sim_time').value} "
        f"clock_type={node.get_clock().clock_type}")
    pose_lookup_diagnostic_at = 0.0

    def on_state(msg):
        nonlocal sim_items_reset
        try:
            snapshot = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if snapshot.get("state") == "STARTING" and not sim_items_reset:
            sim_items_reset = True
            threading.Thread(target=reset_stale_order_models, daemon=True).start()
        snapshot = customer_positions.observe(snapshot)
        with lock:
            mission.clear()
            mission.update(snapshot)

    def on_arm_status(msg):
        nonlocal arm_status_at
        try:
            value = json.loads(msg.data)
            width = float(value["gripper_mm"])
            valid = value.get("feedback_ok") is True and math.isfinite(width)
        except (TypeError, ValueError, KeyError, AttributeError):
            valid, value = False, {}
        with lock:
            arm_status.clear()
            arm_status.update(value if valid else {"feedback_ok": False, "gripper_mm": 0.0})
            arm_status_at = actor_freshness_now(node) if valid else 0.0

    def on_tf(msg):
        nonlocal tcp_seen_at, map_odom_seen_at, odom_base_seen_at, map_base_seen_at
        now = actor_freshness_now(node)
        has_tcp = any(transform.child_frame_id in {"tcp_link", "link6"} for transform in msg.transforms)
        if msg.transforms:
            with lock:
                for transform in msg.transforms:
                    parent = transform.header.frame_id.lstrip("/")
                    child = transform.child_frame_id.lstrip("/")
                    if (parent, child) == ("map", "odom"):
                        map_odom_seen_at = now
                    elif (parent, child) == ("odom", "base_footprint"):
                        odom_base_seen_at = now
                    elif (parent, child) == ("map", "base_footprint"):
                        map_base_seen_at = now
                if has_tcp:
                    tcp_seen_at = now

    def fresh_pose():
        nonlocal pose_lookup_diagnostic_at
        now = actor_freshness_now(node)
        with lock:
            map_odom_at, odom_base_at, map_base_at = (
                map_odom_seen_at, odom_base_seen_at, map_base_seen_at)
        map_odom_age = fresh_tf_receipt_age(now, map_odom_at)
        odom_base_age = fresh_tf_receipt_age(now, odom_base_at)
        map_base_age = fresh_tf_receipt_age(now, map_base_at)
        chain_ages = (map_odom_age, odom_base_age)
        receipt_age = (map_base_age if map_base_age is not None
                       else max(chain_ages) if all(age is not None for age in chain_ages) else None)
        if receipt_age is None:
            wall_now = time.monotonic()
            if wall_now - pose_lookup_diagnostic_at >= 5.0:
                pose_lookup_diagnostic_at = wall_now
                node.get_logger().warning(
                    f"Actor map pose TF receipt stale: use_sim_time="
                    f"{node.get_parameter('use_sim_time').value} now_s={now:.3f} "
                    f"map_odom_age_s={map_odom_age} odom_base_age_s={odom_base_age} "
                    f"map_base_age_s={map_base_age} "
                    f"frames={tf_buffer.all_frames_as_yaml()}")
            return None
        lookup_time = Time(clock_type=node.get_clock().clock_type)
        try:
            transform = tf_buffer.lookup_transform(
                "map", "base_footprint", lookup_time, timeout=Duration(seconds=0.1))
        except Exception as exc:
            wall_now = time.monotonic()
            if wall_now - pose_lookup_diagnostic_at >= 5.0:
                pose_lookup_diagnostic_at = wall_now
                node.get_logger().warning(
                    f"Actor map pose TF unavailable: use_sim_time="
                    f"{node.get_parameter('use_sim_time').value} now_s={now:.3f} "
                    f"clock_type={node.get_clock().clock_type} error={exc}; "
                    f"frames={tf_buffer.all_frames_as_yaml()}")
            return None
        stamp = transform.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        p, q = transform.transform.translation, transform.transform.rotation
        norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if (not all(math.isfinite(v) for v in (p.x, p.y, q.x, q.y, q.z, q.w))
                or abs(norm - 1.0) > 0.01):
            wall_now = time.monotonic()
            if wall_now - pose_lookup_diagnostic_at >= 5.0:
                pose_lookup_diagnostic_at = wall_now
                node.get_logger().warning(
                    f"Actor map pose TF stale/invalid: use_sim_time="
                    f"{node.get_parameter('use_sim_time').value} now_s={now:.3f} "
                    f"stamp_s={stamp_s:.3f} receive_age_s={receipt_age:.3f} "
                    f"frame={transform.header.frame_id} child={transform.child_frame_id}")
            return None
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        return p.x, p.y, yaw

    def gz_command(command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            node.get_logger().warning(f"Gazebo item service failed: {exc}")
            return False
        return "true" in result.stdout.lower()

    def reset_stale_order_models():
        """A campaign starts with an empty bar even if Gazebo kept the prior run's world."""
        for idx in (1, 2):
            for name in (f"restaurant_counter_can_{idx}",
                         f"restaurant_tray_item_{idx}",
                         f"restaurant_counter_can_{idx}_on_tray"):
                gz_command(gz_model.remove_command(world, name))
        with lock:
            staged.clear()

    def watch_dynamic_can(name):
        if name in can_odom_subscriptions:
            return

        def on_can_odom(msg):
            p = msg.pose.pose.position
            with lock:
                can_odom_samples[name] = (time.monotonic(), (p.x, p.y, p.z))
            if can_fall_tracker.observe(name, p.x, p.y, p.z):
                node.get_logger().warning(
                    f"DYNAMIC_CAN_FELL model={name} z={p.z:.3f} xy=({p.x:.3f},{p.y:.3f})")

        can_odom_subscriptions[name] = node.create_subscription(
            Odometry, gz_model.restaurant_item_odometry_topic(name), on_can_odom, 10)

    def wait_for_can_at_tcp(name, tcp_xyz, not_before, timeout_s=1.5):
        """Attach only after fresh Gazebo odometry confirms the can was moved to the TCP."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with lock:
                sample = can_odom_samples.get(name)
            now = time.monotonic()
            if (sample is not None and sample[0] >= not_before and now - sample[0] <= 0.5
                    and can_pose_matches_tcp(sample[1], tcp_xyz)):
                return True
            time.sleep(0.05)
        return False

    def item_names(snapshot):
        idx = max(int(snapshot.get("customer_idx", 0)), int(snapshot.get("served", 0))) + 1
        return f"restaurant_counter_can_{idx}", f"restaurant_tray_item_{idx}"

    def counter_pose(pose):
        try:
            arm_mount = tf_buffer.lookup_transform(
                "base_link", "arm_base_link", Time(), timeout=Duration(seconds=0.2))
        except Exception as exc:
            node.get_logger().warning(f"Barman placement refused: arm mount TF unavailable: {exc}")
            return None
        result = counter_can_pose_world(pose, arm_mount.transform.translation.y)
        if result is None:
            node.get_logger().warning("Barman placement refused: invalid robot pose or arm mount offset")
        return result

    def tray_pose(pose, index=0, count=1):
        x, y, yaw = pose
        lateral = (index - (count - 1) / 2) * 0.12
        tray_x = (x + gz_model.RESTAURANT_TRAY_CENTER_X_M * math.cos(yaw)
                  - lateral * math.sin(yaw))
        tray_y = (y + gz_model.RESTAURANT_TRAY_CENTER_X_M * math.sin(yaw)
                  + lateral * math.cos(yaw))
        tray_top = gz_model.RESTAURANT_TRAY_TOP_Z_M
        item_height = 0.10
        return tray_x, tray_y, tray_top + item_height / 2 + 0.002

    def stage_items(snapshot, *, all_on_tray=False, items=None):
        items = tuple(items) if items is not None else ordered_items_for_state(snapshot)
        if len(items) != 2:
            node.get_logger().warning("Barman placement refused: expected two items")
            return False
        order_key = customer_order_key(snapshot)
        if order_key is None:
            node.get_logger().warning("Barman placement refused: invalid customer order key")
            return False
        key = (*order_key, items, all_on_tray)

        def create_items():
            pose = fresh_pose()
            if pose is None or not pose_faces_target(pose, BAR_FRONT_XY):
                node.get_logger().warning("Barman placement refused: missing fresh odometry or front-facing pose")
                return False
            # This runs only after a listen request in ORDERING_BARMAN.
            # A human barman takes a few seconds; the plain sim does not need the full 3-5 s.
            time.sleep(random.uniform(3.0, 5.0) if real_gap.real_gap_enabled(os.environ) else random.uniform(1.5, 3.0))
            pose = fresh_pose()
            if pose is None or not pose_faces_target(pose, BAR_FRONT_XY):
                return False
            can_name, tray_name = item_names(snapshot)
            for name in (can_name, tray_name):
                watch_dynamic_can(name)
            if all_on_tray:
                names = (can_name, tray_name)
                for index, name in enumerate(names):
                    if not gz_command(gz_model.create_command(world, gz_model.restaurant_tray_item_sdf(
                            tray_pose(pose, index, len(names)), name=name))):
                        for created in names[:index]:
                            gz_command(gz_model.remove_command(world, created))
                        return False
            else:
                can_pose = counter_pose(pose)
                if can_pose is None:
                    return False
                can_ok = gz_command(gz_model.create_command(world, gz_model.restaurant_can_sdf(
                    can_pose, name=can_name)))
                if not can_ok:
                    return False
                tray_ok = gz_command(gz_model.create_command(world, gz_model.restaurant_tray_item_sdf(
                    tray_pose(pose), name=tray_name)))
                if not tray_ok:
                    gz_command(gz_model.remove_command(world, can_name))
                    return False
                names = (tray_name,)
            with lock:
                staged[key] = {"can": can_name, "tray": tray_name, "items": items,
                               "tray_items": items if all_on_tray else items[1:],
                               "tray_names": names}
            location = "tray" if all_on_tray else "counter and tray"
            node.get_logger().info(f"Barman placed requested items {items} on {location}")
            return True

        return stage_flight.run(key, create_items)

    def put_item_on_tray_directly(snapshot, request_key, item):
        def place():
            # Bounded retry (4 x 0.5 s) for the robot to face the bar; then the barman puts the item on the
            # tray anyway if there is any fresh pose, so this never turns into an endless "robot no".
            pose = fresh_pose()
            for _ in range(4):
                if pose is not None and pose_faces_target(pose, BAR_FRONT_XY):
                    break
                time.sleep(0.5)
                pose = fresh_pose()
            if pose is None:
                node.get_logger().warning("Barman direct tray placement refused: no fresh pose")
                return False
            can_name, _ = item_names(snapshot)
            tray_name = f"{can_name}_on_tray"
            watch_dynamic_can(tray_name)
            if not gz_command(gz_model.create_command(
                    world, gz_model.restaurant_tray_item_sdf(tray_pose(pose), name=tray_name))):
                return False
            with lock:
                staged[(*request_key, (item,), True)] = {
                    "can": can_name, "tray": tray_name, "items": (item,),
                    "tray_items": (item,), "tray_names": (tray_name,)}
            node.get_logger().info(f"Barman put {item} on the tray (nothing was staged at the counter)")
            return True
        return tray_transfer_flight.run((request_key, item), place)

    def move_counter_item_to_tray(snapshot, item):
        item = str(item).strip()
        request_key = customer_order_key(snapshot)
        if request_key is None:
            return False
        with lock:
            staged_entry = next(((key, dict(record)) for key, record in staged.items()
                                 if key[:1] == request_key and item in record.get("items", ())), None)
        if staged_entry is None:
            # Only reached when the item was never prepared (staged_entry is None), a case that used to
            # answer "robot no" forever, so the default sim only changes in that dead-end. Nothing was staged (visual pick off / dock skipped): a person at the bar would
            # simply put the item on the tray. Answering "robot no" here looped forever (17/11 logs: 95x).
            return put_item_on_tray_directly(snapshot, request_key, item)
        stage_key, record = staged_entry
        tray_name = f"{record['can']}_on_tray"
        flight_key = (request_key, item)

        def transfer():
            pose = fresh_pose()
            if pose is None or not pose_faces_target(pose, BAR_FRONT_XY):
                return False
            tray_items = tuple(record.get("tray_items", ()))
            tray_position = tray_pose(pose, len(tray_items), len(tray_items) + 1)
            updated = transfer_staged_item_to_tray(
                record, item, tray_name,
                create_item=lambda: gz_command(gz_model.create_command(world,
                    gz_model.restaurant_tray_item_sdf(tray_position, name=tray_name))),
                remove_counter_item=lambda: gz_command(gz_model.remove_command(world, record["can"])),
                remove_tray_item=lambda: gz_command(gz_model.remove_command(world, tray_name)),
            )
            if updated is None:
                return False
            with lock:
                staged[stage_key] = updated
            node.get_logger().info(f"Barman moved {item} from the counter to the tray")
            return True

        return tray_transfer_flight.run(flight_key, transfer)

    def place_counter_item_in_gripper(snapshot, item):
        """Simulate the barman placing the staged can into an open, bar-facing gripper."""
        request_key = customer_order_key(snapshot)
        pose = fresh_pose()
        with lock:
            status = dict(arm_status)
            arm_age = actor_freshness_now(node) - arm_status_at if arm_status_at else float("inf")
            tcp_age = actor_freshness_now(node) - tcp_seen_at if tcp_seen_at else float("inf")
            record_entry = next(((key, dict(record)) for key, record in staged.items()
                                 if request_key is not None and key[:1] == request_key
                                 and item in record.get("items", ())), None)
        facing = pose_faces_target(pose, BAR_FRONT_XY)
        ready = barman_gripper_handoff_ready(snapshot, status, arm_age, facing)
        tcp_fresh = math.isfinite(tcp_age) and 0.0 <= tcp_age < 0.5
        handoff_item = barman_gripper_handoff_item(
            snapshot, record_entry[1] if record_entry is not None else None)
        item_on_tray = record_entry is not None and item in record_entry[1].get("tray_items", ())
        if handoff_item is None or not ready or not tcp_fresh or pose is None:
            node.get_logger().warning(
                "Barman gripper handoff refused: "
                f"staged={record_entry is not None} item_on_tray={item_on_tray} "
                f"feedback_ok={status.get('feedback_ok')} gripper_mm={status.get('gripper_mm')} "
                f"arm_age_s={arm_age:.3f} tcp_age_s={tcp_age:.3f} "
                f"facing_bar={facing} pose_available={pose is not None}")
            return False
        try:
            # Gazebo entity poses use the odom/world origin; map->odom can drift under SLAM.
            transform = tf_buffer.lookup_transform("odom", "tcp_link", Time(), timeout=Duration(seconds=0.2))
        except Exception as exc:
            node.get_logger().warning(f"Barman gripper handoff refused: TCP transform unavailable: {exc}")
            return False
        xyz = transform.transform.translation
        rotation = transform.transform.rotation
        tcp_xyz = (xyz.x, xyz.y, xyz.z)
        yaw = math.atan2(2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
                         1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z))
        if record_entry is not None:
            can_name = record_entry[1]["can"]
            placed = gz_command(gz_model.set_pose_command(world, can_name, (xyz.x, xyz.y, xyz.z), yaw=yaw))
        else:
            # A dock failure can skip normal bar staging. The explicit barman
            # handoff may create the requested can at the measured TCP directly.
            try:
                slot = max(int(snapshot.get("customer_idx", 0)), int(snapshot.get("served", 0))) + 1
            except (TypeError, ValueError, OverflowError):
                slot = 1
            slot = max(1, slot)
            can_name = f"restaurant_counter_can_{slot}"
            watch_dynamic_can(can_name)
            command = gz_model.set_pose_command(world, can_name, tcp_xyz, yaw=yaw)
            placed = gz_command(command)
            if not placed:
                placed = gz_command(gz_model.create_command(
                    world, gz_model.restaurant_can_sdf(tcp_xyz, name=can_name)))
        if not placed:
            node.get_logger().warning(
                f"Barman gripper handoff refused: cannot place/create {can_name} at measured TCP")
            return False
        if can_name not in attach_pub:
            node.get_logger().warning(
                f"Barman gripper handoff refused: no measured-attach publisher for {can_name}")
            return False
        placed_at = time.monotonic()
        if not wait_for_can_at_tcp(can_name, tcp_xyz, placed_at):
            node.get_logger().warning(
                f"Barman gripper handoff refused: {can_name} odometry did not confirm TCP placement")
            return False
        attach_pub[can_name].publish(Empty())
        if record_entry is None:
            with lock:
                if not record_gripper_handoff(staged, request_key, item, can_name):
                    node.get_logger().warning(
                        f"Barman gripper handoff: could not record staged can {can_name} for tray fallback")
                    return False
        node.get_logger().info(
            f"Barman placed {item} in the open gripper via detachable joint; staged={record_entry is not None}")
        return True

    def take_tray_item(snapshot, requested_items=None):
        items = tuple(requested_items) if requested_items is not None else tray_items_for_state(snapshot)
        pose, customer_xy = fresh_pose(), customer_xy_for_state(snapshot)
        if not items or pose is None or customer_xy is None:
            node.get_logger().warning(f"Customer tray pickup refused: items={items} pose={pose} customer_xy={customer_xy}")
            return False
        distance = math.dist(pose[:2], customer_xy)
        if distance >= CUSTOMER_REACH_M:
            node.get_logger().warning(f"Customer tray pickup refused: robot {distance:.2f} m from {customer_xy} (> {CUSTOMER_REACH_M} m)")
            return False
        with lock:
            record = find_staged_tray_order(staged, snapshot, items)
        if record is not None:
            names = tuple((item, name) for item, name in zip(record["tray_items"], record["tray_names"])
                          if item in items)
        else:
            slot = max(int(snapshot.get("customer_idx", 0)), int(snapshot.get("served", 0))) + 1
            orders = (("coke", "water"), ("juice", "tea"))
            order = orders[min(max(slot - 1, 0), len(orders) - 1)]
            names = tray_model_names(items, slot, order)
        if not names:
            node.get_logger().warning(f"Customer tray pickup refused: no matching tray models for items={items}")
            return False
        all_removed = True
        for item, name in names:
            with tray_take_lock:
                if name in taken_tray_models:
                    continue
                removed = gz_command(gz_model.remove_command(world, name))
                if removed:
                    taken_tray_models.add(name)
            node.get_logger().info(
                f"Customer tray pickup {'removed' if removed else 'could not remove'} item={item} model={name} "
                f"distance_m={distance:.2f} staged={record is not None}")
            if not removed:
                all_removed = False
        return all_removed

    def take_gripper_item_on_countdown(msg):
        if "one. two. three." not in msg.data.lower():
            return
        deadline = time.monotonic() + 4.0
        last_reason = "no matching can or fresh pose"
        while time.monotonic() < deadline:
            with lock:
                snapshot = dict(mission)
                key = customer_order_key(snapshot)
                record = next((dict(value) for staged_key, value in staged.items()
                               if key is not None and staged_key[:1] == key
                               and value.get("can") and value.get("can") not in delivered_gripper_cans), None)
                customer_xy = customer_xy_for_state(snapshot)
            can_name = record["can"] if record is not None else ""
            if not can_name and key is not None:
                slot = max(int(snapshot.get("customer_idx", 0)), int(snapshot.get("served", 0))) + 1
                candidate = f"restaurant_counter_can_{slot}"
                if candidate not in delivered_gripper_cans:
                    can_name = candidate
            pose = fresh_pose()
            if customer_gripper_take_ready(snapshot, pose, customer_xy, can_name, msg.data):
                detach_publisher = detach_pub.get(can_name)
                if detach_publisher is None:
                    last_reason = f"no detach publisher for {can_name}"
                else:
                    # Notify the simulated gripper first, then remove the visual can.
                    detach_publisher.publish(Empty())
                    if gz_command(gz_model.remove_command(world, can_name)):
                        with lock:
                            delivered_gripper_cans.add(can_name)
                        node.get_logger().info(
                            f"CUSTOMER_GRIPPER_TAKE model={can_name} customer_xy={customer_xy} "
                            "within_reach=true feedback_independent=true")
                        return
                    last_reason = f"Gazebo could not remove {can_name}"
            else:
                last_reason = (f"state={snapshot.get('state')} pose={pose} customer_xy={customer_xy} "
                               f"can={can_name or 'unknown'}")
            time.sleep(0.2)
        node.get_logger().warning(f"Customer gripper pickup not taken after 4 s: {last_reason}")

    def on_sim_speech(msg):
        speech = msg.data.lower()
        last_robot_say["text"] = msg.data
        if "one. two. three." in speech:
            threading.Thread(target=take_gripper_item_on_countdown, args=(msg,), daemon=True).start()
        elif "please take the" in speech and "from my tray" in speech:
            with lock:
                snapshot = dict(mission)
            items = tray_items_for_state(snapshot)
            threading.Thread(target=take_tray_item, args=(snapshot, items), daemon=True).start()

    def answer_request(request, snapshot, request_id):
        if request.get("prompt") == "Stop.":
            return
        info = snapshot.get("info") or {}
        silence_ok = False  # silencio/eco solo en respuestas de texto sin efecto fisico previo
        if (snapshot.get("state") == "WAITING_BARMAN"
                and info.get("placement") == "gripper"):
            with lock:
                action, handoff_key = barman_gripper_confirmation_action(
                    snapshot, placed_barman_handoffs)
            if action == "place":
                placed = place_counter_item_in_gripper(snapshot, str(info.get("item", "")).strip())
                if placed:
                    with lock:
                        placed_barman_handoffs.add(handoff_key)
                answer = "robot yes" if placed else "robot no"
            else:
                answer = "robot yes" if action == "confirm" else "robot no"
                if action == "confirm" and barman_silent_seen.get(handoff_key, 0) < barman_silent_budget:
                    barman_silent_seen[handoff_key] = barman_silent_seen.get(handoff_key, 0) + 1
                    print(f"PERSONA: <barman callado, ventana {barman_silent_seen[handoff_key]}/"
                          f"{barman_silent_budget}>", flush=True)
                    return
        else:
            counter_item = barman_counter_item_for_request(snapshot, request)
            tray_order = barman_tray_items_for_request(snapshot, request)
            if counter_item:
                answer = "robot yes" if move_counter_item_to_tray(snapshot, counter_item) else "robot no"
            elif tray_order:
                answer = "robot yes" if stage_items(snapshot, all_on_tray=True, items=tray_order) else "robot no"
            elif ordered_items_for_state(snapshot):
                answer = "robot yes" if stage_items(snapshot) else "robot no"
            elif tray_items_for_state(snapshot):
                answer = "robot yes" if take_tray_item(snapshot) else "robot no"
            else:
                answer = reply_for_state(snapshot)
                silence_ok = True
        if not listen_gate.wait_until_listening(request_id):
            node.get_logger().warning(
                f"Actor reply suppressed: listen request {request_id} never opened or its start window expired")
            return
        if asr_noise:
            clean = answer
            answer = real_gap.noisy_reply(
                clean, noise_rng, allow_silence=silence_ok, last_robot_say=last_robot_say["text"])
            if answer is None:
                print(f"PERSONA: <silencio> (era: {clean}; {snapshot.get('state', 'unknown')})", flush=True)
                return
            if answer and answer == last_robot_say["text"]:
                time.sleep(real_gap.echo_delay_s(noise_rng))  # el eco del TTS llega con retardo
        print(f"PERSONA: {answer} ({snapshot.get('state', 'unknown')})", flush=True)
        pub.publish(String(data=answer))

    def on_request(msg):
        try:
            request = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        request_id = listen_gate.register_request(request)
        if not request_id:
            node.get_logger().warning("Actor reply suppressed: listen request has no id")
            return
        with lock:
            snapshot = dict(mission)
        threading.Thread(target=answer_request, args=(request, snapshot, request_id), daemon=True).start()

    def on_voice_event(msg):
        try:
            event = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        listen_gate.mark_listening(event)

    node.create_subscription(String, "/restaurant/state", on_state, 10)
    node.create_subscription(String, "/voice/listen_request", on_request, 10)
    node.create_subscription(String, "/voice/result", on_voice_event, 10)
    # Robot utterances come from Restaurant's TTS request topic; /sim/speech is
    # reserved for speech delivered by simulated humans to the robot.
    node.create_subscription(String, "/hri/say_something", on_sim_speech, 10)
    # BAR_FRONT_XY is in map. Gazebo odometry is in odom and can drift from map;
    # use the current map->base transform that Nav2 uses for its goals.
    node.create_subscription(String, "/arm/status", on_arm_status, 10)
    node.create_subscription(TFMessage, "/tf", on_tf, 20)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
