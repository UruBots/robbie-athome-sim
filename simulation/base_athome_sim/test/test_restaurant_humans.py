"""Rendered Restaurant participants and honest scripted dialogue share one layout."""
from pathlib import Path
import importlib.util
import json
import math
import threading
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from base_athome_sim.restaurant_humans import (
    CUSTOMERS, barman_tray_items_for_request, customer_xy_for_state, ordered_items_for_state, pose_faces_target,
    reply_for_state, tray_items_for_state,
)

ROOT = Path(__file__).parents[1]


def test_world_matches_default_actor_coordinates_and_meshes():
    world = ET.parse(ROOT / 'worlds/restaurant.sdf').getroot()
    humans = [m for m in world.findall('./world/model') if m.get('name', '').startswith('human_')]
    assert len(humans) == len(CUSTOMERS) == 3
    for model, expected in zip(humans, CUSTOMERS):
        assert [float(v) for v in model.findtext('pose').split()[:2]] == list(expected[:2])
        assert model.findtext('static') == 'true'
        assert not model.findall('.//collision')  # no new navigation obstacles
        uri = model.findtext('.//mesh/uri')
        assert (ROOT / 'models/humans' / uri.removeprefix('model://')).is_file()
    assert 'female_visitor' in humans[1].findtext('.//mesh/uri')
    assert 'nurse_female' in humans[2].findtext('.//mesh/uri')


def test_initial_world_meshes_match_dialogue_gesture_state():
    from base_athome_sim.restaurant_humans import ParticipantGestures, LOWERED_RENDERED_MODELS, RAISED_RENDERED_MODELS
    world = ET.parse(ROOT / 'worlds/restaurant.sdf').getroot()
    humans = [m for m in world.findall('./world/model') if m.get('name', '').startswith('human_')]
    gestures = ParticipantGestures()
    assert gestures.raised == frozenset({2})
    for index, model in enumerate(humans):
        model_name = RAISED_RENDERED_MODELS[index] if index in gestures.raised else LOWERED_RENDERED_MODELS[index]
        assert model.findtext('.//mesh/uri').endswith(f'/{model_name}.dae')


def test_rendered_diners_raise_their_real_gesture_mesh_during_customer_scans():
    from base_athome_sim.restaurant_humans import ParticipantGestures, gesture_model_sdf

    gestures = ParticipantGestures()
    for served in (0, 1):
        raised = gestures.observe({'state': 'DETECTING_CUSTOMERS', 'served': served})
        assert raised == frozenset({0, 1})
        for index in raised:
            mesh_name = 'seated_right.dae' if index == 0 else 'seated_left.dae'
            assert mesh_name in gesture_model_sdf(index, True)


def test_sim_barman_gripper_handoff_requires_open_fresh_gripper_at_bar():
    from base_athome_sim.restaurant_actor_protocol import (
        barman_gripper_handoff_item, barman_gripper_handoff_ready,
    )

    state = {"state": "WAITING_BARMAN", "info": {"item": "coke", "placement": "gripper"}}
    status = {"feedback_ok": True, "gripper_mm": 105.0}
    assert barman_gripper_handoff_ready(state, status, 0.2, True)
    assert not barman_gripper_handoff_ready(state, status, 1.0, True)
    assert not barman_gripper_handoff_ready(state, status, 0.2, False)
    assert not barman_gripper_handoff_ready(state, {"feedback_ok": True, "gripper_mm": 2.0}, 0.2, True)
    assert not barman_gripper_handoff_ready(
        {"state": "WAITING_BARMAN", "info": {"item": "coke"}}, status, 0.2, True)
    assert barman_gripper_handoff_item(state) == "coke"  # Plan B may skip bar staging.
    assert barman_gripper_handoff_item(state, {"tray_items": ("water",)}) == "coke"
    assert barman_gripper_handoff_item(state, {"tray_items": ("coke",)}) is None
    assert barman_gripper_handoff_item({"state": "ORDERING_BARMAN", "info": state["info"]}) is None


def test_barman_gripper_double_confirmation_places_once_and_requires_cached_success():
    from base_athome_sim.restaurant_actor_protocol import barman_gripper_confirmation_action

    first = {"state": "WAITING_BARMAN", "served": 0,
             "info": {"item": "coke", "placement": "gripper", "confirmation_step": 1}}
    second = {"state": "WAITING_BARMAN", "served": 0,
              "info": {"item": "coke", "placement": "gripper", "confirmation_step": 2}}
    placements = set()
    action, key = barman_gripper_confirmation_action(first, placements)
    assert action == "place"
    assert barman_gripper_confirmation_action(second, placements) == ("reject", key)
    placements.add(key)
    assert barman_gripper_confirmation_action(second, placements) == ("confirm", key)


def test_direct_gripper_handoff_is_recorded_for_later_tray_fallback():
    from base_athome_sim.restaurant_actor_protocol import (
        record_gripper_handoff, transfer_staged_item_to_tray,
    )
    staged = {}
    key = (0,)
    assert record_gripper_handoff(staged, key, "coke", "restaurant_counter_can_1")
    assert staged[key] == {
        "can": "restaurant_counter_can_1", "tray": "restaurant_counter_can_1_on_tray",
        "items": ("coke",), "tray_items": (), "tray_names": (),
    }
    record = staged[key]
    assert transfer_staged_item_to_tray(
        record, "coke", "restaurant_counter_can_1_on_tray",
        create_item=lambda: True, remove_counter_item=lambda: True,
        remove_tray_item=lambda: True,
    )["tray_items"] == ("coke",)


def test_barman_handoff_can_must_be_measured_at_the_tcp_before_attach():
    from base_athome_sim.restaurant_actor_protocol import can_pose_matches_tcp

    assert can_pose_matches_tcp((1.0, 2.0, 1.0), (1.03, 2.02, 1.04))
    assert not can_pose_matches_tcp((1.205, 2.0, 1.0), (1.0, 2.0, 1.0))
    assert not can_pose_matches_tcp((float('nan'), 0.0, 0.0), (0.0, 0.0, 0.0))


@pytest.mark.parametrize("yaw", [math.pi, math.radians(170)])
def test_counter_can_pose_uses_measured_arm_lateral_offset_and_robot_yaw(yaw):
    from base_athome_sim.restaurant_actor_protocol import counter_can_pose_world

    pose = (-1.28, -0.13, yaw)
    arm_y = 0.175
    result = counter_can_pose_world(pose, arm_y)
    assert result is not None
    assert result[0] == pytest.approx(-1.80)
    c, s = math.cos(yaw), math.sin(yaw)
    base_x = (result[0] - pose[0]) * c + (result[1] - pose[1]) * s
    base_y = -(result[0] - pose[0]) * s + (result[1] - pose[1]) * c
    assert base_y == pytest.approx(arm_y - 0.04)
    centered = counter_can_pose_world(pose, arm_y, robot_right_offset_m=0.0)
    assert result[1] > centered[1]  # world +Y is the robot's right at the bar
    assert math.isfinite(base_x)
    assert result[2] == pytest.approx(1.015)


def test_diner_does_not_impersonate_barman():
    for xy in ([2., 2.8], [3., -2.9], None):
        assert reply_for_state(dict(state='CONFIRMING_BARMAN', info=dict(xy=xy))) == 'robot no'
    assert reply_for_state(dict(state='CONFIRMING_BARMAN', info=dict(xy=[-2.55, .1]))) == 'robot yes'
    assert reply_for_state(dict(state='TAKING_ORDER', customer_idx=0)) == 'my order is coke and water'
    assert reply_for_state(dict(state='TAKING_ORDER', customer_idx=1)) == 'my order is juice and tea'
    assert reply_for_state(dict(state='TAKING_ORDER', customer_idx=0, served=1)) == 'my order is juice and tea'


def test_barman_items_are_only_available_after_post_dock_order_request_state():
    request = dict(state='ORDERING_BARMAN', customer_idx=1, served=1,
                   info=dict(items=['juice', 'tea'], use_tray=True))
    assert ordered_items_for_state(request) == ('juice', 'tea')
    assert ordered_items_for_state({**request, 'state': 'RETURNING_TO_BAR'}) == ()
    assert ordered_items_for_state({**request, 'info': {**request['info'], 'use_tray': False}}) == ()


def test_second_delivery_finds_staged_tray_order_after_mission_customer_index_restarts():
    from base_athome_sim.restaurant_actor_protocol import customer_order_key, find_staged_tray_order

    # First diner was customer_idx=0, served=0. The next search round restarts
    # customer_idx at zero even though the mission is now serving diner two.
    staged = {
        (1, 1, ('juice', 'tea'), False): {
            'items': ('juice', 'tea'), 'tray_items': ('tea', 'juice'),
            'tray_names': ('tea_on_tray', 'juice_on_tray'),
        }
    }
    snapshot = {'state': 'WAITING_CUSTOMER', 'customer_idx': 0, 'served': 1}
    assert customer_order_key(snapshot) == (1,)
    assert find_staged_tray_order(staged, snapshot, ('juice', 'tea')) == next(iter(staged.values()))


def test_staged_order_key_uses_global_delivery_slot_when_local_index_changes():
    from base_athome_sim.restaurant_actor_protocol import customer_order_key, find_staged_tray_order

    # The barman can observe local customer 1 before the mission's global
    # served count advances. WAITING_CUSTOMER may then use local index 0.
    order = {'items': ('coke', 'water'), 'tray_items': ('water', 'coke'),
             'tray_names': ('water_on_tray', 'coke_on_tray')}
    staged_snapshot = {'state': 'ORDERING_BARMAN', 'customer_idx': 1, 'served': 0}
    pickup_snapshot = {'state': 'WAITING_CUSTOMER', 'customer_idx': 0, 'served': 0}
    staged = {(*customer_order_key(staged_snapshot), ('coke', 'water'), False): order}

    assert customer_order_key(staged_snapshot) == customer_order_key(pickup_snapshot) == (0,)
    assert find_staged_tray_order(staged, pickup_snapshot, ('coke', 'water')) is order


def test_staged_order_lookup_survives_search_round_index_reset_after_first_delivery():
    from base_athome_sim.restaurant_actor_protocol import customer_order_key, find_staged_tray_order

    first_order = {'items': ('coke', 'water'), 'tray_items': ('water', 'coke'),
                   'tray_names': ('water_0_on_tray', 'coke_0_on_tray')}
    second_order = {'items': ('juice', 'tea'), 'tray_items': ('tea', 'juice'),
                    'tray_names': ('tea_1_on_tray', 'juice_1_on_tray')}
    first_customer = {'state': 'ORDERING_BARMAN', 'customer_idx': 0, 'served': 0}
    staged_snapshot = {'state': 'ORDERING_BARMAN', 'customer_idx': 1, 'served': 1}
    pickup_snapshot = {'state': 'WAITING_CUSTOMER', 'customer_idx': 0, 'served': 1}
    staged = {
        (*customer_order_key(first_customer), ('coke', 'water'), False): first_order,
        (*customer_order_key(staged_snapshot), ('juice', 'tea'), False): second_order,
    }

    assert customer_order_key(staged_snapshot) == customer_order_key(pickup_snapshot) == (1,)
    assert find_staged_tray_order(staged, pickup_snapshot, ('juice', 'tea')) is second_order


def test_barman_stages_both_items_on_tray_only_for_full_tray_fallback_request():
    state = dict(state='WAITING_BARMAN', info=dict(items=['juice', 'tea']))
    request = {'prompt': 'Please place the juice and tea on my tray.'}
    assert barman_tray_items_for_request(state, request) == ('juice', 'tea')
    assert barman_tray_items_for_request(state, {'prompt': 'Please place the juice on the counter.'}) == ()
    assert barman_tray_items_for_request({**state, 'state': 'ORDERING_BARMAN'}, request) == ()
    assert barman_tray_items_for_request({**state, 'info': {'items': ['juice']}}, request) == ()


def test_actor_reply_waits_for_matching_open_listen_and_expires_with_speech_start_window():
    from base_athome_sim.restaurant_actor_protocol import ListenRequestGate

    class Clock:
        now = 10.0

        def __call__(self):
            return self.now

    clock = Clock()
    gate = ListenRequestGate(clock=clock)
    assert gate.register_request({'id': '4662-13', 'start_timeout': 6.0}) == '4662-13'
    assert not gate.response_ready('4662-13')
    assert gate.mark_listening({'id': '4662-12', 'event': 'listening'})
    assert not gate.response_ready('4662-13')
    assert gate.mark_listening({'id': '4662-13', 'event': 'listening'})
    assert gate.response_ready('4662-13')
    clock.now += 5.9
    assert gate.response_ready('4662-13')
    clock.now += 0.2
    assert not gate.response_ready('4662-13')


def test_actor_gate_handles_listening_event_arriving_before_request_callback():
    from base_athome_sim.restaurant_actor_protocol import ListenRequestGate

    gate = ListenRequestGate()
    assert gate.mark_listening({'id': '4662-14', 'event': 'listening'})
    assert gate.register_request({'id': '4662-14', 'start_timeout': 6.0}) == '4662-14'
    assert gate.response_ready('4662-14')


def test_actor_waits_until_matching_open_event_before_releasing_reply():
    from base_athome_sim.restaurant_actor_protocol import ListenRequestGate

    gate = ListenRequestGate()
    gate.register_request({'id': '4662-16', 'start_timeout': 6.0})
    reply_allowed = []
    waiter = threading.Thread(target=lambda: reply_allowed.append(
        gate.wait_until_listening('4662-16', open_timeout=1.0)))
    waiter.start()
    assert waiter.is_alive()
    gate.mark_listening({'id': '4662-16', 'event': 'listening'})
    waiter.join(timeout=0.5)
    assert not waiter.is_alive() and reply_allowed == [True]


def test_actor_suppresses_requests_without_ids_and_non_listening_events():
    from base_athome_sim.restaurant_actor_protocol import ListenRequestGate

    gate = ListenRequestGate()
    assert gate.register_request({'prompt': 'Are you the barman?'}) == ''
    assert not gate.mark_listening({'id': '4662-15', 'event': 'result'})
    assert not gate.response_ready('4662-15')


def test_restaurant_actor_subscribes_to_listen_open_events_before_answering():
    actor_source = (ROOT.parent / 'scripts/restaurant_actor.py').read_text()
    assert 'node.create_subscription(String, "/voice/result", on_voice_event, 10)' in actor_source
    assert 'listen_gate.wait_until_listening(request_id)' in actor_source
    assert 'tf_buffer.lookup_transform(' in actor_source
    assert 'tf_buffer = Buffer(node=node)' in actor_source
    assert '"map", "base_footprint", lookup_time, timeout=Duration(seconds=0.1)' in actor_source
    assert 'fresh_tf_receipt_age(now, map_odom_at)' in actor_source
    assert 'map_odom_seen_at = now' in actor_source
    assert 'odom_base_seen_at = now' in actor_source
    assert 'map_base_seen_at = now' in actor_source
    assert 'fresh_tf_receipt_age' in actor_source
    assert '"/amcl_pose"' not in actor_source
    assert 'node.create_subscription(Odometry, "/model/base_athome/odometry", on_odom, 10)' not in actor_source
    assert 'from nav_msgs.msg import Odometry' in actor_source
    assert 'Odometry, gz_model.restaurant_item_odometry_topic(name), on_can_odom, 10' in actor_source


def test_restaurant_actor_can_create_missing_counter_can_at_open_gripper_for_plan_b():
    actor_source = (ROOT.parent / 'scripts/restaurant_actor.py').read_text()
    assert 'parameter_overrides=[Parameter("use_sim_time", value=True)]' in actor_source
    assert 'actor_freshness_now(node) - arm_status_at' in actor_source
    assert 'actor_freshness_now(node) - tcp_seen_at' in actor_source
    assert 'restaurant_can_sdf(tcp_xyz, name=can_name)' in actor_source
    assert 'Barman gripper handoff refused:' in actor_source


def test_concurrent_barman_listens_stage_the_same_order_once():
    from base_athome_sim.restaurant_actor_protocol import SingleFlight

    flight = SingleFlight()
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    calls, results = [], []

    def stage():
        calls.append('stage')
        entered.set()
        assert release.wait(timeout=2)
        return True

    first = threading.Thread(target=lambda: results.append(flight.run(('order', 1), stage)))
    first.start()
    assert entered.wait(timeout=1)
    second = threading.Thread(target=lambda: (second_started.set(), results.append(flight.run(('order', 1), stage))))
    second.start()
    assert second_started.wait(timeout=1)
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert calls == ['stage'] and sorted(results) == [True, True]
    assert flight.run(('order', 1), stage) is True
    assert calls == ['stage']


def test_actor_freshness_clock_uses_ros_time_in_sim_and_monotonic_for_real(monkeypatch):
    from types import SimpleNamespace
    import base_athome_sim.restaurant_actor_protocol as protocol

    simulated = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=name == "use_sim_time"),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=12_500_000_000)),
    )
    assert protocol.actor_freshness_now(simulated) == 12.5

    real = SimpleNamespace(get_parameter=lambda name: SimpleNamespace(value=False))
    monkeypatch.setattr(protocol.time, "monotonic", lambda: 123.0)
    assert protocol.actor_freshness_now(real) == 123.0


def test_actor_tf_freshness_uses_ros_receipt_time_not_incompatible_header_stamp():
    from base_athome_sim.restaurant_actor_protocol import fresh_tf_receipt_age

    # Simulation time can be hundreds of seconds while a TF bridge still stamps
    # the message in wall time; freshness is measured when the actor receives it.
    assert fresh_tf_receipt_age(223.48, 223.0) == pytest.approx(0.48)
    assert fresh_tf_receipt_age(223.48, 220.0) is None
    assert fresh_tf_receipt_age(223.0, 223.1) is None


def test_customer_gripper_take_uses_countdown_and_physical_reach_not_arm_feedback():
    from base_athome_sim.restaurant_actor_protocol import customer_gripper_take_ready

    state = {'state': 'RELEASING_ITEM'}
    args = (state, (3.8, -2.3, 0.0), (3.0, -2.9), 'restaurant_counter_can_1',
            'Here is your coke. One. Two. Three.')
    assert customer_gripper_take_ready(*args)
    assert customer_gripper_take_ready({**state, 'state': 'WAITING_CUSTOMER'}, *args[1:])
    assert not customer_gripper_take_ready(*args[:1], (0.0, 0.0, 0.0), *args[2:])
    assert not customer_gripper_take_ready(*args[:2], None, *args[3:])
    assert not customer_gripper_take_ready(*args[:3], '', args[4])
    assert not customer_gripper_take_ready(*args[:4], 'One. Two.')
    assert not customer_gripper_take_ready({**state, 'state': 'DELIVERING_TO_CUSTOMER'}, *args[1:])


def test_tray_item_models_can_be_found_without_a_staged_record():
    from base_athome_sim.restaurant_actor_protocol import tray_model_names

    assert tray_model_names(('coke', 'water'), 1, ('coke', 'water')) == (
        ('coke', 'restaurant_counter_can_1'), ('water', 'restaurant_tray_item_1'))
    assert tray_model_names(('water',), 1, ('coke', 'water')) == (
        ('water', 'restaurant_tray_item_1'),)
    assert tray_model_names(('unknown',), 1, ('coke', 'water')) == ()


def test_customer_position_cache_preserves_target_through_gripper_countdown():
    from base_athome_sim.restaurant_actor_protocol import CustomerPositionCache

    cache = CustomerPositionCache()
    cache.observe({'state': 'APPROACHING_CUSTOMER', 'customer_idx': 0,
                   'info': {'customer_xy': [3.0, -2.9]}})
    releasing = cache.observe({'state': 'RELEASING_ITEM', 'customer_idx': 0, 'info': {}})
    assert releasing['info']['customer_xy'] == [3.0, -2.9]


def test_restaurant_actor_removes_gripper_can_on_countdown_before_release():
    actor_source = (ROOT.parent / 'scripts/restaurant_actor.py').read_text()
    assert 'node.create_subscription(String, "/hri/say_something", on_sim_speech, 10)' in actor_source
    assert 'customer_gripper_take_ready(' in actor_source
    assert 'gz_model.remove_command(world, can_name)' in actor_source
    assert 'CUSTOMER_GRIPPER_TAKE model=' in actor_source
    take_body = actor_source.split('def take_gripper_item_on_countdown', 1)[1].split('def on_sim_speech', 1)[0]
    assert take_body.index('detach_publisher.publish(Empty())') < take_body.index(
        'gz_model.remove_command(world, can_name)')
    assert 'deadline = time.monotonic() + 4.0' in take_body
    assert 'feedback_independent=true' in take_body
    assert 'no staged order is recorded' not in actor_source
    assert 'tray_model_names(items, slot, order)' in actor_source


def test_restaurant_actor_verifies_can_at_gazebo_world_tcp_before_attaching():
    actor_source = (ROOT.parent / 'scripts/restaurant_actor.py').read_text()
    handoff = actor_source.split('def place_counter_item_in_gripper', 1)[1].split(
        'def take_tray_item', 1)[0]
    assert 'lookup_transform("odom", "tcp_link"' in handoff
    assert 'wait_for_can_at_tcp(can_name, tcp_xyz, placed_at)' in handoff
    assert handoff.index('wait_for_can_at_tcp(') < handoff.index('attach_pub[can_name].publish(Empty())')


def test_failed_counter_pick_request_identifies_the_item_to_move_to_tray():
    from base_athome_sim.restaurant_actor_protocol import barman_counter_item_for_request

    state = {'state': 'WAITING_BARMAN', 'info': {'item': 'coke'}}
    assert barman_counter_item_for_request(state, {'prompt': 'Please put the coke on my tray.'}) == 'coke'
    # The state names the item; an empty or generic listen prompt must still trigger the move.
    assert barman_counter_item_for_request(state, {'prompt': 'Tell me robot yes when you are done.'}) == 'coke'
    assert barman_counter_item_for_request(state, {}) == 'coke'
    assert barman_counter_item_for_request({'state': 'WAITING_BARMAN', 'info': {}}, {}) == ''
    assert barman_counter_item_for_request({'state': 'ORDERING_BARMAN', 'info': {'item': 'coke'}},
                                           {'prompt': 'Please put the coke on my tray.'}) == ''
    actor_source = (ROOT.parent / 'scripts/restaurant_actor.py').read_text()
    assert 'return stage_flight.run(key, create_items)' in actor_source
    assert 'transfer_staged_item_to_tray(' in actor_source
    assert 'counter_item = barman_counter_item_for_request(snapshot, request)' in actor_source


def test_failed_counter_pick_moves_item_and_updates_the_staged_tray_record():
    from base_athome_sim.restaurant_actor_protocol import transfer_staged_item_to_tray

    record = {'items': ('coke', 'water'), 'tray_items': ('water',), 'tray_names': ('water_on_tray',)}
    events = []
    updated = transfer_staged_item_to_tray(
        record, 'coke', 'coke_on_tray',
        create_item=lambda: events.append('create') or True,
        remove_counter_item=lambda: events.append('remove-counter') or True,
        remove_tray_item=lambda: events.append('rollback') or True,
    )
    assert events == ['create', 'remove-counter']
    assert updated['tray_items'] == ('water', 'coke')
    assert updated['tray_names'] == ('water_on_tray', 'coke_on_tray')
    assert record['tray_items'] == ('water',)  # update is returned atomically

    failed_events = []
    failed = transfer_staged_item_to_tray(
        record, 'coke', 'coke_on_tray',
        create_item=lambda: failed_events.append('create') or True,
        remove_counter_item=lambda: failed_events.append('remove-counter') or False,
        remove_tray_item=lambda: failed_events.append('rollback') or True,
    )
    assert failed is None and failed_events == ['create', 'remove-counter', 'rollback']


def test_customer_tray_pickup_requires_waiting_state_and_ordered_tray_items():
    request = dict(state='WAITING_CUSTOMER', customer_idx=1, served=1, info=dict(items=['tea']))
    assert tray_items_for_state(request) == ('tea',)
    assert tray_items_for_state({**request, 'state': 'DELIVERING_TO_CUSTOMER'}) == ()
    assert customer_xy_for_state(request) is None  # mission index does not identify a physical table
    assert customer_xy_for_state({**request, 'info': {'items': ['tea'], 'customer_xy': [2.2, 1.7]}}) == (2.2, 1.7)


def test_customer_actor_reuses_observed_south_target_when_waiting_for_tray_confirmation():
    from base_athome_sim.restaurant_actor_protocol import CustomerPositionCache

    targets = CustomerPositionCache()
    targets.observe({'state': 'APPROACHING_CUSTOMER', 'customer_idx': 0,
                     'info': {'customer_xy': [3.0, -2.9]}})
    waiting = targets.observe({'state': 'WAITING_CUSTOMER', 'customer_idx': 0,
                               'info': {'items': ['tea']}})
    target_xy = customer_xy_for_state(waiting)
    assert target_xy == (3.0, -2.9)
    assert np.linalg.norm(np.array([2.03, -2.60]) - np.array(target_xy)) < 1.2


def test_customer_actor_does_not_guess_table_from_index_without_an_observed_approach():
    from base_athome_sim.restaurant_actor_protocol import CustomerPositionCache

    waiting = CustomerPositionCache().observe({'state': 'WAITING_CUSTOMER', 'customer_idx': 0,
                                               'info': {'items': ['water']}})
    assert customer_xy_for_state(waiting) is None


def test_actor_range_and_facing_gate_checks_distance_and_heading():
    assert pose_faces_target((-1.2, 0.0, 3.14159), (-2.6, 0.0))
    assert not pose_faces_target((-1.0, 0.0, 0.0), (-2.6, 0.0))
    assert not pose_faces_target((-1.0, 0.0, 3.14159), (-2.6, 0.0))


def test_barman_item_staging_uses_the_accessible_bar_front():
    service_pose = (-1.1, 0.0, 3.14159)
    bar_front = (-1.70, 0.0)
    assert pose_faces_target(service_pose, bar_front)
    actor_source = (ROOT.parent / 'scripts/restaurant_actor.py').read_text()
    assert 'BAR_FRONT_XY = (-1.70, 0.0)' in actor_source
    assert 'pose_faces_target(pose, BAR_FRONT_XY)' in actor_source


def test_baked_skin_is_finite_human_scale_and_has_seated_legs(tmp_path):
    path = ROOT / 'models/humans/download_restaurant.py'
    spec = importlib.util.spec_from_file_location('restaurant_rig', path)
    rig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rig)
    src = path.parent / 'male_visitor/meshes/MaleVisitorWalk.dae'
    standing = rig.bake_pose(src, tmp_path / 'standing.dae', False, 'right')
    seated = rig.bake_pose(src, tmp_path / 'seated.dae', True, 'left')
    down = rig.bake_pose(src, tmp_path / 'down.dae', True, None)
    for points in (standing, seated, down):
        assert np.isfinite(points).all()
        assert np.ptp(points[:, 0]) < .9 and np.ptp(points[:, 1]) < 1.0
        assert -.02 < points[:, 2].min() < .08
    assert 1.9 < standing[:, 2].max() < 2.2  # raised hand, not a stretched mesh
    assert 1.5 < seated[:, 2].max() < 1.7
    assert down[:, 2].max() < seated[:, 2].max()
    tree = ET.parse(tmp_path / 'seated.dae')
    assert tree.find('.//{*}instance_controller') is None  # static skin, no animation drift


def test_distinct_fuel_humans_keep_attribution_and_static_textures(tmp_path):
    path = ROOT / 'models/humans/download_restaurant.py'
    spec = importlib.util.spec_from_file_location('restaurant_rig_multi', path)
    rig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rig)
    for model, source_name, pose in (
        ('female_visitor', 'FemaleVisitorWalk.dae', 'seated_down'),
        ('nurse_female', 'NurseFemaleWalk.dae', 'standing_down'),
    ):
        directory = ROOT / f'models/humans/{model}'
        source = directory / 'meshes' / source_name
        assert source.is_file()
        metadata = json.loads((directory / 'SOURCE.json').read_text())
        assert metadata['license'] == 'CC-BY-4.0' and metadata['attribution']
        assert metadata['url'].endswith('/1')
        baked = rig.bake_pose(source, tmp_path / f'{model}.dae', seated='seated' in pose, raised=None)
        min_height = 1.0 if pose.startswith('seated') else 1.4
        assert np.isfinite(baked).all() and min_height < np.ptp(baked[:, 2]) < 2.2
        assert ET.parse(tmp_path / f'{model}.dae').find('.//{*}instance_controller') is None


def test_measured_head_never_reports_targets_or_stale_feedback(monkeypatch):
    from types import SimpleNamespace as NS
    from base_athome_sim.restaurant_humans import apply_measured_head
    import time
    monkeypatch.setattr(time, 'monotonic', lambda: 10.)
    def messages():
        return NS(name=['joint1', 'pan_joint', 'tilt_joint'], position=[.5, 1., 2.]), NS(name=['pan_joint', 'tilt_joint'], position=[1., 2.])
    full, head = messages()
    apply_measured_head(NS(rendered_head=(9.8, dict(pan_joint=.1, tilt_joint=.2))), full, head)
    assert full.position == [.5, .1, .2] and head.position == [.1, .2]
    for sample in (None, (8., dict(pan_joint=.1, tilt_joint=.2))):
        full, head = messages()
        apply_measured_head(NS(rendered_head=sample), full, head)
        assert full.name == ['joint1'] and head.name == []


def test_served_participant_gesture_latches_down_and_never_reselects():
    from base_athome_sim.restaurant_humans import ParticipantGestures, lowered_model_sdf
    gestures = ParticipantGestures()
    assert gestures.observe({"state": "APPROACHING_CUSTOMER", "customer_idx": 1,
                             "info": {"customer_xy": [2.0, 2.8]}}) == frozenset()
    retired = gestures.observe({"state": "RETURNING_TO_BAR", "customer_idx": 1, "served": 1})
    assert retired == frozenset({1})
    assert gestures.observe({"state": "DETECTING_CUSTOMERS", "served": 1}) == retired
    assert gestures.observe({"state": "ORDERING_BARMAN", "served": 1}) == retired
    assert 'seated_down.dae' in lowered_model_sdf(0)
    assert 'human_barman' in lowered_model_sdf(2)
    assert 'model://nurse_female/meshes/standing_down.dae' in lowered_model_sdf(2)


def test_requested_visible_gesture_sequence_from_search_through_both_customers():
    from base_athome_sim.restaurant_humans import ParticipantGestures
    gestures = ParticipantGestures()
    assert gestures.raised == frozenset({2})
    assert gestures.observe({"state": "STARTING"}) == frozenset({2})
    assert gestures.observe({"state": "FINDING_BARMAN"}) == frozenset({2})
    assert gestures.observe({"state": "CONFIRMING_BARMAN"}) == frozenset({2})
    assert gestures.observe({"state": "ENROLLING_FACE"}) == frozenset()
    assert gestures.observe({"state": "CLEARING_COUNTER"}) == frozenset()
    assert gestures.observe({"state": "DETECTING_CUSTOMERS"}) == frozenset({0, 1})
    assert gestures.observe({"state": "APPROACHING_CUSTOMER", "customer_idx": 0,
                             "info": {"customer_xy": [2., 2.8]}}) == frozenset({0, 1})
    assert gestures.observe({"state": "RETURNING_TO_BAR", "customer_idx": 0, "served": 1}) == frozenset({1})
    assert gestures.observe({"state": "DETECTING_CUSTOMERS", "customer_idx": 0, "served": 1}) == frozenset({1})
    assert gestures.observe({"state": "APPROACHING_CUSTOMER", "customer_idx": 1,
                             "info": {"customer_xy": [3., -2.9]}}) == frozenset({1})
    assert gestures.observe({"state": "RETURNING_TO_BAR", "customer_idx": 1, "served": 2}) == frozenset()
