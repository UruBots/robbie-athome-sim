"""Opt-in rendered visitors: image provenance, actor lifecycle and rate accounting."""
import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from base_athome_sim import receptionist_humans as humans

campaign_spec = importlib.util.spec_from_file_location(
    'receptionist_campaign', Path(__file__).resolve().parents[2] / 'scripts/receptionist_campaign.py')
campaign = importlib.util.module_from_spec(campaign_spec)
campaign_spec.loader.exec_module(campaign)
experiment_spec = importlib.util.spec_from_file_location(
    'receptionist_real_perception', Path(__file__).resolve().parents[2] / 'scripts/receptionist_real_perception.py')
experiment = importlib.util.module_from_spec(experiment_spec)
experiment_spec.loader.exec_module(experiment)

ROOT = Path(__file__).resolve().parents[3]


def method(path, class_name, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


def test_mode_requires_both_flags(monkeypatch):
    monkeypatch.setenv('ATHOME_SIM_REAL_PERCEPTION', '1')
    monkeypatch.delenv('ATHOME_SIM', raising=False)
    assert not humans.enabled()
    monkeypatch.setenv('ATHOME_SIM', '1')
    assert humans.enabled()


@pytest.mark.parametrize('end', [(3.5, 1.5), (3.5, -1.5), (-1.35, 0.0), (0.0, 0.0)])
def test_translation_is_bounded_and_finishes_exactly_on_target(end):
    start = (0, 0)
    samples = humans.path_samples(start, end)
    assert samples[-1] == end
    for previous, current in zip([start] + samples, samples):
        assert math.dist(previous, current) <= 0.65 * 0.15 + 1e-12


def test_mesh_is_textured_asset_at_ground_level_without_proxy_collisions(tmp_path):
    asset = 'FemaleVisitorSit'
    mesh = tmp_path / asset / 'meshes' / humans.MESHES[asset]
    mesh.parent.mkdir(parents=True)
    mesh.touch()
    root = ET.fromstring(humans.visitor_sdf('recep_guest_1', (3.5, 1.5), asset, root=tmp_path))
    assert root.find('model').attrib['name'] == 'recep_guest_1'
    assert root.find('.//mesh/uri').text == mesh.as_uri()
    assert root.find('.//collision') is None
    assert root.find('model/pose').text.split()[:3] == ['3.5', '1.5', '0']


def test_missing_assets_fail_before_scene_creation(tmp_path):
    with pytest.raises(FileNotFoundError, match='--fetch'):
        humans.visitor_sdf('guest', (0, 0), 'FemaleVisitor', root=tmp_path)


def test_receptionist_world_does_not_spawn_duplicate_furniture():
    director = object.__new__(humans.ReceptionistHumans)
    director.lock = __import__('threading').RLock()
    director.ready = __import__('threading').Event()
    director.world = 'receptionist_humans'
    director.node = SimpleNamespace(receptionist_host_xy=(3.5, 0), receptionist_drop_xy=(4.35, 0.55))
    commands = []
    director.command = commands.append
    director.replace = lambda *args: None
    director.arrive = lambda index: None
    director.spawn()
    assert director.ready.is_set()
    assert commands == []


def test_campaign_ledger_rates_keep_numerators_denominators_and_na():
    assert campaign.rate_note([dict(guest=2, state=15, face_hits=2, frames=4,
        pose_hits=3, pose_frames=5, seated_hits=2, attribute_hits=0, attribute_frames=0)]) == (
        'g2s15:face2/4=50%/pose3/5=60%/seated2/5=40%/attributes0/0=NA')
    assert campaign.rate_note([]) == 'unavailable'


def test_voice_confirmation_requires_current_guest_to_have_sat():
    prompt = 'Please answer yes or no'
    assert experiment.simulated_answer(prompt, 1, []) is None
    assert experiment.simulated_answer(prompt, 2, [{'guest_index': 1}]) is None
    assert experiment.simulated_answer(prompt, 2, [{'guest_index': 2}]) == 'yes'
    assert 'John' in experiment.simulated_answer('What is your name?', 1, [])
    assert 'cider' in experiment.simulated_answer('What is your name?', 1, []).lower()
    assert 'Mary' in experiment.simulated_answer('What is your name?', 2, [])
    assert 'red wine' in experiment.simulated_answer('What is your name?', 2, []).lower()


@pytest.mark.parametrize('env,expected', [
    ({}, 45.0),
    ({'ATHOME_SIM_REAL_PERCEPTION': '1'}, 45.0),
    ({'ATHOME_SIM': '1'}, 45.0),
    ({'ATHOME_SIM': '1', 'ATHOME_SIM_REAL_PERCEPTION': '1'}, 90.0),
])
def test_extended_seat_wait_is_restricted_to_rendered_simulation(env, expected):
    path = ROOT / 'src/base_athome_perception/scripts/hri_receptionist_challenge.py'
    tree = ast.parse(path.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'seat_wait_timeout')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(path), 'exec'), namespace)
    assert namespace['seat_wait_timeout'](env) == expected


def test_campaign_detects_gazebo_death_in_its_own_process_group(monkeypatch):
    process = SimpleNamespace(pid=1234, poll=lambda: None)
    monkeypatch.setattr(campaign.subprocess, 'check_output',
        lambda *args, **kwargs: '1234 gz sim -r receptionist_humans.sdf\n9999 gz sim other.sdf\n')
    assert campaign.gazebo_alive(process)
    monkeypatch.setattr(campaign.subprocess, 'check_output',
        lambda *args, **kwargs: '9999 gz sim other.sdf\n')
    assert not campaign.gazebo_alive(process)


def test_campaign_aborts_mission_promptly_when_gazebo_dies(monkeypatch):
    mission = SimpleNamespace(poll=lambda: None, args=['receptionist_real_perception.py'])
    simulator = SimpleNamespace(pid=1234, poll=lambda: None)
    monkeypatch.setattr(campaign, 'gazebo_alive', lambda _: False)
    with pytest.raises(RuntimeError, match='Gazebo exited'):
        campaign.wait_for_hri(mission, simulator, timeout_s=30)


def test_campaign_detects_gazebo_death_even_if_child_just_finished(monkeypatch):
    benchmark = SimpleNamespace(poll=lambda: 0, args=['bench.py'])
    monkeypatch.setattr(campaign, 'gazebo_alive', lambda _: False)
    with pytest.raises(RuntimeError, match='Gazebo exited'):
        campaign.wait_for_hri(benchmark, object(), timeout_s=150)


def test_seating_keeps_first_guest_when_second_arrives(monkeypatch):
    monkeypatch.setattr(humans.time, 'sleep', lambda _: None)
    calls = []
    director = object.__new__(humans.ReceptionistHumans)
    director.lock = __import__('threading').RLock()
    director.positions = {'recep_guest_1': (-1.35, 0), 'recep_guest_2': (-1.35, 0)}
    director.sitting = ('FemaleVisitorSit', 'MaleVisitorSit')
    director.attended = set()
    director.world = 'test'
    director.command = calls.append
    director.replace = lambda name, xy, asset, **kw: calls.append(('replace', name, xy, asset))
    records = []
    director.node = SimpleNamespace(base=SimpleNamespace(x=2.0, y=0.0), _record_seated_guest=lambda xy, i: records.append((xy, i)),
        _publish_receptionist_state=lambda: calls.append('state'),
        get_logger=lambda: SimpleNamespace(info=lambda _: None))
    director.seat((3.5, 1.5), 1)
    director.seat((3.5, -1.5), 2)
    initial_calls = list(calls)
    calls.clear()
    director.seat((3.5, 1.5), 1)  # repeated prompt must not reselect/reanimate guest 1
    assert director.positions['recep_guest_1'] == (3.5, 1.5)
    assert director.positions['recep_guest_2'] == (3.5, -1.5)
    assert records == [((3.5, 1.5), 1), ((3.5, -1.5), 2)]
    assert ('replace', 'recep_guest_1', (3.5, 1.5), 'FemaleVisitorSit') in initial_calls
    assert ('replace', 'recep_guest_2', (3.5, -1.5), 'MaleVisitorSit') in initial_calls
    assert calls == []
    assert director.attended == {1, 2}


def test_real_mode_never_overlays_rgb_or_depth():
    fn = method('simulation/base_athome_sim/base_athome_sim/sim_robot.py',
        'SimRobot', '_overlay_guest', {'os': SimpleNamespace(environ={'ATHOME_SIM_REAL_PERCEPTION': '1'})})
    # No guest fields are present: any access beyond the provenance guard fails.
    fn(SimpleNamespace(), object(), object())


def test_real_mode_occupancy_ignores_ground_truth():
    fn = method('src/base_athome_perception/scripts/hri_receptionist_challenge.py',
        'HriReceptionistChallengeNode', 'scan_seats', {
            'os': SimpleNamespace(environ={'ATHOME_SIM': '1', 'ATHOME_SIM_REAL_PERCEPTION': '1'}),
            'Tuple': tuple, 'Dict': dict, 'List': list,
            'seatocc': SimpleNamespace(merge_nearby=lambda p: p)})
    logs = []
    # No sim_receptionist_state field: a ground-truth read would fail.
    node = SimpleNamespace(seats={}, get_logger=lambda: SimpleNamespace(info=logs.append))
    assert fn(node) == ({}, False)
    assert 'imagen' in logs[0]


def test_detection_rate_denominators_separate_inference_from_throttled_frames():
    fn = method('src/base_athome_perception/scripts/hri_receptionist_challenge.py',
        'HriReceptionistChallengeNode', '_real_perception_sample', {'json': json})
    logs = []
    node = SimpleNamespace(guest_index=1, state=3, get_logger=lambda: SimpleNamespace(info=logs.append))
    fn(node, 10.0, [{}], [{'posture': 'SENTADO'}], {'torso': 'blue'})
    fn(node, 10.1, [], None, None)
    fn(node, 12.1, [], [], {})
    sample = json.loads(logs[-1].split(' ', 1)[1])
    assert sample['frames'] == 3 and sample['face_hits'] == 1
    assert sample['pose_frames'] == 2 and sample['pose_hits'] == 1
    assert sample['seated_hits'] == 1
    assert sample['attribute_frames'] == 2 and sample['attribute_hits'] == 1


def test_pose_sync_waits_for_inference_after_camera_settles():
    clock = [10.0]
    fn = method('src/base_athome_perception/scripts/hri_receptionist_challenge.py',
        'HriReceptionistChallengeNode', '_wait_real_pose', {
            'time': SimpleNamespace(monotonic=lambda: clock[0])})
    node = SimpleNamespace(request_shutdown=False, face_lock=__import__('threading').Lock(),
                           face_dets_stamp=9.0, last_pose_time=9.0)
    def sleep(delay):
        clock[0] += delay
        if clock[0] > 10.1:
            node.face_dets_stamp = node.last_pose_time = 10.1
    node.sleep_gui = sleep
    assert fn(node, 10.0)
    node.face_dets_stamp = node.last_pose_time = 9.0
    node.sleep_gui = lambda delay: clock.__setitem__(0, clock[0] + delay)
    assert not fn(node, 11.0, timeout_s=0.1)


def test_only_idempotent_pose_requests_retry_after_a_missing_response(monkeypatch):
    monkeypatch.setattr(humans.time, 'sleep', lambda _: None)
    replies = iter(['', 'data: true'])
    calls = []
    director = object.__new__(humans.ReceptionistHumans)
    def run(cmd):
        calls.append(cmd)
        return next(replies)
    director.node = SimpleNamespace(_gz_run=run)
    director.world = 'test'
    director.command(humans.gz_model.set_pose_command('test', 'guest', (1, 2, 0)))
    assert len(calls) == 2
    assert calls[0][calls[0].index('--timeout') + 1] == '5000'
    calls.clear()
    director.node._gz_run = lambda cmd: calls.append(cmd) or ''
    with pytest.raises(RuntimeError, match='command failed'):
        director.command(humans.gz_model.remove_command('test', 'guest'))
    assert len([c for c in calls if c[3].endswith('/remove')]) == 1


def test_fresh_seated_image_evidence_survives_planar_depth_projection_error():
    import numpy as np
    import time
    from base_athome_perception import seat_occupancy
    fn = method('src/base_athome_perception/scripts/hri_receptionist_challenge.py',
        'HriReceptionistChallengeNode', 'scan_seats', {
            'os': SimpleNamespace(environ={'ATHOME_SIM': '1', 'ATHOME_SIM_REAL_PERCEPTION': '1'}),
            'Tuple': tuple, 'Dict': dict, 'List': list, 'time': time,
            'math': math, 'json': json, 'seatocc': seat_occupancy})
    node = SimpleNamespace(real_perception=True,
        seats={'left': {'x': 3.5, 'y': 1.5, 'capacity': 1}},
        _head_pan_to_seat=lambda key: 45.0, move_head_smooth=lambda *a: None,
        SEAT_TILT_DEG=-22.0, sleep_gui=lambda _: None, _wait_real_pose=lambda _: True,
        face_lock=__import__('threading').Lock(), depth_lock=__import__('threading').Lock(),
        current_poses=[{'bbox': [250, 100, 390, 450], 'posture': 'SENTADO'}],
        latest_frame=np.zeros((480, 640, 3)), latest_depth_m=np.full((480, 640), 5.0),
        cam_fx_cx=(615.0, 320.0), robot_pose=(2.0, 0.0, 0.0), head_pan_sign=1.0,
        get_logger=lambda: SimpleNamespace(info=lambda _: None))
    assert fn(node) == ({'left': 1}, True)


def test_following_uses_walked_trail_at_corners_instead_of_crossing_wall():
    # A tangent offset at the turn to south would put the robot at y=3.17,
    # through the y=3 wall. Following the polyline keeps it below y=2.45.
    history = [(2.6, 2.45), (4.35, 2.45), (4.35, 2.35)]
    x, y = humans.trail_position(history)
    assert (x, y) == pytest.approx((3.73, 2.45))
    assert humans.trail_position([(0, 0), (0.2, 0)]) is None


def test_garment_material_selection_preserves_skin_hair_and_accessories():
    assert humans.garment_material('MaleVisitorSit', 0.9, (40, 40, 40), 'body')
    assert not humans.garment_material('MaleVisitorSit', 1.3, (40, 40, 40), 'hair')
    assert not humans.garment_material('MaleVisitorSit', 0.9, (130, 90, 65), 'body')
    assert not humans.garment_material('MaleVisitorSit', 0.9, (40, 40, 40), 'Phone.001')
    assert humans.garment_material('FemaleVisitorSit', 0.9, (25, 50, 35), 'body')
    assert not humans.garment_material('FemaleVisitorSit', 0.9, (130, 90, 65), 'body')


def test_door_oscillation_moves_guest_laterally(monkeypatch):
    monkeypatch.setattr(humans.time, 'sleep', lambda _: None)
    director = object.__new__(humans.ReceptionistHumans)
    director.lock = __import__('threading').RLock()
    director.ready = __import__('threading').Event()
    director.ready.set()
    director.world = 'test'
    director.positions = {}
    director.arrived = set()
    director.attended = set()
    director.standing = ('MaleVisitorOnPhone', 'FemaleVisitor')
    director.sitting = ('MaleVisitorSit', 'FemaleVisitorSit')
    director._sway_stop = __import__('threading').Event()
    director._sway_thread = None
    director._swaying_guest = None
    director.door_sway_amp = 0.25
    director.door_sway_period = 8.0
    director.node = SimpleNamespace(person_xy=[-1.35, 0.0],
                                    get_logger=lambda: SimpleNamespace(info=lambda _: None))
    calls = []
    director.command = calls.append
    director.replace = lambda name, xy, asset, yaw=0, **kw: director.positions.__setitem__(name, tuple(xy))
    director.walk = lambda name, end: director.positions.__setitem__(name, tuple(end))

    director.arrive(1)
    assert 1 in director.arrived
    assert director._swaying_guest == 1
    # Give the sway worker a moment to execute a step
    import time
    time.sleep(0.2)
    assert director.positions['recep_guest_1'][0] == -1.35
    assert abs(director.positions['recep_guest_1'][1]) <= 0.25
    director._stop_door_sway()
    assert director._sway_thread is None
