"""Bounded campaign evidence collection stays scoped to navigation states."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

SIM_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SIM_ROOT.parent / 'scripts'
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(SCRIPTS))

from base_athome_sim.restaurant_actor_protocol import DynamicCanFallTracker

spec = importlib.util.spec_from_file_location('restaurant_campaign_evidence',
                                               SCRIPTS / 'restaurant_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_campaign_environment_uses_local_gazebo_and_pid_scoped_partition(monkeypatch):
    monkeypatch.setattr(campaign.os, 'getpid', lambda: 2468)

    env = campaign.campaign_environment(42, {'KEEP': 'value', 'GZ_IP': 'remote'})

    assert env['GZ_IP'] == '127.0.0.1'
    assert env['GZ_PARTITION'] == 'rest_humanos_d42_2468'
    assert env['ROS_DOMAIN_ID'] == '42'
    assert env['KEEP'] == 'value'


def test_campaign_mode_switches_preserve_explicit_real_robot_settings():
    assert campaign.campaign_mode_switches({
        'RESTAURANT_DOCK': '0', 'VISUAL_PICK': '0',
    }) == {'RESTAURANT_DOCK': '0', 'VISUAL_PICK': '0'}
    assert campaign.campaign_mode_switches({}) == {
        'RESTAURANT_DOCK': '1', 'VISUAL_PICK': '1',
    }


def test_campaign_accepts_one_delivered_customer_when_mission_succeeds():
    rows = [
        {'state': 'CONFIRMING_BARMAN', 'customer_idx': 0, 'total_customers': 1,
         'info': {'xy': [-2.6, 0.0]}},
        {'state': 'SUCCEEDED', 'customer_idx': 0, 'total_customers': 1,
         'info': {'served': 1}},
    ]

    summary = campaign.summarize(rows, 'Enjoy your order. See you soon.')

    assert summary['success'] is True
    assert summary['served'] == 1


def test_clock_diagnostics_records_node_parameters_and_source_stamps(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output = '/bag_finder\n/restaurant_actor\n/sim_robot' if command[1:3] == ['node', 'list'] else 'sample'
        return SimpleNamespace(stdout=output, returncode=0)

    monkeypatch.setattr(campaign.subprocess, 'run', fake_run)
    path = tmp_path / 'clock.log'
    campaign.clock_diagnostics({'ROS_DOMAIN_ID': '42'}, path)
    text = path.read_text()
    assert '/bag_finder use_sim_time: sample' in text
    assert '/restaurant_actor use_sim_time: sample' in text
    assert '/sim/gz/head/image header.stamp: sample' in text
    assert '/sim/gz/head/camera_info header.stamp: sample' in text
    assert '/scan header.stamp: sample' in text
    assert '/scan_nav header.stamp: sample' in text
    assert '/clock: sample' in text and '/tf sample: sample' in text


class OwnedProcess:
    pid = 12345

    def __init__(self, group_state):
        self.group_state = group_state
        self.reaped = False

    def wait(self, timeout=None):
        if self.group_state['alive']:
            raise campaign.subprocess.TimeoutExpired('owned', timeout)
        self.reaped = True
        return 0


def test_campaign_stop_signals_and_reaps_the_whole_process_group(monkeypatch):
    signals = []
    state = {'alive': True}

    def killpg(pgid, sig):
        assert pgid == 12345
        if sig == 0:
            if state['alive']:
                return
            raise ProcessLookupError
        signals.append(sig)
        state['alive'] = False

    process = OwnedProcess(state)
    campaign.stop(process, killpg=killpg)
    assert signals == [campaign.signal.SIGINT]
    assert process.reaped


def test_campaign_stop_escalates_from_interrupt_to_term_then_kill():
    signals = []
    state = {'alive': True}

    def killpg(pgid, sig):
        assert pgid == 12345
        if sig == 0:
            if state['alive']:
                return
            raise ProcessLookupError
        signals.append(sig)
        if sig == campaign.signal.SIGKILL:
            state['alive'] = False

    process = OwnedProcess(state)
    campaign.stop(process, killpg=killpg, sleep=lambda _: None, grace_s=0.002)
    assert signals == [campaign.signal.SIGINT, campaign.signal.SIGTERM, campaign.signal.SIGKILL]
    assert process.reaped


class FakeProcess:
    def poll(self):
        return None


def test_campaign_rosbag_starts_only_for_delivery_approach_and_return(tmp_path):
    commands, stopped = [], []
    recorder = campaign.StateScopedRosbag(
        tmp_path / 'evidence', lambda command: commands.append(command) or FakeProcess(),
        stopped.append, max_bytes=4096)

    recorder.update('ORDERING_BARMAN')
    assert commands == []
    recorder.update('DELIVERING_TO_CUSTOMER')
    assert commands[0][0:6] == ['ros2', 'bag', 'record', '--storage', 'mcap', '-o']
    assert commands[0][-len(campaign.EVIDENCE_TOPICS):] == list(campaign.EVIDENCE_TOPICS)
    assert {'/joint_states', '/tf', '/tf_static'}.issubset(campaign.EVIDENCE_TOPICS)
    assert commands[0][commands[0].index('--max-bag-size') + 1] == '4096'
    assert commands[0][commands[0].index('--max-cache-size') + 1] == str(campaign.EVIDENCE_CACHE_BYTES)
    first_process = recorder.process

    recorder.update('WAITING_CUSTOMER')
    assert stopped == [first_process]
    recorder.update('APPROACHING_CUSTOMER')
    second_process = recorder.process
    recorder.update('RETURNING_TO_BAR')
    assert recorder.process is second_process
    recorder.update('SUCCEEDED')
    assert stopped == [first_process, second_process]
    assert len(commands) == 2


def test_campaign_rosbag_total_size_limit_disables_later_segments(tmp_path):
    stopped, commands = [], []
    evidence = tmp_path / 'evidence'

    def start(command):
        commands.append(command)
        bag = Path(command[command.index('-o') + 1])
        bag.mkdir()
        (bag / 'recording.mcap').write_bytes(b'x' * 1024)
        return FakeProcess()

    recorder = campaign.StateScopedRosbag(evidence, start, stopped.append, max_bytes=1024)
    recorder.update('DELIVERING_TO_CUSTOMER')
    recorder.update('APPROACHING_CUSTOMER')
    assert recorder.process is None
    assert len(stopped) == 1
    recorder.update('RETURNING_TO_BAR')
    assert len(commands) == 1
    assert recorder.disabled


def test_campaign_starts_moveit_for_arm_mode_unless_resource_fallback_is_requested():
    assert campaign.campaign_moveit_requested(True, 'auto')
    assert campaign.campaign_moveit_requested(True, 'on')
    assert not campaign.campaign_moveit_requested(True, 'off')
    assert not campaign.campaign_moveit_requested(False, 'on')
    assert campaign.moveit_mode_from_environment({'ATHOME_SIM_MOVEIT': '0'}) == 'off'
    assert campaign.moveit_mode_from_environment({'ATHOME_SIM_MOVEIT': '1'}) == 'on'
    assert campaign.moveit_mode_from_environment({}) == 'auto'


def test_campaign_waits_for_ik_and_collision_planning_services():
    ready = '/compute_ik\n/plan_kinematic_path\n'
    process = FakeProcess()
    ok, reason = campaign.wait_for_moveit(
        {}, process, service_probe=lambda: SimpleNamespace(returncode=0, stdout=ready),
        monotonic=lambda: 0.0, sleep=lambda _: None)
    assert ok and 'ready' in reason
    assert campaign.moveit_services_ready(ready)
    assert not campaign.moveit_services_ready('/compute_ik\n')

    class ExitedProcess:
        def poll(self):
            return 1

    ok, reason = campaign.wait_for_moveit({}, ExitedProcess(), service_probe=lambda: None)
    assert not ok and 'exited with status 1' in reason


def test_default_moveit_service_probe_constructs_valid_subprocess_kwargs(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='/compute_ik\n/plan_kinematic_path\n')

    monkeypatch.setattr(campaign.subprocess, 'run', run)
    ok, _ = campaign.wait_for_moveit(
        {'ROS_DOMAIN_ID': '66'}, FakeProcess(), monotonic=lambda: 0.0, sleep=lambda _: None)

    assert ok and len(calls) == 1
    command, kwargs = calls[0]
    assert command == ['ros2', 'service', 'list']
    assert kwargs['stdout'] == campaign.subprocess.PIPE
    assert kwargs['stderr'] == campaign.subprocess.DEVNULL
    assert 'capture_output' not in kwargs
    assert kwargs['env'] == {'ROS_DOMAIN_ID': '66'}


def test_moveit_planning_uses_the_robot_urdf_loaded_by_simulation():
    repo = SIM_ROOT.parents[1]
    moveit_launch = (repo / 'src/base_athome_moveit_config/launch/move_group.launch.py').read_text()
    sim_launch = (SIM_ROOT / 'launch/sim.launch.py').read_text()
    robot_urdf = 'get_package_share_directory("hoverboard_base"), "urdf", "robot.urdf"'
    assert robot_urdf in moveit_launch and robot_urdf in sim_launch
    assert '"allow_trajectory_execution": False' in moveit_launch
    assert '.planning_pipelines(pipelines=["ompl"]' in moveit_launch


def test_dynamic_can_fall_is_reported_once_per_fall_with_finite_xy():
    tracker = DynamicCanFallTracker()
    assert not tracker.observe('late_publisher_can', -1.8, 0.0, 0.1)
    assert not tracker.observe('restaurant_counter_can_1', -1.8, 0.0, 0.9)
    assert tracker.observe('restaurant_counter_can_1', -1.7, 0.2, 0.29)
    assert not tracker.observe('restaurant_counter_can_1', -1.6, 0.3, 0.2)
    assert not tracker.observe('restaurant_counter_can_1', float('nan'), 0.0, 0.2)
    assert not tracker.observe('restaurant_counter_can_1', -1.6, 0.3, 0.9)
    assert tracker.observe('restaurant_counter_can_1', -1.5, 0.4, 0.1)

    actor = (SCRIPTS / 'restaurant_actor.py').read_text()
    assert 'restaurant_item_odometry_topic(name)' in actor
    assert 'DYNAMIC_CAN_FELL' in actor
