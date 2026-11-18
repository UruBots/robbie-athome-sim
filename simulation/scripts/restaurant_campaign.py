#!/usr/bin/env python3
"""Bounded Restaurant runs on an isolated lab ROS domain under the Gazebo turn.

Example (source install/setup.bash first):
  python3 simulation/scripts/restaurant_campaign.py --mode real --runs 5
Runs use honest dialogue from restaurant_actor.py, retain state/perception JSONL,
benchmark the rendered camera once at a stable dialogue view, and append the lab
ledger. Points are the historical lab estimate, not official competition scores;
tray loading/unloading remains assisted and physical validation is never claimed.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from sim_domain import validate_lab_domain

REPO = Path(__file__).resolve().parents[2]
LAB = Path(os.environ.get('SIM_LAB_DIR', str(Path.home() / 'sim_lab')))
sys.path.insert(0, str(REPO / 'simulation/base_athome_sim'))
from base_athome_sim import real_gap  # noqa: E402  (pure module, no ROS import)

EVIDENCE_STATES = frozenset(('DELIVERING_TO_CUSTOMER', 'APPROACHING_CUSTOMER',
                             'RETURNING_TO_BAR'))
EVIDENCE_TOPICS = ('/scan', '/local_costmap/costmap', '/odom', '/restaurant/state',
                   '/joint_states', '/tf', '/tf_static')
EVIDENCE_MAX_BYTES = 128 * 1024 * 1024
EVIDENCE_CACHE_BYTES = 16 * 1024 * 1024
MOVEIT_REQUIRED_SERVICES = frozenset(('/compute_ik', '/plan_kinematic_path'))


def campaign_moveit_requested(arm_enabled, setting):
    """MoveIt starts for arm campaigns unless explicitly disabled for resource limits."""
    return bool(arm_enabled and setting in ('auto', 'on'))


def moveit_mode_from_environment(environment):
    value = str(environment.get('ATHOME_SIM_MOVEIT', 'auto')).strip().casefold()
    return {'1': 'on', 'true': 'on', 'yes': 'on', '0': 'off', 'false': 'off', 'no': 'off'}.get(value, value)


def moveit_services_ready(service_list):
    return MOVEIT_REQUIRED_SERVICES.issubset(set(service_list.splitlines()))


def wait_for_moveit(env, process, timeout_s=45.0, *, service_probe=None, monotonic=time.monotonic,
                    sleep=time.sleep):
    """Wait for both planning services before the mission can request arm motion."""
    if service_probe is None:
        service_probe = lambda: subprocess.run(
            ['ros2', 'service', 'list'], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=5)
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            return False, f'move_group exited with status {return_code}'
        try:
            result = service_probe()
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and moveit_services_ready(result.stdout):
            return True, 'MoveIt planning services ready'
        sleep(0.5)
    return False, f'MoveIt planning services not ready after {timeout_s:.1f}s'


def directory_size(path):
    """Total bytes written by rosbag under a campaign evidence directory."""
    total = 0
    if path.exists():
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    return total


class StateScopedRosbag:
    """Record the navigation evidence topics only during relevant Restaurant states."""

    def __init__(self, directory, start_process, stop_process, max_bytes=EVIDENCE_MAX_BYTES):
        self.directory = Path(directory)
        self.start_process = start_process
        self.stop_process = stop_process
        self.max_bytes = int(max_bytes)
        self.process = None
        self.segment = 0
        self.disabled = self.max_bytes <= 0

    def update(self, state):
        """Start/stop recording on state boundaries and enforce a total byte ceiling."""
        active = state in EVIDENCE_STATES
        if self.process is not None and self.process.poll() is not None:
            self.process = None
            self.disabled = True
        if self.process is not None:
            if not active:
                self._stop()
            elif directory_size(self.directory) >= self.max_bytes:
                self._stop()
                self.disabled = True
        elif active and not self.disabled:
            used = directory_size(self.directory)
            if used >= self.max_bytes:
                self.disabled = True
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            bag_path = self.directory / f'navigation_{self.segment:02d}'
            self.segment += 1
            command = ['ros2', 'bag', 'record', '--storage', 'mcap', '-o', str(bag_path),
                       '--max-bag-size', str(self.max_bytes - used),
                       '--max-cache-size', str(EVIDENCE_CACHE_BYTES), *EVIDENCE_TOPICS]
            try:
                self.process = self.start_process(command)
            except OSError:
                self.disabled = True

    def _stop(self):
        process, self.process = self.process, None
        if process is not None:
            self.stop_process(process)
            if directory_size(self.directory) >= self.max_bytes:
                self.disabled = True

    def finish(self):
        """Close a recording at campaign shutdown."""
        self._stop()


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # writer can be midway through its last row
                pass
    return rows


def summarize(rows, text):
    def count(state):
        return len({r['customer_idx'] for r in rows if r['state'] == state})
    detected = min(2, max([r.get('total_customers', 0) for r in rows] + [0]))
    ordered = max([sum(bool(v) for v in r.get('orders', {}).values()) for r in rows] + [0])
    reached, communicated = count('GREETING'), count('ORDERING_BARMAN')
    loaded, returned = count('DELIVERING_TO_CUSTOMER'), count('WAITING_CUSTOMER')
    served = text.count('Enjoy your order')
    points = 80 * (detected + reached + communicated + returned) + 160 * ordered + 200 * (loaded + served)
    points += (100 if loaded == 2 else 0) + (100 if served == 2 else 0)
    barman_seen = any(r.get('state') == 'CONFIRMING_BARMAN' and
                      isinstance(r.get('info', {}).get('xy'), list) and
                      math.dist(r['info']['xy'], [-2.6, 0.]) < .8 for r in rows)
    terminal = rows[-1].get('state', 'startup') if rows else 'startup'
    return dict(success=terminal == 'SUCCEEDED' and served >= 1 and barman_seen,
                mission_succeeded=terminal == 'SUCCEEDED', barman_seen=barman_seen,
                last=terminal, points_estimated=points, detected=detected, reached=reached,
                ordered=ordered, communicated=communicated, loaded=loaded, returned=returned,
                served=served, assistance='simulated human tray; no unassisted pick/place credit',
                hardware_verified=False)


def process_group_alive(pgid, *, killpg=os.killpg):
    """Check whether a campaign's isolated process group still has members."""
    try:
        killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop(process, interrupt=False, *, killpg=os.killpg, monotonic=time.monotonic,
         sleep=time.sleep, grace_s=5.0):
    """Stop the entire start_new_session process group, escalating and reaping it."""
    del interrupt  # All owned groups use the same graceful-to-forced shutdown sequence.
    pgid = process.pid
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            killpg(pgid, sig)
        except ProcessLookupError:
            pass
        deadline = monotonic() + grace_s
        while process_group_alive(pgid, killpg=killpg) and monotonic() < deadline:
            try:
                process.wait(timeout=min(0.1, max(0.0, deadline - monotonic())))
            except subprocess.TimeoutExpired:
                pass
            sleep(0.05)
        if not process_group_alive(pgid, killpg=killpg):
            process.wait()
            return
    # SIGKILL was sent. Reap the direct child and refuse to report cleanup while
    # any descendant in the isolated group remains visible.
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    if process_group_alive(pgid, killpg=killpg):
        raise RuntimeError(f'Owned process group {pgid} survived SIGKILL; no further run started')


def restaurant_dock_default(environ):
    """Docking on by default in the sim; off under ATHOME_SIM_REAL_GAP=1; an explicit RESTAURANT_DOCK wins."""
    return real_gap.manipulation_defaults(environ)['RESTAURANT_DOCK']


def campaign_environment(domain, environ=None):
    """Return the isolated environment shared by one Restaurant campaign run."""
    source = os.environ if environ is None else environ
    return dict(source, ROS_DOMAIN_ID=str(domain), SIM_DOMAIN=str(domain),
                GZ_PARTITION=f'rest_humanos_d{domain}_{os.getpid()}', GZ_IP='127.0.0.1')


def campaign_mode_switches(environ=None):
    """Keep explicit operator choices for the real-robot Restaurant mode in sim."""
    source = os.environ if environ is None else environ
    return real_gap.manipulation_defaults(source)


def clock_diagnostics(env, output):
    """Capture ROS-clock settings and representative source timestamps once per campaign."""
    def run(command, timeout=5):
        try:
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=timeout)
            return result.stdout.strip() or f'exit={result.returncode} (no output)'
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f'{type(exc).__name__}: {exc}'

    lines = ['ROS clock startup diagnostics']
    nodes = run(['ros2', 'node', 'list'], timeout=8).splitlines()
    watched = ('bag_finder', 'voice', 'actor', 'sim_robot', 'restaurant', 'amcl', 'camera')
    for node in nodes:
        if any(token in node.lower() for token in watched):
            lines.append(f'{node} use_sim_time: {run(["ros2", "param", "get", node, "use_sim_time"])}')
    for topic, field in (('/sim/gz/head/image', 'header.stamp'),
                         ('/sim/gz/head/camera_info', 'header.stamp'),
                         ('/scan', 'header.stamp'),
                         ('/scan_nav', 'header.stamp')):
        lines.append(f'{topic} {field}: {run(["ros2", "topic", "echo", topic, "--field", field, "--once"], timeout=8)}')
    lines.append('/clock: ' + run(['ros2', 'topic', 'echo', '/clock', '--field', 'clock', '--once'], timeout=8))
    lines.append('/tf sample: ' + run(['ros2', 'topic', 'echo', '/tf', '--once'], timeout=8))
    Path(output).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_one(args, index):
    label = f'humanos-restaurant-campana-{args.mode}'
    if args.scenario != 'default':
        label += '-' + args.scenario
    out = LAB / 'logs' / f'{label}_{args.tag}_{index:02d}'
    out.mkdir(parents=True, exist_ok=False)
    domain = args.domain
    env = dict(campaign_environment(domain),
               DISPLAY=os.environ.get('SIM_DISPLAY', os.environ.get('DISPLAY', ':0')),
               SIM='1', GZ_GUI=os.environ.get('GZ_GUI', '1'), AUTO='1', RVIZ='0', SIM_TABLE_RETREAT='1',
               ATHOME_SIM_REAL_PERCEPTION='1' if args.mode == 'real' else '0',
               ATHOME_SIM_WRIST_DETECTOR='yolo', ARM='1', **campaign_mode_switches(),
               SIM_TIMEOUT_S=str(args.timeout),
               RESTAURANT_ARM_POSES_FILE=str(REPO / 'simulation/base_athome_sim/config/restaurant_arm_poses.yaml'),
               RESTAURANT_LOG_DIR=str(out), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    moveit_requested = campaign_moveit_requested(args.arm, args.moveit)
    env['ATHOME_SIM_MOVEIT'] = '1' if moveit_requested else '0'
    effective_arm = bool(args.arm and moveit_requested)
    # Docking needs ARM=1 in restaurant_node; without MoveIt keep ARM=1 but leave the
    # simulated arm still (tray path), as in the 1960-point baseline runs.
    env['ARM'] = '1' if args.arm else '0'
    env['ATHOME_SIM_ARM_MOTION'] = '1' if effective_arm else '0'
    # The campaign may be invoked from a parent shell sourced for another
    # worktree; ensure actors/probes import this checkout's ROS packages.
    setup = REPO / 'install/setup.bash'
    if setup.is_file():
        prefix = subprocess.check_output(
            ['bash', '-c', f'source /opt/ros/jazzy/setup.bash && source "{setup}" && printf %s "$PYTHONPATH"'],
            cwd=REPO, text=True).strip()
        env['PYTHONPATH'] = prefix
    env.pop('BASE_ATHOME_SESSION', None)
    local_dds = Path.home() / 'cyclonedds_local.xml'
    if local_dds.exists():
        env['CYCLONEDDS_URI'] = 'file://' + str(local_dds)
    if args.scenario == 'arena':
        env['ATHOME_SIM_ARENA'] = '1'
    base = os.environ.get('SIM_BASE_SHA') or os.environ.get('SIM_CAMPAIGN_BASE')
    if not base:
        try:
            base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            # remote_sim.sh copies sources without .git metadata; callers can supply the local SHA.
            base = 'workspace-copy'
    children, streams = [], []
    def start(name, command):
        stream = (out / (name + '.log')).open('w')
        streams.append(stream)
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process
    evidence = StateScopedRosbag(
        out / 'navigation_evidence',
        lambda command: start(f'evidence-{len(children)}', command),
        lambda process: stop(process, interrupt=True))
    start('probe', ['python3', 'simulation/scripts/restaurant_perception_probe.py', str(out / 'metrics.jsonl'),
                    '--states', str(out / 'states.jsonl')])
    # Real mode uses spatially honest responses; projected mode retains its current dialogue.
    start('actor', ['python3', 'simulation/scripts/restaurant_actor.py'])
    if moveit_requested:
        moveit = start('moveit', ['ros2', 'launch', 'base_athome_moveit_config', 'move_group.launch.py'])
        moveit_ready, moveit_detail = wait_for_moveit(env, moveit)
        if not moveit_ready:
            stop(moveit, interrupt=True)
            # Keep the campaign useful under CPU or package constraints: no arm
            # action is requested, and the existing session uses the tray path.
            effective_arm = False
            env['ATHOME_SIM_ARM_MOTION'] = '0'
            env['ATHOME_SIM_MOVEIT'] = '0'
        (out / 'moveit-startup.json').write_text(json.dumps({
            'requested': True, 'ready': moveit_ready, 'detail': moveit_detail,
            'fallback': 'tray' if not effective_arm else None,
            'planning_only': True,
        }, indent=2) + '\n')
    else:
        (out / 'moveit-startup.json').write_text(json.dumps({
            'requested': False, 'ready': False, 'detail': f'--moveit={args.moveit}',
            'fallback': 'tray', 'planning_only': True,
        }, indent=2) + '\n')
    mission = start('run', ['bash', 'src/base_athome_skills/scripts/restaurant.sh'])
    (out / 'pids.json').write_text(json.dumps([p.pid for p in children]) + '\n')
    begun, unpaused, benchmark, captured, clock_logged = time.monotonic(), False, False, False, False
    try:
        while mission.poll() is None and time.monotonic() - begun < args.timeout + 120:
            elapsed = time.monotonic() - begun
            if not clock_logged and elapsed > 20:
                clock_diagnostics(env, out / 'clock-diagnostics.log')
                clock_logged = True
            if not unpaused and elapsed > 10:
                world = 'arena' if args.scenario == 'arena' else 'restaurant'
                result = subprocess.run(['gz', 'service', '-s', f'/world/{world}/control', '--reqtype',
                    'gz.msgs.WorldControl', '--reptype', 'gz.msgs.Boolean', '--timeout', '2000', '--req', 'pause: false'],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
                unpaused = 'true' in result.stdout
            if not captured and elapsed > 25:
                capture_name = f"restaurant/{label.rsplit('-', 1)[-1]}_{args.tag}_{index:02d}"
                result = subprocess.run([os.path.expanduser('~/sim_lab/')+'tools/capturar.sh', capture_name], env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
                (out / 'capture.log').write_text(result.stdout)
                captured = True
            rows = read_rows(out / 'states.jsonl')
            evidence.update(rows[-1].get('state', '') if rows else '')
            if not benchmark and rows and rows[-1]['state'] in ('CHECKING_ORDER', 'CONFIRMING_BARMAN'):
                python = os.environ.get('ATHOME_POSE_PYTHON', str(Path.home() / 'ros2_vision_venv/bin/python3'))
                bench_ready = False
                if Path(python).is_file() and os.access(python, os.X_OK):
                    try:
                        bench_ready = subprocess.run([python, '-c', 'import cv2, ultralytics'], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20).returncode == 0
                    except subprocess.TimeoutExpired:
                        pass
                if bench_ready:
                    start('bench', [python, 'tools/sim_perception_bench/bench.py', '--domain', str(domain),
                        '--topic', '/sim/gz/head/image', '--frames', '30', '--timeout', '90',
                        '--models-dir', 'src/base_athome_perception/models', '--out', str(out / 'bench')])
                else:
                    (out / 'bench.log').write_text(
                        f'SKIP: perception benchmark requires cv2 and ultralytics in {python}\n')
                benchmark = True
            time.sleep(.1)
        stop(mission)
    finally:
        cleanup_errors = []
        try:
            evidence.finish()
        except Exception as exc:
            cleanup_errors.append(exc)
        for child in reversed(children):
            try:
                stop(child, interrupt=child is not mission)
            except Exception as exc:
                cleanup_errors.append(exc)
        for stream in streams:
            try:
                stream.close()
            except OSError as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise RuntimeError(
                f'Campaign cleanup failed for {len(cleanup_errors)} owned resource(s)') from cleanup_errors[0]
    elapsed = time.monotonic() - begun
    rows = read_rows(out / 'states.jsonl')
    text = (out / 'run.log').read_text()
    summary = summarize(rows, text)
    summary.update(elapsed_s=round(elapsed, 2), commit=base, mode=args.mode,
                   scenario=args.scenario, arm=effective_arm, exit=mission.returncode)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    metrics = {}
    for row in read_rows(out / 'metrics.jsonl'):
        metrics[row['scenario']] = row
    (out / 'metrics_summary.json').write_text(json.dumps(metrics, indent=2) + '\n')
    with (LAB / 'ledger.tsv').open('a') as ledger:
        ledger.write('\t'.join([datetime.datetime.now().astimezone().isoformat(), 'codex-restaurant', label, base,
            f'{args.tag}-{index:02d}', 'PASS' if summary['success'] else 'FAIL', str(summary['points_estimated']),
            summary['last'], str(out / 'run.log'),
            f"elapsed_s={elapsed:.2f}; arm={effective_arm}; stage={summary['last']}; metrics={out / 'metrics_summary.json'}"]) + '\n')
    print(json.dumps(dict(out=str(out), **summary)), flush=True)
    return summary['success']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('real', 'projected'), required=True)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--tag', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--scenario', choices=('default', 'arena'), default='default')
    parser.add_argument('--domain', help='isolated lab ROS domain (default: SIM_DOMAIN or ROS_DOMAIN_ID)')
    parser.add_argument('--arm', action='store_true', default=True,
                        help='simulated arm and docking are always enabled')
    parser.add_argument('--moveit', choices=('auto', 'on', 'off'),
                        default=moveit_mode_from_environment(os.environ),
                        help='start planning-only MoveIt for arm runs; off selects tray fallback')
    parser.add_argument('--turn-held', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.domain = validate_lab_domain(args.domain or os.environ.get('SIM_DOMAIN')
                                          or os.environ.get('ROS_DOMAIN_ID'))
    except ValueError as exc:
        parser.error(str(exc))
    if not (1 <= args.runs <= 20 and math.isfinite(args.timeout) and 30 <= args.timeout <= 1800):
        parser.error('runs must be 1..20 and timeout 30..1800 seconds')
    if not args.turn_held:
        env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain), SIM_DOMAIN=str(args.domain))
        return subprocess.call([str(LAB / 'tools/gz_turno.sh'), sys.executable, str(Path(__file__).resolve()),
                                *sys.argv[1:], '--turn-held'], cwd=REPO, env=env)
    results = [run_one(args, i + 1) for i in range(args.runs)]
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
