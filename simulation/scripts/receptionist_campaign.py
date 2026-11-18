#!/usr/bin/env python3
"""Finite, isolated rendered-camera A/B campaign (speech/navigation are mocks).

Source ROS Jazzy and this worktree's install first. Run with the vision venv:
  ROS_DOMAIN_ID=77 GZ_PARTITION=recep_humanos DISPLAY=:1 ATHOME_SIM=1 \
    python simulation/scripts/receptionist_campaign.py --runs 5 --modes default real
Each mode uses the SAME rendered visitors; only real enables image-based seat
occupancy/confirmation. Tests and benchmarks never supply detections to the HRI.
No hardware, audio playback, photographs, Nav2 goals, screen captures or sudo.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from receptionist_evaluator import evaluate_receptionist_run

ROOT = Path(__file__).resolve().parents[2]


def stop(process):
    """Only process groups started by this campaign; never unbounded kill rules."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def gazebo_alive(process):
    """Return whether this campaign's process group still owns a gz sim process."""
    if process is None or process.poll() is not None:
        return False
    try:
        rows = subprocess.check_output(
            ['ps', '-eo', 'pgid=,args='], text=True, timeout=2).splitlines()
    except (subprocess.SubprocessError, OSError):
        return False
    return any(row.split(None, 1)[0] == str(process.pid)
               and 'gz sim' in row and 'receptionist_campaign.py' not in row
               for row in rows if row.split())


def wait_for_hri(process, simulator, timeout_s):
    """Watch a campaign child and Gazebo, failing promptly if the renderer exits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not gazebo_alive(simulator):
            raise RuntimeError('Gazebo exited while Receptionist mission was running')
        code = process.poll()
        if code is not None:
            return code
        time.sleep(0.5)
    raise subprocess.TimeoutExpired(process.args, timeout_s)


def metrics(log):
    samples = {}
    for line in log.splitlines():
        if 'REAL_PERCEPTION_METRICS ' in line:
            try:
                sample = json.loads(line.split('REAL_PERCEPTION_METRICS ', 1)[1])
                samples[sample['guest'], sample['state']] = sample
            except ValueError:
                pass
    return list(samples.values())


def rate_note(samples):
    """Compact cumulative detection rates for ledger entries; retain n/N."""
    if not samples:
        return 'unavailable'

    def ratio(hit, total):
        return f'{hit}/{total}={100.0 * hit / total:.0f}%' if total else '0/0=NA'

    return ','.join(
        f"g{s['guest']}s{s['state']}:face{ratio(s['face_hits'], s['frames'])}"
        f"/pose{ratio(s['pose_hits'], s['pose_frames'])}"
        f"/seated{ratio(s['seated_hits'], s['pose_frames'])}"
        f"/attributes{ratio(s['attribute_hits'], s['attribute_frames'])}"
        for s in sorted(samples, key=lambda row: (row['guest'], row['state'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--modes', nargs='+', choices=['default', 'real'], default=['default', 'real'])
    parser.add_argument('--out', type=Path, default=Path(os.path.expanduser('~/sim_lab/')+'logs'))
    parser.add_argument('--ledger', type=Path, default=Path(os.path.expanduser('~/sim_lab/')+'ledger.tsv'))
    parser.add_argument('--world', default='receptionist_humans.sdf')
    parser.add_argument('--tag', default='humanos-receptionist-campana')
    parser.add_argument('--vision-python', type=Path,
                        default=Path.home()/('ros2_vision_venv/bin/python3'))
    parser.add_argument('--sim-start-timeout', type=int, default=180,
                        help='maximum seconds for this run to reach the first visitor (default: 180)')
    args = parser.parse_args()
    if (os.environ.get('ROS_DOMAIN_ID') != '77' or os.environ.get('GZ_PARTITION') != 'recep_humanos'
            or os.environ.get('ATHOME_SIM') != '1'):
        raise SystemExit('Requires isolated domain77 / recep_humanos / ATHOME_SIM=1')
    if not 1 <= args.runs <= 20:
        raise SystemExit('--runs must be 1..20')
    if not args.vision_python.is_file():
        raise SystemExit(f'Vision Python not found: {args.vision_python}')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    from base_athome_sim import gz_model
    rows = []
    for mode in args.modes:
        for iteration in range(1, args.runs + 1):
            folder = args.out / f'{args.tag}-{mode}-{iteration:02d}'
            folder.mkdir(parents=True, exist_ok=False)
            env = os.environ.copy()
            env.update(ATHOME_SIM_RENDERED_HUMANS='1',
                       ATHOME_SIM_HUMANS_DIR=str(ROOT / 'simulation/base_athome_sim/models/humans'),
                       ATHOME_SIM_REAL_PERCEPTION=str(int(mode == 'real')),
                       ATHOME_SIM_WORLD=args.world, ATHOME_FACES_DB=str(folder / 'synthetic_faces'),
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            start = time.monotonic()
            sim = hri = None
            samples = []
            capture_path = None
            failure = 'sim_start'
            outcome = 'FAIL'
            try:
                with (folder / 'sim.log').open('w') as output:
                    sim = subprocess.Popen([os.path.expanduser('~/sim_lab/')+'tools/gz_turno.sh', 'ros2', 'launch', 'base_athome_sim', 'sim.launch.py',
                        'gazebo:=true', 'gz_gui:=true', 'rviz:=false', 'auto_guest:=true', 'world:=' + args.world],
                        cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + args.sim_start_timeout
                gz_seen = False
                while time.monotonic() < deadline:
                    if sim.poll() is not None:
                        raise RuntimeError('Simulator exited before creating visitors')
                    if gazebo_alive(sim):
                        gz_seen = True
                    elif gz_seen:
                        raise RuntimeError('Gazebo exited before creating visitors')
                    if 'stage=at_door' in (folder / 'sim.log').read_text(errors='replace'):
                        break
                    time.sleep(0.2)
                else:
                    raise TimeoutError('Rendered visitor never reached door')
                # Fix a reproducible greeting view before sampling real images.
                command = gz_model.set_pose_command(Path(args.world).stem, gz_model.MODEL, (0, 0, 0), yaw=math.pi)
                response = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                if response.returncode or 'true' not in response.stdout:
                    raise RuntimeError('Gazebo viewpoint command failed')
                time.sleep(2)
                capture_name = f'receptionist/{args.tag}_{mode}_{iteration:02d}'
                capture = subprocess.run([os.path.expanduser('~/sim_lab/')+'tools/capturar.sh', capture_name],
                    env=env, capture_output=True, text=True, timeout=20)
                if capture.returncode:
                    raise RuntimeError('Gazebo scene capture failed: ' + capture.stderr[-300:])
                capture_path = capture.stdout.strip().splitlines()[-1]
                failure = 'benchmark'
                with (folder / 'bench.log').open('w') as output:
                    bench = subprocess.Popen([str(args.vision_python), str(ROOT / 'tools/sim_perception_bench/bench.py'),
                        '--domain', '77', '--frames', '30', '--models-dir', str(ROOT / 'src/base_athome_perception/models'),
                        '--out', str(folder / 'bench')], cwd=ROOT, env=env, stdout=output,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        bench_code = wait_for_hri(bench, sim, timeout_s=150)
                    finally:
                        stop(bench)
                if bench_code:
                    raise RuntimeError('Perception benchmark failed')
                failure = 'hri'
                with (folder / 'hri.log').open('w') as output:
                    hri = subprocess.Popen([str(args.vision_python), '-u', str(ROOT / 'simulation/scripts/receptionist_real_perception.py')],
                        cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    code = wait_for_hri(hri, sim, timeout_s=660)
                log = (folder / 'hri.log').read_text(errors='replace')
                samples = metrics(log)
                result = next((json.loads(line.split('RENDERED_EXPERIMENT_RESULT ', 1)[1])
                    for line in log.splitlines() if line.startswith('RENDERED_EXPERIMENT_RESULT ')), {})
                scene = (folder / 'sim.log').read_text(errors='replace')
                if code == 0 and result.get('completed') and result.get('unique_seats') and 'RuntimeError:' not in scene:
                    outcome, failure = 'PASS', 'none'
                else:
                    failure = 'seating_or_actor_failure' if result.get('completed') else 'hri_incomplete'
            except (RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                (folder / 'failure.txt').write_text(str(exc) + '\n')
            finally:
                if hri is not None:
                    stop(hri)
                if sim is not None:
                    stop(sim)
            elapsed = time.monotonic() - start
            log_text = (folder / 'hri.log').read_text(errors='replace') if (folder / 'hri.log').is_file() else ''
            sim_text = (folder / 'sim.log').read_text(errors='replace') if (folder / 'sim.log').is_file() else ''
            eval_result = evaluate_receptionist_run(
                log_text=log_text,
                sim_log=sim_text,
                duration_s=elapsed,
                experiment_result=result if 'result' in locals() else {},
            )
            metrics_payload = {
                'total_score': eval_result['total_score'],
                'estimated_score': eval_result.get('estimated_score', eval_result['total_score']),
                'is_estimated': True,
                'score_status': 'estimated',
                'max_possible': eval_result['max_possible'],
                'score_breakdown': eval_result['items'],
                'penalties': eval_result['penalties'],
                'summary_note': eval_result['summary_note'],
                'perception_samples': samples,
                'limits': eval_result.get('limits', ''),
            }
            (folder / 'metrics.json').write_text(json.dumps(metrics_payload, indent=2) + '\n')
            row = dict(mode=mode, iteration=iteration, sha=sha, outcome=outcome, failure=failure,
                       score=eval_result['total_score'], estimated=True, score_status='estimated',
                       max_possible=eval_result['max_possible'],
                       score_summary=eval_result['summary_note'],
                       elapsed_s=round(elapsed, 2), directory=str(folder), world=args.world,
                       scene_capture=capture_path,
                       limits='Three rendered people (two guests plus host); translated mesh paths, speech/TTS/navigation/bag motion are simulation fixtures; no collision certification; score is estimated')
            (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            with args.ledger.open('a') as ledger:
                ledger.write('\t'.join([datetime.datetime.now().astimezone().isoformat(), 'codex-receptionist',
                    f'{args.tag}-{mode}', sha, str(iteration), outcome, str(eval_result['total_score']), failure,
                    str(folder / 'hri.log'), f"elapsed={elapsed:.2f}s; world={args.world}; rendered_humans=3; stage={failure}; rates={rate_note(samples) if hri is not None else 'unavailable'}; score={eval_result['total_score']}/{eval_result['max_possible']} ({eval_result['summary_note']}); no official physical score claimed"]) + '\n')
            print(json.dumps(row), flush=True)
    (args.out / f'{args.tag}-summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    return 0 if all(row['outcome'] == 'PASS' for row in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
