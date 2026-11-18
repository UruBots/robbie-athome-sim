#!/usr/bin/env python3
"""Run real Receptionist vision against an isolated Gazebo scene.

Speech inputs, TTS and navigation are simulation fixtures. Face/pose/seat/
clothing decisions and all HRI timeouts remain production code. Start sim.launch
with gazebo:=true auto_guest:=true and ATHOME_SIM_REAL_PERCEPTION=1 first.
No audio, screenshots, detections or biometric outputs are injected.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def simulated_answer(prompt, guest_index, seated_guests):
    """Only a guest who has reached the seat can confirm sitting by voice."""
    if 'answer yes or no' in prompt.lower():
        return 'yes' if any(int(g.get('guest_index', 0)) == guest_index
                            for g in seated_guests) else None
    return ('My name is John and my favorite drink is cider' if guest_index == 1
            else 'My name is Mary and my favorite drink is red wine')


def main():
    if (os.environ.get('ROS_DOMAIN_ID') != '77' or os.environ.get('GZ_PARTITION') != 'recep_humanos'
            or os.environ.get('ATHOME_SIM') != '1'
            or not (os.environ.get('ATHOME_SIM_REAL_PERCEPTION') == '1'
                    or os.environ.get('ATHOME_SIM_RENDERED_HUMANS') == '1')):
        raise SystemExit('Requires ROS_DOMAIN_ID=77 GZ_PARTITION=recep_humanos ATHOME_SIM=1 ATHOME_SIM_REAL_PERCEPTION=1')
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault('ATHOME_SIM_SEMANTIC_MAP', str(root / 'simulation/base_athome_sim/config/receptionist_semantic_map.yaml'))
    os.environ.setdefault('ATHOME_FACES_DB', str(Path.home() / '.cache/base_athome/receptionist_rendered_faces'))
    spec = importlib.util.spec_from_file_location('receptionist_rendered',
        root / 'src/base_athome_perception/scripts/hri_receptionist_challenge.py')
    hri = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hri)
    from base_athome_sim import gz_model
    world = Path(os.environ.get('ATHOME_SIM_WORLD', 'casa_simple.sdf')).stem
    # Bound CPU contention with the other isolated simulation on this host.
    import torch
    torch.set_num_threads(2)

    def subtitle(self, text, blocking=False, on_done=None):
        print('SIM_SUBTITLE:', text, flush=True)
        if on_done:
            on_done()
    hri.RoboCupSpeaker.speak = subtitle
    hri.RoboCupSpeaker.play_beep = lambda *a, **kw: None
    outcome = {'completed': False, 'seat_failures': 0}
    init = hri.HriReceptionistChallengeNode.__init__

    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.rendered_speech = self.create_publisher(hri.String, '/sim/speech', 10)
        self.rendered_guests = self.create_publisher(hri.String, '/sim/guest', 10)
        self.rendered_auto = self.create_publisher(hri.Bool, '/sim/auto_guest', 10)
        self.rendered_auto.publish(hri.Bool(data=False))
        self.rendered_guest_number = 0
        self.rendered_started = time.monotonic()
        self.rendered_success = False
        def respond(msg):
            prompt = json.loads(msg.data).get('prompt', '').lower()
            guests = self.sim_receptionist_state.get('seated_guests', [])
            line = simulated_answer(prompt, self.guest_index, guests)
            if line is not None:
                threading.Timer(0.4, lambda: self.rendered_speech.publish(hri.String(data=line))).start()
        self.create_subscription(hri.String, '/voice/listen_request', respond, 10)
    hri.HriReceptionistChallengeNode.__init__ = initialize
    original_nav = hri.HriReceptionistChallengeNode.navigate_to_waypoint

    def navigate(self, waypoint):
        # Kinematic mock, not Nav2 validation: move only our isolated Gazebo base.
        xyz, yaw = ((0.0, 0.0, 0.0), math.pi) if waypoint == self.door_wp else ((1.0, 0.0, 0.0), 0.0)
        result = subprocess.run(gz_model.set_pose_command(world, gz_model.MODEL, xyz, yaw=yaw),
                                capture_output=True, text=True, timeout=10)
        if result.returncode or 'true' not in result.stdout:
            raise RuntimeError('Isolated Gazebo viewpoint failed: ' + result.stdout)
        return original_nav(self, waypoint)
    hri.HriReceptionistChallengeNode.navigate_to_waypoint = navigate
    original_step = hri.HriReceptionistChallengeNode.step

    def step(self):
        if self.state == self.STATE_NAV_TO_DOOR and self.rendered_guest_number != self.guest_index:
            self.rendered_guests.publish(hri.String(data='door' if self.guest_index == 1 else 'next'))
            self.rendered_guest_number = self.guest_index
        previous = self.state
        original_step(self)
        if (previous == self.STATE_WAIT_SEATED and self.state != previous
                and getattr(self, 'seat_confirmation_method', None) not in ('vision', 'voice')):
            outcome['seat_failures'] += 1
        if self.state == self.STATE_DONE:
            outcome['completed'] = True
            outcome['seats'] = [g.get('seat') for g in self.registered_guests]
            outcome['unique_seats'] = len(set(outcome['seats'])) == 2
            profiles = [self.recognizer.profiles[g['person_id']] for g in self.registered_guests]
            outcome['real_face_samples'] = [p.samples_count if not p.attributes.get('synthetic') else 0 for p in profiles]
            self.rendered_success = True
            print('REAL_PERCEPTION_TASK_DONE', flush=True)
            self.request_shutdown = True
        if time.monotonic() - self.rendered_started > float(os.environ.get('ATHOME_SIM_RUN_TIMEOUT', '600')):
            print('REAL_PERCEPTION_TASK_TIMEOUT state=' + str(self.state), flush=True)
            self.request_shutdown = True
    hri.HriReceptionistChallengeNode.step = step
    sys.argv = ['receptionist_rendered', '--sim-nav', '--no-wait', '--no-gui']
    hri.main()
    print('RENDERED_EXPERIMENT_RESULT ' + json.dumps(outcome), flush=True)
    if (not outcome['completed'] or outcome['seat_failures'] or not outcome.get('unique_seats')
            or any(count < 4 for count in outcome.get('real_face_samples', [0]))):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
