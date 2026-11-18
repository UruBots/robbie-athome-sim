"""Textured Fuel visitors for the opt-in Receptionist camera experiment.

Geometry / textures: Wan Yi Seow, Open Robotics, CC BY 4.0.
Sitting garment materials are changed to match the standing visitors; UV
textures, faces, skin and geometry remain unmodified.
https://fuel.gazebosim.org/1.0/OpenRobotics/models/{FemaleVisitor,
FemaleVisitorSit,MaleVisitorOnPhone,MaleVisitorSit}/2
https://creativecommons.org/licenses/by/4.0/

These are textured static meshes translated along paths (no skeletal gait).
Standing/sitting assets are paired by gender; sitting garment colors are matched.
The mesh variants still differ in garment cut and accessories.
No pixels or detections are synthesized. Downloads are explicit, never on launch.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

from . import gz_model

ASSETS = ('FemaleVisitor', 'FemaleVisitorSit', 'MaleVisitorOnPhone', 'MaleVisitorSit')
MESHES = {'FemaleVisitor': 'FemaleVisitor.obj', 'FemaleVisitorSit': 'FemaleVisitorSit_receptionist.obj',
          'MaleVisitorOnPhone': 'MaleVisitorStatic.obj', 'MaleVisitorSit': 'MaleVisitorSit_receptionist.obj'}


def enabled():
    return os.environ.get('ATHOME_SIM') == '1' and os.environ.get('ATHOME_SIM_REAL_PERCEPTION') == '1'


def asset_root():
    return Path(os.environ.get('ATHOME_SIM_HUMANS_DIR',
        Path(__file__).resolve().parents[1] / 'models/humans'))


def garment_material(asset, z, rgb, material):
    """Select only seated garment polygons; never recolor face/skin textures."""
    if 'phone' in material.lower() or 'handbag' in material.lower():
        return False
    r, g, b = rgb
    if asset == 'MaleVisitorSit':
        return 0.65 < z < 1.18 and max(rgb) - min(rgb) < 0.25 * max(max(rgb), 1)
    return asset == 'FemaleVisitorSit' and 0.35 < z < 1.15 and g > r * 1.08 and g > b * 1.02


def match_sitting_clothes(root):
    """Generate OBJ material variants, preserving the original UV image files.

    The supplied sitting models use different shirt/dress colors from their
    standing partners. Per-polygon material assignment keeps guest descriptions
    consistent without painting photographic faces or replacing camera pixels.
    """
    from PIL import Image
    for asset, color in (('MaleVisitorSit', '0.35 0.20 0.55'),
                         ('FemaleVisitorSit', '0.20 0.055 0.055')):
        directory = Path(root) / asset / 'meshes'
        source = directory / f'{asset}.obj'
        texture = Image.open(directory / f'{asset}.png').convert('RGB')
        vertices, uv = [], []
        material = 'body'
        output = []
        mtl_name = f'{asset}_receptionist.mtl'
        for line in source.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == 'mtllib':
                output.append('mtllib ' + mtl_name)
                original = (directory / parts[1]).read_text()
                (directory / mtl_name).write_text(original + '\nnewmtl receptionist_garment\nKa ' + color +
                    '\nKd ' + color + '\nKs 0.05 0.05 0.05\nNs 10\nd 1\nillum 2\n')
                continue
            if parts and parts[0] == 'v':
                vertices.append(tuple(map(float, parts[1:4])))
            elif parts and parts[0] == 'vt':
                uv.append(tuple(map(float, parts[1:3])))
            elif parts and parts[0] == 'usemtl':
                material = parts[1]
            elif parts and parts[0] == 'f':
                corners = [p.split('/') for p in parts[1:]]
                z = sum(vertices[int(p[0]) - 1][2] for p in corners) / len(corners)
                coords = [uv[int(p[1]) - 1] for p in corners]
                u, v = (sum(p[i] for p in coords) / len(coords) for i in range(2))
                x = round(min(1, max(0, u)) * (texture.width - 1))
                y = round((1 - min(1, max(0, v))) * (texture.height - 1))
                selected = garment_material(asset, z, texture.getpixel((x, y)), material)
                output.append('usemtl ' + ('receptionist_garment' if selected else material))
            output.append(line)
        (directory / MESHES[asset]).write_text('\n'.join(output) + '\n')
        texture.close()


def fetch_assets(root):
    """Download pinned public assets, retaining attribution and license metadata."""
    root = Path(root)
    for name in ASSETS:
        url = f'https://fuel.gazebosim.org/1.0/OpenRobotics/models/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            meta = json.load(response)
        if meta.get('license_name') != 'Creative Commons Attribution 4.0 International':
            raise ValueError(f'Unexpected license for {name}')
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f'{url}/2/{name}.zip', timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                dest = (target / member.filename).resolve()
                if not dest.is_relative_to(target.resolve()):
                    raise ValueError('Unsafe archive path')
            archive.extractall(target)
        (target / 'fuel-license.json').write_text(json.dumps({
            'author': 'Wan Yi Seow / Open Robotics', 'source': url + '/2',
            'license': meta['license_name'], 'license_url': meta['license_url'],
            'changes': 'Generated sitting OBJ garment material variants; original geometry and UV textures unmodified.'}, indent=2) + '\n')
    match_sitting_clothes(root)


def visitor_sdf(name, xy, asset, yaw=-math.pi / 2, root=None):
    """Visual-only model; mesh feet are in metres at z≈0, facing local -Y."""
    mesh = (Path(root or asset_root()) / asset / 'meshes' / MESHES[asset]).resolve()
    if not mesh.is_file():
        raise FileNotFoundError(f'{mesh}: run python -m base_athome_sim.receptionist_humans --fetch')
    sdf = ET.Element('sdf', version='1.9')
    model = ET.SubElement(sdf, 'model', name=name)
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = f'{xy[0]} {xy[1]} 0 0 0 {yaw}'
    link = ET.SubElement(model, 'link', name='body')
    visual = ET.SubElement(link, 'visual', name='visitor')
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(ET.SubElement(geometry, 'mesh'), 'uri').text = mesh.as_uri()
    return ET.tostring(sdf, encoding='unicode')


def path_samples(start, end, speed=0.65, period=0.15):
    """Bounded translation with the final sample exactly on the measured seat."""
    steps = max(1, math.ceil(math.dist(start, end) / (speed * period)))
    samples = [tuple(a + (b - a) * step / steps for a, b in zip(start, end))
               for step in range(1, steps + 1)]
    samples[-1] = tuple(end)
    return samples


def trail_position(history, gap=0.72):
    """Follow distance along the walked polyline, preserving clearance at corners."""
    remaining = gap
    for a, b in zip(reversed(history[:-1]), reversed(history[1:])):
        length = math.dist(a, b)
        if length >= remaining and length > 0:
            return tuple(b[i] + (a[i] - b[i]) * remaining / length for i in range(2))
        remaining -= length
    return None


class ReceptionistHumans:
    """Scene director; ground truth controls props but never supplies perception."""
    def __init__(self, node, world):
        if os.environ.get('ROS_DOMAIN_ID') not in ('42', '77') or not os.environ.get('GZ_PARTITION'):
            raise RuntimeError('Human experiment requires isolated ROS domain 42/77 and GZ_PARTITION')
        self.node, self.world = node, world
        # Inside the existing ±60 degree living-room scan, between guest chairs.
        node.receptionist_host_xy = [3.5, 0.0]
        if world == 'receptionist_humans':
            node.receptionist_drop_xy = [4.35, 0.55]
        self.lock = threading.RLock()
        self.positions = {}
        self.ready = threading.Event()
        self.arrived = set()
        self.attended = set()
        self.standing = ('MaleVisitorOnPhone', 'FemaleVisitor')
        self.sitting = ('MaleVisitorSit', 'FemaleVisitorSit')
        self._sway_stop = threading.Event()
        self._sway_thread = None
        self._swaying_guest = None
        self.door_sway_amp = 0.25
        self.door_sway_period = 8.0
        # Validate every asset before creating the first scene entity.
        for asset in ASSETS:
            visitor_sdf('check', (0, 0), asset)

    def command(self, cmd):
        cmd = list(cmd)
        if '--timeout' in cmd:
            cmd[cmd.index('--timeout') + 1] = '5000'
        # set_pose is idempotent; short-lived CLI clients can miss a response
        # while two Gazebo renderers share CPU. Never retry create/remove blindly.
        for attempt in range(3 if cmd[3].endswith('/set_pose') else 1):
            result = self.node._gz_run(cmd)
            if 'true' in result.lower():
                return
            if attempt < 2 and cmd[3].endswith('/set_pose'):
                time.sleep(0.2)
        # A queued create/remove can succeed after its CLI client timed out.
        # Verify its actual scene postcondition instead of lying about failure
        # or replaying a non-idempotent command against a live entity.
        service = cmd[3]
        if service.endswith(('/create', '/remove')):
            request = cmd[cmd.index('--req') + 1]
            if service.endswith('/create'):
                model = ET.fromstring(json.loads(request.removeprefix('sdf: '))).find('model')
                name = model.attrib['name']
            else:
                name = request.split('"')[1]
            scene = self.node._gz_run([
                'gz', 'service', '-s', f'/world/{self.world}/scene/info',
                '--reqtype', 'gz.msgs.Empty', '--reptype', 'gz.msgs.Scene',
                '--timeout', '5000', '--req', ''])
            present = f'name: "{name}"' in scene
            if scene.strip() and present == service.endswith('/create'):
                return
        raise RuntimeError(f'Gazebo scene command failed: {cmd[:4]}: {result}')

    def replace(self, name, xy, asset, yaw=-math.pi / 2):
        if name in self.positions:
            self.command(gz_model.remove_command(self.world, name))
        self.command(gz_model.create_command(self.world, visitor_sdf(name, xy, asset, yaw)))
        self.positions[name] = tuple(xy)

    def spawn(self):
        with self.lock:
            self.replace('recep_host', self.node.receptionist_host_xy, 'MaleVisitorSit')
            # A sofa at the host's existing scripted position, with no extra collisions.
            if self.world != 'receptionist_humans':
                for name, xyz, size in (
                    ('recep_sofa_seat', (3.5, 0.0, 0.225), (0.6, 1.2, 0.45)),
                    ('recep_sofa_back', (3.77, 0.0, 0.65), (0.1, 1.2, 0.4))):
                    self.command(gz_model.create_command(self.world,
                        gz_model._static_box(name, xyz, size, (0.35, 0.22, 0.12))))
                self.command(gz_model.create_command(self.world,
                    gz_model.receptionist_drop_sdf(self.node.receptionist_drop_xy)))
            self.ready.set()
        self.arrive(1)

    def walk(self, name, end):
        start = self.positions[name]
        yaw = math.atan2(end[1] - start[1], end[0] - start[0]) + math.pi / 2
        for xy in path_samples(start, end, period=0.15):
            self.command(gz_model.set_pose_command(self.world, name, (*xy, 0), yaw=yaw))
            self.positions[name] = xy
            time.sleep(0.15)

    def _start_door_sway(self, index):
        self._stop_door_sway()
        self._sway_stop.clear()
        self._swaying_guest = index
        name = f'recep_guest_{index}'

        def sway_worker():
            t0 = time.monotonic()
            while not self._sway_stop.wait(0.15):
                with self.lock:
                    if self._sway_stop.is_set():
                        break
                    elapsed = time.monotonic() - t0
                    y = self.door_sway_amp * math.sin(2.0 * math.pi * elapsed / self.door_sway_period)
                    x = -1.35
                    self.positions[name] = (x, y)
                    self.node.person_xy = [x, y]
                    try:
                        self.command(gz_model.set_pose_command(self.world, name, (x, y, 0), yaw=math.pi / 2))
                    except Exception:
                        pass

        self._sway_thread = threading.Thread(target=sway_worker, daemon=True)
        self._sway_thread.start()

    def _stop_door_sway(self):
        thread = getattr(self, '_sway_thread', None)
        if thread is not None and thread.is_alive():
            stop = getattr(self, '_sway_stop', None)
            if stop is not None:
                stop.set()
            thread.join(timeout=1.0)
        self._sway_thread = None
        self._swaying_guest = None

    def arrive(self, index):
        if not self.ready.wait(30):
            raise RuntimeError('Receptionist scene not ready')
        with self.lock:
            if index in self.arrived:
                return
            name = f'recep_guest_{index}'
            self.replace(name, (-3.8, 0.0), self.standing[(index - 1) % 2], math.pi / 2)
            self.walk(name, (-1.35, 0.0))
            self.arrived.add(index)
            self.node.person_xy = [-1.35, 0.0]
            self.node.get_logger().info(f'REAL_HUMAN_STAGE guest={index} stage=at_door')
        self._start_door_sway(index)

    def seat(self, xy, index):
        self._stop_door_sway()
        with self.lock:
            name = f'recep_guest_{index}'
            # A welcomed guest is no longer calling for service. In particular,
            # MaleVisitorOnPhone's raised hand disappears when its seated mesh
            # replaces it; repeated speech callbacks cannot restart that guest.
            if index in self.attended:
                self.node.get_logger().info(
                    f'REAL_HUMAN_STAGE guest={index} stage=already_attended')
                return
            # Go through the clear middle aisle, avoiding the demonstration table.
            aisle = 1.05 if xy[1] > 0 else -1.05
            self.walk(name, (-0.4, aisle))
            self.walk(name, (2.7, aisle))
            self.walk(name, xy)
            base = self.node.base
            yaw = math.atan2(base.y - xy[1], base.x - xy[0]) + math.pi / 2
            self.replace(name, xy, self.sitting[(index - 1) % 2], yaw=yaw)
            self.node._record_seated_guest(xy, index)
            self.node._publish_receptionist_state()
            self.attended.add(index)
            self.node.get_logger().info(
                f'REAL_HUMAN_STAGE guest={index} stage=attended_arm_lowered_seated')

    def follow_host(self):
        node = self.node
        with self.lock:
            node.receptionist_bag_state = 'following'
            node.receptionist_follow_active = True
            node._publish_receptionist_task_state()
            if hasattr(node, 'pub_voice'):
                node.pub_voice.publish(String(data='Follow me'))
            self.replace('recep_host', node.receptionist_host_xy, 'MaleVisitorOnPhone')
            # Let the host leave the sofa before the robot starts following.
            self.walk('recep_host', (2.6, 1.0))
            for end in ((2.6, 2.45), (4.35, 2.45), (4.35, 0.55), tuple(node.receptionist_drop_xy)):
                start = self.positions['recep_host']
                yaw = math.atan2(end[1] - start[1], end[0] - start[0])
                for xy in path_samples(start, end, period=0.15):
                    self.command(gz_model.set_pose_command(self.world, 'recep_host', (*xy, 0), yaw=yaw + math.pi / 2))
                    self.positions['recep_host'] = xy
                    node.receptionist_host_xy = list(xy)
                    node.receptionist_host_yaw = yaw
                    time.sleep(0.15)
            node.receptionist_host_path_done = True
            node.receptionist_follow_active = False
            node.receptionist_bag_state = 'host_at_drop'
            node._publish_receptionist_task_state()
            if hasattr(node, 'pub_voice'):
                node.pub_voice.publish(String(data='You can drop the bag here.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', required=True)
    parser.add_argument('--directory', type=Path, default=asset_root())
    args = parser.parse_args()
    fetch_assets(args.directory)
