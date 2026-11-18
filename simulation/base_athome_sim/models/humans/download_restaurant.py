#!/usr/bin/env python3
"""Fetch/vendor licensed Fuel visitors and bake their rig into static poses.

Source https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male%20visitor/2
License https://creativecommons.org/publicdomain/zero/1.0/
Additional CC-BY-4.0 models: Luca/FemaleVisitorWalk/1 and Luca/NurseFemaleWalk/1;
their source files and image textures are vendored under their respective directories.
The skin weights, geometry and textures are retained; only bone poses change.
"""
from pathlib import Path
import copy
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
URL = 'https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male%20visitor/2'
NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
ET.register_namespace('', NS['c'])


def rotation(axis, angle):
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    matrix = np.eye(4)
    if axis == 'x':
        matrix[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    else:
        matrix[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return matrix


def bake_pose(source, destination, seated=False, raised='right'):
    """Linear blend skinning with the asset's actual bone weights (metres/Z-up)."""
    tree = ET.parse(source)
    root = tree.getroot()
    skin = root.find('.//c:skin', NS)
    sources = {s.get('id'): s for s in skin.findall('c:source', NS)}
    joint_inputs = {e.get('semantic'): e.get('source')[1:] for e in skin.findall('c:joints/c:input', NS)}
    names = sources[joint_inputs['JOINT']].find('c:Name_array', NS).text.split()
    inverse_bind = np.fromstring(sources[joint_inputs['INV_BIND_MATRIX']].find('c:float_array', NS).text, sep=' ').reshape(-1, 4, 4)
    rest = {name: np.linalg.inv(m) for name, m in zip(names, inverse_bind)}
    parents = {}
    scene = root.find('.//c:visual_scene', NS)
    def visit(node, parent=None):
        name = node.get('sid')
        if name in rest:
            parents[name] = parent
            parent = name
        for child in node.findall('c:node', NS):
            visit(child, parent)
    for node in scene.findall('c:node', NS):
        visit(node)
    posed = {}
    def pose(name):
        if name in posed:
            return posed[name]
        parent = parents[name]
        world = rest[name].copy() if parent is None else pose(parent) @ np.linalg.inv(rest[parent]) @ rest[name]
        if name.startswith('UpperArm_'):
            left = name.endswith('_L')
            up = raised == ('left' if left else 'right')
            rot = rotation('x', (65 if left else -65) if up else (-85 if left else 85))
            pivot = world[:3, 3].copy()
            world = rot @ world
            world[:3, 3] = pivot
        if seated and name.startswith(('UpperLeg_', 'LowerLeg_')):
            rot = rotation('y', -90 if name.startswith('UpperLeg_') else 90)
            pivot = world[:3, 3].copy()
            world = rot @ world
            world[:3, 3] = pivot
        posed[name] = world
        return world
    matrices = np.array([pose(n) @ inv for n, inv in zip(names, inverse_bind)])
    weights_node = skin.find('c:vertex_weights', NS)
    inputs = {e.get('semantic'): (int(e.get('offset')), e.get('source')[1:]) for e in weights_node.findall('c:input', NS)}
    stride = max(v[0] for v in inputs.values()) + 1
    weight_values = np.fromstring(sources[inputs['WEIGHT'][1]].find('c:float_array', NS).text, sep=' ')
    counts = np.fromstring(weights_node.find('c:vcount', NS).text, sep=' ', dtype=int)
    indices = np.fromstring(weights_node.find('c:v', NS).text, sep=' ', dtype=int).reshape(-1, stride)
    blended = np.zeros((len(counts), 4, 4))
    cursor = 0
    for i, count in enumerate(counts):
        for row in indices[cursor:cursor + count]:
            blended[i] += weight_values[row[inputs['WEIGHT'][0]]] * matrices[row[inputs['JOINT'][0]]]
        cursor += count
    geometry = root.find('.//c:geometry', NS)
    mesh = geometry.find('c:mesh', NS)
    position_id = mesh.find('c:vertices/c:input', NS).get('source')[1:]
    positions = next(s for s in mesh.findall('c:source', NS) if s.get('id') == position_id).find('c:float_array', NS)
    vertices = np.fromstring(positions.text, sep=' ').reshape(-1, 3)
    bind_shape = np.fromstring(skin.find('c:bind_shape_matrix', NS).text, sep=' ').reshape(4, 4)
    points = np.c_[vertices, np.ones(len(vertices))] @ bind_shape.T
    points = np.einsum('nij,nj->ni', blended, points)[:, :3]
    if seated:
        points[:, 2] -= .44
    positions.text = ' '.join(f'{v:.7g}' for v in points.ravel())
    # Normal indices can be split at UV seams; map each to its vertex's blended rig.
    # Materials can share a COLLADA normal source across primitives; transform each
    # source once to avoid rotating its normals twice for multi-texture humans.
    normal_maps = {}
    for primitive in list(mesh):
        if primitive.tag.split('}')[-1] not in ('triangles', 'polylist'):
            continue
        inp = {e.get('semantic'): (int(e.get('offset')), e.get('source')[1:]) for e in primitive.findall('c:input', NS)}
        if 'NORMAL' not in inp:
            continue
        normal_id = inp['NORMAL'][1]
        normals = next(s for s in mesh.findall('c:source', NS) if s.get('id') == normal_id).find('c:float_array', NS)
        values = np.fromstring(normals.text, sep=' ').reshape(-1, 3)
        index = np.fromstring(primitive.find('c:p', NS).text, sep=' ', dtype=int).reshape(-1, max(v[0] for v in inp.values()) + 1)
        mapping = normal_maps.setdefault(normal_id, np.zeros(len(values), dtype=int))
        mapping[index[:, inp['NORMAL'][0]]] = index[:, inp['VERTEX'][0]]
    for normal_id, mapping in normal_maps.items():
        normals = next(s for s in mesh.findall('c:source', NS) if s.get('id') == normal_id).find('c:float_array', NS)
        values = np.fromstring(normals.text, sep=' ').reshape(-1, 3)
        values = np.einsum('nij,nj->ni', blended[mapping, :3, :3], values)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        normals.text = ' '.join(f'{v:.7g}' for v in values.ravel())
    material = copy.deepcopy(scene.find('.//c:instance_controller/c:bind_material', NS))
    scene.clear()
    scene.set('id', 'Scene')
    node = ET.SubElement(scene, '{'+NS['c']+'}node', {'id': 'StaticHuman'})
    instance = ET.SubElement(node, '{'+NS['c']+'}instance_geometry', {'url': '#'+geometry.get('id')})
    instance.append(material)
    for tag in ('library_controllers', 'library_animations'):
        element = root.find('c:'+tag, NS)
        if element is not None:
            root.remove(element)
    tree.write(destination, encoding='utf-8', xml_declaration=True)
    return points


def main():
    target = ROOT / 'male_visitor'
    source = target / 'meshes/MaleVisitorWalk.dae'
    if not source.exists():
        config = ROOT / 'cache.config'
        config.write_text('---\ncache:\n  path: ' + str(ROOT / 'cache') + '\nservers:\n  - url: https://fuel.gazebosim.org\n')
        try:
            subprocess.run(['gz', 'fuel', 'download', '-c', str(config), '-u', URL], check=True)
        finally:
            config.unlink(missing_ok=True)
        downloaded = list((ROOT / 'cache').rglob('MaleVisitorWalk.dae'))
        if len(downloaded) != 1:
            raise RuntimeError('Expected exactly one pinned Male visitor mesh')
        shutil.copytree(downloaded[0].parents[1], target, dirs_exist_ok=True)
    assets = [
        (target, source, 'CC0-1.0', 'OpenRobotics / Wan Yi Seow', URL,
         'https://creativecommons.org/publicdomain/zero/1.0/',
         [('seated_right', True, 'right'), ('seated_left', True, 'left'),
          ('seated_down', True, None), ('standing_right', False, 'right'), ('standing_down', False, None)]),
        (ROOT / 'female_visitor', ROOT / 'female_visitor/meshes/FemaleVisitorWalk.dae',
         'CC-BY-4.0', 'Luca (uploader) / Wan Yi Seow (author)',
         'https://fuel.gazebosim.org/1.0/Luca/models/FemaleVisitorWalk/1',
         'https://creativecommons.org/licenses/by/4.0/',
         [('seated_left', True, 'left'), ('seated_down', True, None)]),
        (ROOT / 'nurse_female', ROOT / 'nurse_female/meshes/NurseFemaleWalk.dae',
         'CC-BY-4.0', 'Luca (uploader) / Wan Yi Seow (author)',
         'https://fuel.gazebosim.org/1.0/Luca/models/NurseFemaleWalk/1',
         'https://creativecommons.org/licenses/by/4.0/',
         [('standing_right', False, 'right'), ('standing_down', False, None)]),
    ]
    for directory, source_mesh, license_name, author, url, license_url, poses in assets:
        if not source_mesh.is_file():
            raise RuntimeError(f'Missing vendored licensed source mesh: {source_mesh}')
        for name, seated, hand in poses:
            bake_pose(source_mesh, directory / f'meshes/{name}.dae', seated, hand)
        (directory / 'SOURCE.json').write_text(json.dumps(dict(url=url, version=1, license=license_name,
            license_url=license_url, attribution=author,
            modifications='Static bone poses baked using original skin weights'), indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
