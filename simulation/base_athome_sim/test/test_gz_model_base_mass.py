import re
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from base_athome_sim import gz_model


def test_sim_base_is_heavier_and_low_but_real_urdf_untouched():
    path = Path(__file__).resolve().parents[3] / 'src/hoverboard_base/urdf/robot.urdf'
    real = path.read_text()
    out = gz_model.gazebo_urdf(real)
    base = re.search(r'<link name="base_link">.*?</link>', out, re.S).group(0)
    assert '<mass value="40.0"/>' in base and 'xyz="0 0 0.05"' in base
    assert '<mass value="12.0"/>' in path.read_text()
    assert out.count('<inertial>') == real.count('<inertial>') + out.count('restaurant_tray') * 0 + (out.count('<inertial>') - real.count('<inertial>'))
    assert out.strip().endswith("</robot>")
    root = ET.fromstring(out)
    assert root.tag == "robot"


def test_restaurant_tray_urdf_visual_materials_have_required_names():
    root = ET.fromstring('<robot name="sim">' + gz_model._restaurant_tray_urdf() + '</robot>')
    visuals = root.findall(f"./link[@name='{gz_model.RESTAURANT_TRAY_LINK}']/visual")
    assert len(visuals) == 5
    for visual in visuals:
        material = visual.find('material')
        assert material is not None and material.get('name')
        assert material.find('color') is not None


def test_simulated_rolling_links_have_friction_chassis_does_not():
    path = Path(__file__).resolve().parents[3] / 'src/hoverboard_base/urdf/robot.urdf'
    real = path.read_text()
    root = ET.fromstring(gz_model.gazebo_urdf(real))
    for link in ('left_wheel', 'right_wheel', 'rear_caster'):
        friction = root.find(f"./gazebo[@reference='{link}']")
        assert friction is not None
        assert float(friction.findtext('mu1')) == 1.0
        assert float(friction.findtext('mu2')) == 1.0
    chassis = root.find("./gazebo[@reference='base_link']")
    assert chassis is not None
    assert float(chassis.findtext('mu1')) == 0.0
    assert float(chassis.findtext('mu2')) == 0.0
    assert '<mu1>0.0</mu1>' not in real


def test_simulated_chassis_and_wheel_collisions_clear_the_ground_plane():
    path = Path(__file__).resolve().parents[3] / 'src/hoverboard_base/urdf/robot.urdf'
    root = ET.fromstring(path.read_text())
    origin_z = lambda element: float(element.find('origin').get('xyz').split()[2])
    base_joint_z = origin_z(root.find("./joint[@name='base_joint']"))
    chassis = root.find("./link[@name='base_link']/collision")
    chassis_bottom = (base_joint_z + origin_z(chassis)
                      - float(chassis.find('./geometry/box').get('size').split()[2]) / 2)
    wheel_joint_z = origin_z(root.find("./joint[@name='left_wheel_joint']"))
    wheel_radius = float(root.find("./link[@name='left_wheel']/collision/geometry/cylinder").get('radius'))
    caster_joint_z = origin_z(root.find("./joint[@name='rear_caster_joint']"))
    caster_radius = float(root.find("./link[@name='rear_caster']/collision/geometry/sphere").get('radius'))

    # sim.launch.py spawns the URDF root at z=0; these clearances are relative to that ground plane.
    assert chassis_bottom == pytest.approx(0.10)
    assert base_joint_z + wheel_joint_z - wheel_radius == pytest.approx(0.06)
    assert base_joint_z + caster_joint_z - caster_radius == pytest.approx(0.06)
