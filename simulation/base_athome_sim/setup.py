from glob import glob
from pathlib import Path
from setuptools import setup

package_name = 'base_athome_sim'
# Every model folder under models/humans is installed (tracked ones and the Fuel meshes fetched by
# receptionist_humans --fetch); caches are not.
IGNORED_MODEL_DIRS = {'__pycache__', 'cache', 'standing_person'}
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.xml')),
        ('share/' + package_name + '/models/humans', ['models/humans/download_restaurant.py']),
    ] + [(str(Path('share') / package_name / asset.parent), [str(asset)])
         for model in sorted(Path('models/humans').iterdir())
         if model.is_dir() and model.name not in IGNORED_MODEL_DIRS
         for asset in sorted(model.rglob('*')) if asset.is_file() and '__pycache__' not in asset.parts],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UruBots',
    maintainer_email='urubots@users.noreply.github.com',
    description='Simulacion sencilla de Base@Home',
    license='Apache-2.0',
    entry_points={'console_scripts': ['sim_robot = base_athome_sim.sim_robot:main']},
)
