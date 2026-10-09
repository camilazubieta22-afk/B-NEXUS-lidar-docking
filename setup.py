from glob import glob
from setuptools import setup

package_name = 'dock_lidar'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Camila Zuco',
    maintainer_email='proyectos00088@gmail.com',
    description='Docking del Create 3 solo con LiDAR',
    license='MIT',
    entry_points={
        'console_scripts': [
            'dock_node = dock_lidar.dock_node:main',
        ],
    },
)
