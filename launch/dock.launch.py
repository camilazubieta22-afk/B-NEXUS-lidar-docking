import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('dock_lidar'), 'config', 'params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true en Gazebo, false en el robot real'),
        Node(
            package='dock_lidar',
            executable='dock_node',
            name='dock_lidar',
            output='screen',
            parameters=[params, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
