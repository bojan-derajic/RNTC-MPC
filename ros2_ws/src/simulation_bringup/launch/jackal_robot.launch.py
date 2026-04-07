import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    TimerAction,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    Command,
    PythonExpression,
)
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            name="world_model",
            default_value="empty",
            description="Name of the world model",
        ),
        DeclareLaunchArgument(
            name="robot_model",
            default_value="jackal_robot",
            description="Name of the robot model",
        ),
        DeclareLaunchArgument(
            name="gazebo_gui",
            default_value="false",
            description="Launch Gazebo GUI",
        ),
        DeclareLaunchArgument(
            name="rviz",
            default_value="false",
            description="Launch RViz",
        ),
    ]

    world_model = LaunchConfiguration("world_model")
    robot_model = LaunchConfiguration("robot_model")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    rviz = LaunchConfiguration("rviz")

    pkg_share = FindPackageShare("simulation_bringup").find("simulation_bringup")

    world_model_path = PathJoinSubstitution(
        [
            pkg_share,
            "worlds",
            PythonExpression(["'", world_model, "" + ".world'"]),
        ]
    )
    robot_model_path = PathJoinSubstitution(
        [
            pkg_share,
            "urdf",
            PythonExpression(["'", robot_model, "" + ".urdf.xacro'"]),
        ]
    )
    gazebo_gui_config_path = PathJoinSubstitution(
        [pkg_share, "config", "gazebo_gui.config"]
    )
    ros_gz_bridge_config_path = PathJoinSubstitution(
        [pkg_share, "config", "ros_gz_bridge.yaml"]
    )
    ekf_config_path = PathJoinSubstitution([pkg_share, "config", "ekf.yaml"])
    rviz_gui_config_path = PathJoinSubstitution([pkg_share, "config", "rviz_gui.rviz"])

    gazebo_env = {
        "GZ_SIM_RESOURCE_PATH": ":".join(
            [path for path, _, _ in os.walk(os.path.dirname(pkg_share))]
        ),
        "GZ_SIM_SYSTEM_PLUGIN_PATH": "/opt/ros/jazzy/lib/",
    }

    gazebo_gui_launch = GroupAction(
        actions=[
            ExecuteProcess(
                cmd=(
                    [
                        "gz",
                        "sim",
                        "-r",
                        "--seed",
                        "0",
                        "--gui-config",
                        gazebo_gui_config_path,
                        world_model_path,
                    ]
                ),
                additional_env=gazebo_env,
            )
        ],
        condition=IfCondition(gazebo_gui),
    )

    gazebo_server_launch = GroupAction(
        actions=[
            ExecuteProcess(
                cmd=(
                    [
                        "gz",
                        "sim",
                        "-r",
                        "--seed",
                        "0",
                        "-s",
                        world_model_path,
                    ]
                ),
                additional_env=gazebo_env,
            )
        ],
        condition=UnlessCondition(gazebo_gui),
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": Command(["xacro ", robot_model_path]),
                "publish_frequency": 50.0,
                "use_sim_time": True,
            }
        ],
    )

    spawn_robot_node = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_robot",
        output="screen",
        # Spawn at the robot start position defined by DynamicEnv.X_START = [-2, 0].
        # Matching the initial spawn position to the evaluation start position
        # avoids a large initialisation transient in the first scenario.
        arguments=[
            "-name",
            robot_model,
            "-topic",
            "robot_description",
            "-x",
            "-2.0",
            "-y",
            "0.0",
            "-z",
            "0.25",
            "-yaw",
            "0.0",
        ],
    )

    ros_gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{"config_file": ros_gz_bridge_config_path, "use_sim_time": True}],
    )

    robot_localization_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_node",
        output="screen",
        parameters=[ekf_config_path, {"use_sim_time": True}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        arguments=["-d", rviz_gui_config_path],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(rviz),
    )

    ros2_control_node = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "--switch-timeout",
                "10",
            ],
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "platform_velocity_controller",
                "--switch-timeout",
                "10",
            ],
            parameters=[{"use_sim_time": True}],
        ),
    ]

    dynamic_env_node = TimerAction(
        period=10.0,
        actions=[
            Node(
                package="dynamic_env",
                executable="dynamic_env",
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    return LaunchDescription(
        [
            *launch_args,
            gazebo_gui_launch,
            gazebo_server_launch,
            robot_state_publisher_node,
            spawn_robot_node,
            ros_gz_bridge_node,
            robot_localization_node,
            rviz_node,
            *ros2_control_node,
            dynamic_env_node,
        ]
    )
