"""
Dynamic environment manager for the RNTC-MPC evaluation framework.

This node manages the automated evaluation loop:
  1. For each (MPC type, horizon length) combination it:
       a. Launches the corresponding MPC node.
       b. Iterates through all 100 pre-generated scenarios.
       c. Records success rate, travel time, MPC solve time, and trajectories.
       d. Saves results to JSON.
  2. Publishes the 4 nearest obstacles within the local costmap as a
     Float64MultiArray on /local_obstacles.
  3. Commands obstacle velocities via /cmd_vel_XXX topics.
  4. Detects collisions from the /contact topic.
  5. Bounces obstacles off the scenario boundary when they leave it.

Scenario workspace:
    x ∈ [0, 16] m,  y ∈ [−5, 5] m
Robot start: (−2, 0),  goal: (13, 0).
"""

import os
import json
import threading
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseArray, Twist, PoseStamped
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float32


class DynamicEnv(Node):
    """Automated evaluation environment for the RNTC-MPC benchmark.

    Spawns and removes Gazebo obstacles, monitors robot progress, publishes
    local-obstacle data to the active MPC node, and records experiment results.
    """

    # ------------------------------------------------------------------ #
    # Scenario / workspace constants                                       #
    # ------------------------------------------------------------------ #

    NUM_SCENARIOS: int = 100
    NUM_OBSTACLES: int = 6  # Total obstacles in each scenario
    NUM_LOCAL_OBSTACLES: int = 4  # Obstacles visible to the MPC planner

    # Obstacle spawn region [m]
    POS_LB = np.array([0.0, -5.0])
    POS_UB = np.array([16.0, 5.0])

    # Robot start and goal positions [m, rad]
    X_START = np.array([-2.0, 0.0, 0.0])
    X_GOAL = np.array([13.0, 0.0, 0.0])

    # Obstacle speed: all obstacles move at constant speed, random direction
    SPEED_LB: float = 1.0
    SPEED_UB: float = 1.0

    # Local costmap size used to filter which obstacles the MPC sees
    COSTMAP_SIZE: float = 8.0  # [m]

    # Goal-reached threshold [m]
    GOAL_THRESHOLD: float = 0.2

    # Per-scenario time limit [s]
    SCENARIO_TIMEOUT: float = 90.0

    # MPC variants and horizon lengths to evaluate
    MPC_TYPES = ["rntc_mpc", "ntc_mpc", "dcbf_mpc", "vo_mpc", "sdf_mpc"]
    MPC_HORIZONS = ["5", "10", "20", "30"]

    def __init__(self):
        super().__init__("dynamic_env")
        self.declare_parameter("generate_sdf_models", value=False)
        self.declare_parameter("generate_scenarios", value=False)

        # Resolve asset paths via ament index (works after colcon build)
        pkg_share = get_package_share_directory("dynamic_env")
        assets_dir = os.path.join(pkg_share, "assets")
        xacro_dir = os.path.join(assets_dir, "xacro_models")
        xacro_file = os.path.join(xacro_dir, "obstacle.urdf.xacro")
        urdf_dir = os.path.join(assets_dir, "urdf_models")
        self.sdf_dir = os.path.join(assets_dir, "sdf_models")
        self.scenarios_path = os.path.join(assets_dir, "scenarios", "scenarios.json")

        self.world_model = "empty"
        self.robot_model = "jackal_robot"

        # Regenerate SDF models if requested (e.g. after changing the obstacle shape)
        if self.get_parameter("generate_sdf_models").value:
            self._regenerate_sdf_models(xacro_file, urdf_dir)

        # Per-obstacle state (updated from /obstacle_pose)
        self.positions = np.zeros((self.NUM_OBSTACLES, 2))
        self.velocities = np.zeros((self.NUM_OBSTACLES, 2))

        # Locks to prevent repeated velocity sign flips at the boundary
        self.cmd_vel_lock = np.zeros(self.NUM_OBSTACLES, dtype=bool)

        self.env_ready = False
        self.collision = False
        self.goal_reached = False
        self.mpc_opt_time: list = []
        self.measured_state: list = []

        self.load_scenarios()

        # ---- Subscriptions ----
        self.obstacle_pose_sub = self.create_subscription(
            msg_type=PoseArray,
            topic="/obstacle_pose",
            callback=self.obstacle_pose_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.robot_odom_sub = self.create_subscription(
            msg_type=Odometry,
            topic="/odometry",
            callback=self.robot_odom_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.contact_sub = self.create_subscription(
            msg_type=Contacts,
            topic="/contact",
            callback=self.contact_callback,
            qos_profile=10,
        )
        self.mpc_opt_time_sub = self.create_subscription(
            msg_type=Float32,
            topic="/mpc_opt_time",
            callback=self.mpc_opt_time_callback,
            qos_profile=10,
        )

        # ---- Publishers ----
        self.local_obstacles_pub = self.create_publisher(
            msg_type=Float64MultiArray,
            topic="/local_obstacles",
            qos_profile=10,
        )
        self.goal_pose_pub = self.create_publisher(
            msg_type=PoseStamped,
            topic="/goal_pose",
            qos_profile=10,
        )
        # One velocity publisher per obstacle
        self.obstacle_cmd_vel_pub = [
            self.create_publisher(Twist, f"/cmd_vel_{i:03}", qos_profile=5)
            for i in range(self.NUM_OBSTACLES)
        ]

        # Start the evaluation loop in a daemon thread so it does not block spinning
        thread = threading.Thread(target=self.iterate_scenarios, daemon=True)
        thread.start()

    # ------------------------------------------------------------------ #
    # Scenario generation / loading                                        #
    # ------------------------------------------------------------------ #

    def generate_scenarios(self) -> None:
        """Generate and save NUM_SCENARIOS random obstacle configurations.

        Each scenario has NUM_OBSTACLES obstacles with random positions
        inside [POS_LB, POS_UB] and random constant velocities of
        magnitude in [SPEED_LB, SPEED_UB].
        """
        scenarios = []
        for _ in range(self.NUM_SCENARIOS):
            positions = self.POS_LB + np.random.rand(self.NUM_OBSTACLES, 2) * (
                self.POS_UB - self.POS_LB
            )
            speeds = self.SPEED_LB + np.random.rand(self.NUM_OBSTACLES) * (
                self.SPEED_UB - self.SPEED_LB
            )
            thetas = -np.pi + np.random.rand(self.NUM_OBSTACLES) * 2 * np.pi
            velocities = np.stack(
                [speeds * np.cos(thetas), speeds * np.sin(thetas)], axis=-1
            )
            scenarios.append(
                {"positions": positions.tolist(), "velocities": velocities.tolist()}
            )

        with open(self.scenarios_path, "w") as f:
            json.dump(scenarios, f, indent=4)
        self.get_logger().info(
            f"Generated {self.NUM_SCENARIOS} scenarios → {self.scenarios_path}"
        )

    def load_scenarios(self) -> None:
        """Load pre-generated scenarios from the JSON file."""
        with open(self.scenarios_path, "r") as f:
            self.scenarios = json.load(f)
        self.get_logger().info(
            f"Loaded {len(self.scenarios)} scenarios from {self.scenarios_path}"
        )

    # ------------------------------------------------------------------ #
    # Evaluation loop                                                      #
    # ------------------------------------------------------------------ #

    def iterate_scenarios(self) -> None:
        """Main evaluation loop: test every MPC variant × horizon combination.

        For each (mpc_type, N) pair:
          1. Pause the simulation.
          2. Launch the MPC node (with NLP compilation).
          3. Wait for compilation to finish.
          4. Run all 100 scenarios and record metrics.
          5. Kill the MPC node.
          6. Save results to JSON.
        """
        for mpc_type in self.MPC_TYPES:
            for N in self.MPC_HORIZONS:
                self._pause_simulation()

                # Launch MPC node in a background thread
                def _start_mpc(mt=mpc_type, n=N):
                    subprocess.run(
                        [
                            "ros2",
                            "run",
                            "mpc_local_planner",
                            mt,
                            "--ros-args",
                            "-p",
                            "compile_nlp_solver:=true",
                            "-p",
                            f"horizon_length:={n}",
                            "-p",
                            "use_sim_time:=true",
                        ]
                    )

                mpc_thread = threading.Thread(target=_start_mpc, daemon=True)
                mpc_thread.start()

                # VO-MPC compiles a larger NLP (geometric constraints per step),
                # so it requires more compilation time than the other variants.
                compile_wait = 90 if mpc_type == "vo_mpc" else 45
                self.get_logger().info(
                    f"[{mpc_type} N={N}] Waiting {compile_wait}s for NLP compilation…"
                )
                time.sleep(compile_wait)

                travel_times: list = []
                mean_opt_times: list = []
                successes: list = []
                position_logs: list = []
                timed_out: list = []

                for i in range(self.NUM_SCENARIOS):
                    self.mpc_opt_time = []
                    self.measured_state = []
                    self.get_logger().info(
                        f"[{mpc_type} N={N}] Scenario {i:3d} | "
                        f"Success so far: {sum(successes)}/{len(successes)}"
                    )

                    self.initialize_robot(reset_flags=True)
                    self.spawn_scenario(i)
                    time.sleep(2)  # Let Gazebo settle after spawning obstacles

                    self._unpause_simulation()
                    self.env_ready = True
                    self.publish_goal_pose(self.X_GOAL)

                    t_start = self.get_clock().now().nanoseconds / 1e9
                    t_now = t_start
                    while (
                        not self.goal_reached
                        and not self.collision
                        and (t_now - t_start) <= self.SCENARIO_TIMEOUT
                    ):
                        t_now = self.get_clock().now().nanoseconds / 1e9

                    self.env_ready = False
                    self.clear_scenario()
                    self.initialize_robot(reset_flags=False)
                    time.sleep(3)  # Let Gazebo settle after removing obstacles
                    self._pause_simulation()

                    # Record results
                    # Down-sample trajectory to ~10 Hz (every other 20 Hz sample)
                    position_logs.append(self.measured_state[::2])
                    elapsed = t_now - t_start

                    if self.goal_reached:
                        travel_times.append(elapsed)
                        mean_opt_times.append(float(np.mean(self.mpc_opt_time)))
                        successes.append(1)
                        timed_out.append(0)
                        self.get_logger().info(f"  → GOAL REACHED in {elapsed:.1f}s")
                    else:
                        travel_times.append(float("nan"))
                        mean_opt_times.append(float("nan"))
                        successes.append(0)
                        if self.collision:
                            self.get_logger().info("  → COLLISION")
                            timed_out.append(0)
                        else:
                            self.get_logger().info("  → TIMED OUT")
                            timed_out.append(1)

                # Kill the MPC process
                subprocess.run(["pkill", "-f", mpc_type])
                time.sleep(2)

                self._save_results(
                    mpc_type=mpc_type,
                    N=N,
                    successes=successes,
                    travel_times=travel_times,
                    mean_opt_times=mean_opt_times,
                    timed_out=timed_out,
                    position_logs=position_logs,
                )

    # ------------------------------------------------------------------ #
    # Gazebo service wrappers                                              #
    # ------------------------------------------------------------------ #

    def _gz_service(self, service: str, reqtype: str, reptype: str, req: str) -> None:
        """Call a Gazebo service via the gz CLI tool."""
        subprocess.run(
            [
                "gz",
                "service",
                "-s",
                service,
                "--reqtype",
                reqtype,
                "--reptype",
                reptype,
                "--timeout",
                "1000",
                "--req",
                req,
            ],
            check=False,  # Non-fatal: Gazebo may miss occasional calls
        )

    def _pause_simulation(self) -> None:
        """Pause the Gazebo simulation clock."""
        self._gz_service(
            f"/world/{self.world_model}/control",
            "gz.msgs.WorldControl",
            "gz.msgs.Boolean",
            "pause: true",
        )

    def _unpause_simulation(self) -> None:
        """Resume the Gazebo simulation clock."""
        self._gz_service(
            f"/world/{self.world_model}/control",
            "gz.msgs.WorldControl",
            "gz.msgs.Boolean",
            "pause: false",
        )

    def spawn_scenario(self, scenario_idx: int) -> None:
        """Spawn obstacles for the given scenario index.

        Loads positions and velocities from the pre-generated scenarios list,
        creates each obstacle model in Gazebo, then commands initial velocities.
        """
        scenario = self.scenarios[scenario_idx]
        self.positions = np.array(scenario["positions"])
        self.velocities = np.array(scenario["velocities"])

        for i in range(self.NUM_OBSTACLES):
            sdf_file = os.path.join(self.sdf_dir, f"obstacle_{i:03}.sdf")
            self._gz_service(
                f"/world/{self.world_model}/create",
                "gz.msgs.EntityFactory",
                "gz.msgs.Boolean",
                (
                    f'sdf_filename: "{sdf_file}", '
                    f'name: "obstacle_{i:03}", '
                    f"pose: {{position: {{x: {self.positions[i, 0]}, y: {self.positions[i, 1]}}}}}"
                ),
            )

        # Brief delay so Gazebo can finish creating the entities before we
        # command their velocities
        time.sleep(2)

        for i in range(self.NUM_OBSTACLES):
            twist_msg = Twist()
            twist_msg.linear.x = float(self.velocities[i, 0])
            twist_msg.linear.y = float(self.velocities[i, 1])
            self.obstacle_cmd_vel_pub[i].publish(twist_msg)

    def clear_scenario(self) -> None:
        """Remove all obstacle entities from the Gazebo world."""
        for i in range(self.NUM_OBSTACLES):
            self._gz_service(
                f"/world/{self.world_model}/remove",
                "gz.msgs.Entity",
                "gz.msgs.Boolean",
                f'name: "obstacle_{i:03}", type: MODEL',
            )

    def initialize_robot(self, reset_flags: bool) -> None:
        """Teleport the robot to the start pose and optionally reset status flags.

        Args:
            reset_flags: If True, clear collision and goal_reached flags.
        """
        self._gz_service(
            f"/world/{self.world_model}/set_pose",
            "gz.msgs.Pose",
            "gz.msgs.Boolean",
            (
                f'name: "{self.robot_model}", '
                f"position: {{x: {self.X_START[0]}, y: {self.X_START[1]}, z: 0.25}}, "
                "orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"
            ),
        )

        # Publish the start pose as the initial goal so the MPC tracks (0,0,0) relative
        start_msg = PoseStamped()
        start_msg.header.frame_id = "map"
        start_msg.header.stamp = self.get_clock().now().to_msg()
        start_msg.pose.position.x = float(self.X_START[0])
        start_msg.pose.position.y = float(self.X_START[1])
        for _ in range(3):
            self.goal_pose_pub.publish(start_msg)

        if reset_flags:
            self.collision = False
            self.goal_reached = False

    # ------------------------------------------------------------------ #
    # Subscriptions                                                        #
    # ------------------------------------------------------------------ #

    def robot_odom_callback(self, msg: Odometry) -> None:
        """Track robot state: check for goal reaching and log trajectory."""
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        yaw = float(np.arctan2(w * z, 0.5 - z**2))
        robot_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # Goal detection
        if np.linalg.norm(robot_pos - self.X_GOAL[:2]) < self.GOAL_THRESHOLD:
            self.goal_reached = True

        # Trajectory logging (only while a scenario is active)
        if self.env_ready:
            self.measured_state.append(robot_pos.tolist())

        # Compute and publish the 4 nearest visible obstacles
        self._publish_local_obstacles(robot_pos)

    def _publish_local_obstacles(self, robot_pos: np.ndarray) -> None:
        """Select up to NUM_LOCAL_OBSTACLES nearest obstacles inside the costmap
        and publish them as a Float64MultiArray with layout [obs_idx, field]
        where fields are [px, py, vx, vy].

        Absent obstacle slots are filled with NaN so the MPC can identify them.
        """
        half = self.COSTMAP_SIZE / 2.0
        lb = robot_pos - half
        ub = robot_pos + half

        # Collect indices of obstacles within the local costmap
        visible_idx = [
            i
            for i in range(self.NUM_OBSTACLES)
            if self._point_inside_bounds(self.positions[i], lb, ub)
        ]

        local_obs = np.full((self.NUM_LOCAL_OBSTACLES, 4), np.nan)

        if len(visible_idx) > self.NUM_LOCAL_OBSTACLES:
            # Too many visible: keep the nearest NUM_LOCAL_OBSTACLES
            dists = np.linalg.norm(self.positions[visible_idx, :] - robot_pos, axis=-1)
            closest = np.array(visible_idx)[
                np.argsort(dists)[: self.NUM_LOCAL_OBSTACLES]
            ]
            local_obs[:, :2] = self.positions[closest, :]
            local_obs[:, 2:] = self.velocities[closest, :]
        else:
            n = len(visible_idx)
            local_obs[:n, :2] = self.positions[visible_idx, :]
            local_obs[:n, 2:] = self.velocities[visible_idx, :]

        # Build Float64MultiArray message with explicit layout
        msg = Float64MultiArray()
        msg.data = local_obs.flatten().tolist()
        rows_dim = MultiArrayDimension()
        rows_dim.label = "rows"
        rows_dim.size = local_obs.shape[0]
        rows_dim.stride = local_obs.shape[0] * local_obs.shape[1]
        cols_dim = MultiArrayDimension()
        cols_dim.label = "columns"
        cols_dim.size = local_obs.shape[1]
        cols_dim.stride = local_obs.shape[1]
        msg.layout.dim = [rows_dim, cols_dim]
        self.local_obstacles_pub.publish(msg)

    def obstacle_pose_callback(self, msg: PoseArray) -> None:
        """Update obstacle positions from Gazebo world-pose broadcast.

        Gazebo publishes all entity poses in one message.  The first two entries
        are the ground plane and the robot, so obstacles start at index 2.

        When an obstacle leaves the spawn region it has its velocity reversed
        (wall bounce).  A lock bit prevents repeated sign flips while the
        obstacle is still outside the boundary.
        """
        if not self.env_ready:
            return

        offset = 2  # Skip ground-plane and robot entries
        for i in range(self.NUM_OBSTACLES):
            self.positions[i, 0] = msg.poses[i + offset].position.x
            self.positions[i, 1] = msg.poses[i + offset].position.y

            inside = self._point_inside_bounds(
                self.positions[i], self.POS_LB, self.POS_UB
            )

            if not inside and not self.cmd_vel_lock[i]:
                # Reverse velocity (bounce) and lock to prevent oscillation
                self.velocities[i] = -self.velocities[i]
                twist = Twist()
                twist.linear.x = float(self.velocities[i, 0])
                twist.linear.y = float(self.velocities[i, 1])
                self.obstacle_cmd_vel_pub[i].publish(twist)
                self.cmd_vel_lock[i] = True
            elif inside and self.cmd_vel_lock[i]:
                # Obstacle re-entered the boundary: release the lock
                self.cmd_vel_lock[i] = False

    def contact_callback(self, msg: Contacts) -> None:
        """Set collision flag when any contact event is detected."""
        if self.env_ready:
            self.collision = True

    def mpc_opt_time_callback(self, msg: Float32) -> None:
        """Record MPC optimisation time for the current scenario."""
        if self.env_ready:
            self.mpc_opt_time.append(msg.data)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def publish_goal_pose(self, goal: np.ndarray) -> None:
        """Publish the goal position to /goal_pose three times for reliability."""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal[0])
        msg.pose.position.y = float(goal[1])
        for _ in range(3):
            self.goal_pose_pub.publish(msg)

    @staticmethod
    def _point_inside_bounds(point: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> bool:
        """Return True iff point is strictly inside the axis-aligned bounding box."""
        return bool(lb[0] < point[0] < ub[0] and lb[1] < point[1] < ub[1])

    def _save_results(
        self,
        mpc_type: str,
        N: str,
        successes: list,
        travel_times: list,
        mean_opt_times: list,
        timed_out: list,
        position_logs: list,
    ) -> None:
        """Serialise evaluation results to a JSON file.

        Files are written to ./simulation_results/ relative to the working
        directory at launch time.
        """
        results = {
            "world_model": self.world_model,
            "robot_model": self.robot_model,
            "mpc_type": mpc_type,
            "N": N,
            "success": successes,
            "success_rate": float(np.mean(successes)),
            "travel_time": travel_times,
            "mean_travel_time": float(np.nanmean(travel_times)),
            "mean_mpc_opt_time": mean_opt_times,
            "timed_out": timed_out,
            "timed_out_rate": float(np.mean(timed_out)),
            "position_log": position_logs,
        }

        out_dir = "simulation_results"
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{self.world_model}_{self.robot_model}_{mpc_type}_{N}.json"
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f)
        self.get_logger().info(f"Results saved → {path}")

    def _regenerate_sdf_models(self, xacro_file: str, urdf_dir: str) -> None:
        """Re-generate URDF and SDF obstacle models from the xacro template.

        Only needed when the obstacle geometry changes.  Existing files are
        removed before re-generation.
        """
        for directory in [urdf_dir, self.sdf_dir]:
            os.makedirs(directory, exist_ok=True)
            for fname in os.listdir(directory):
                fpath = os.path.join(directory, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)

        for i in range(self.NUM_OBSTACLES):
            urdf_file = os.path.join(urdf_dir, f"obstacle_{i:03}.urdf")
            sdf_file = os.path.join(self.sdf_dir, f"obstacle_{i:03}.sdf")
            subprocess.run(
                [
                    "ros2",
                    "run",
                    "xacro",
                    "xacro",
                    f"id:={i:03}",
                    "v_x:=0.0",
                    "v_y:=0.0",
                    xacro_file,
                ],
                stdout=open(urdf_file, "w"),
                check=True,
            )
            subprocess.run(
                ["gz", "sdf", "-p", urdf_file],
                stdout=open(sdf_file, "w"),
                check=True,
            )
        self.get_logger().info("SDF obstacle models regenerated successfully.")


def main(args=None):
    rclpy.init(args=args)
    node = DynamicEnv()
    rclpy.spin(node)
    node.destroy_node()  # Fixed: was incorrectly spelled "destry_node"
    rclpy.shutdown()


if __name__ == "__main__":
    main()
