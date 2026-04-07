"""
Base class for all MPC-based local planners in the RNTC-MPC framework.

All five MPC variants (RNTC, NTC, DCBF, VO, SDF) share the same ROS2 interface,
dynamics model, solver infrastructure, and callback structure. This module extracts
that shared logic so that each variant only implements its unique NLP formulation.

Robot model: Jackal (unicycle / Dubins-car kinematics)
    x_{k+1} = x_k + dt * [v_k cos θ_k,  v_k sin θ_k,  ω_k]^T

NLP structure (common to all variants):
    min  Σ (x_i - x_ref)^T Q (x_i - x_ref) + u_i^T R u_i
    s.t. x_0  = x_init                     (initial condition)
         x_{i+1} = f(x_i, u_i)             (dynamics)
         u_min ≤ u_i ≤ u_max               (control limits)
         variant-specific obstacle constraints

The NLP is generated symbolically with CasADi, compiled to a C shared library
once (with GCC -O3), and then loaded for fast repeated evaluation.  The HSL MA57
sparse linear solver (via IPOPT) is used for efficiency.
"""

import os
import shutil
import tempfile
import subprocess
import time

import numpy as np
import casadi as ca
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TwistStamped
from launch_ros.substitutions import FindPackageShare
from std_msgs.msg import Float64MultiArray, Float32


class BaseMPC(Node):
    """Abstract base class for all MPC local planner variants.

    Subclasses must implement:
        NODE_NAME  (class attribute) – ROS2 node name
        configure_mpc()              – builds the NLP and loads the solver;
                                       must set self.solver, self.lbX, self.ubX,
                                       self.lbG, self.ubG, self.x_opt,
                                       self.lam_x_opt, self.lam_g_opt,
                                       self.predictions, and self.N
        local_obstacles_callback()   – processes incoming obstacle data

    Subclasses may override:
        _build_nlp_params()          – appends extra parameters (e.g. neural-net
                                       weights or obstacle velocities) to P
    """

    # ------------------------------------------------------------------ #
    # Class-level constants shared by all MPC variants                    #
    # ------------------------------------------------------------------ #

    NODE_NAME: str = ""  # Override in subclass

    # Unicycle model dimensions
    NX: int = 3   # State: [x, y, yaw]
    NU: int = 2   # Control: [linear_velocity, angular_velocity]

    # Physical parameters
    ROBOT_RADIUS: float = 0.4      # [m]
    OBSTACLE_RADIUS: float = 0.5   # [m]
    MIN_DISTANCE: float = ROBOT_RADIUS + OBSTACLE_RADIUS  # Minimum centre-to-centre clearance [m]

    # Planner parameters
    DT: float = 0.1                 # MPC sampling time [s]
    MAX_NUM_OBSTACLES: int = 4      # Maximum number of obstacles tracked locally

    # Control bounds [m/s, rad/s]
    U_MIN: list = [-0.5, -0.5]
    U_MAX: list = [0.5, 0.5]

    # Cost weights
    Q_DIAG: list = [50.0, 50.0, 1.0]    # Stage state cost
    QN_DIAG: list = [50.0, 50.0, 1.0]   # Terminal state cost
    R_DIAG: list = [1.0, 1.0]            # Control cost

    # IPOPT settings
    IPOPT_MAX_WALL_TIME: float = 0.1   # Maximum solve time per step [s]
    HSL_LIB_PATH: str = "/usr/local/lib/libcoinhsl.so"

    # MPC control frequency
    MPC_TIMER_PERIOD: float = 0.05   # 20 Hz [s]

    # Sentinel value used to fill NaN obstacle positions in the NLP.
    # Must be far enough from the operating area so all distance
    # constraints are trivially satisfied for these phantom obstacles.
    NAN_SENTINEL: float = -100.0

    def __init__(self):
        if not self.NODE_NAME:
            raise NotImplementedError("Subclass must define NODE_NAME")
        super().__init__(self.NODE_NAME)

        # Declare ROS2 parameters
        self.declare_parameter("compile_nlp_solver", value=False)
        self.declare_parameter("horizon_length", value=20)

        # Cache frequently used values
        self.dt = self.DT
        self.nx = self.NX
        self.nu = self.NU
        self.max_num_obstacles = self.MAX_NUM_OBSTACLES
        self.min_distance = self.MIN_DISTANCE
        self.N: int = self.get_parameter("horizon_length").value

        # Initialise state/reference to safe defaults (overwritten by callbacks)
        self.x_init = np.zeros(self.nx)
        self.x_ref = np.zeros(self.nx)

        # Build the NLP and load the compiled solver (subclass responsibility)
        self.configure_mpc()

        # ---- Publishers ----
        self.cmd_vel_pub = self.create_publisher(
            msg_type=TwistStamped,
            topic="/cmd_vel",
            qos_profile=10,
        )
        self.predicted_path_pub = self.create_publisher(
            msg_type=Path,
            topic="/predicted_path",
            qos_profile=10,
        )
        self.mpc_opt_time_pub = self.create_publisher(
            msg_type=Float32,
            topic="/mpc_opt_time",
            qos_profile=10,
        )

        # ---- Subscriptions ----
        self.goal_pose_sub = self.create_subscription(
            msg_type=PoseStamped,
            topic="/goal_pose",
            callback=self.goal_pose_callback,
            qos_profile=5,
        )
        self.odometry_sub = self.create_subscription(
            msg_type=Odometry,
            topic="/odometry",
            callback=self.odometry_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.local_obstacles_sub = self.create_subscription(
            msg_type=Float64MultiArray,
            topic="/local_obstacles",
            callback=self.local_obstacles_callback,
            qos_profile=qos_profile_sensor_data,
        )

        # ---- Control timer ----
        self.timer = self.create_timer(
            timer_period_sec=self.MPC_TIMER_PERIOD,
            callback=self.mpc_callback,
        )

    # ------------------------------------------------------------------ #
    # Shared callbacks                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _quat_z_to_yaw(z: float, w: float) -> float:
        """Extract yaw from a quaternion with zero roll and pitch.

        For a rotation purely around the z-axis the quaternion is
        q = (0, 0, sin(θ/2), cos(θ/2)).  The formula below is equivalent
        to the standard 2·atan2(z, w) but expressed in a numerically
        convenient form:

            arctan2(w·z, 0.5 − z²)
            = arctan2(cos(θ/2)·sin(θ/2), 0.5 − sin²(θ/2))
            = arctan2(sin(θ)/2, cos(θ)/2)
            = arctan2(sin θ, cos θ)
            = θ
        """
        return float(np.arctan2(w * z, 0.5 - z ** 2))

    def goal_pose_callback(self, msg: PoseStamped) -> None:
        """Update the goal state from the /goal_pose topic."""
        yaw = self._quat_z_to_yaw(msg.pose.orientation.z, msg.pose.orientation.w)
        self.x_ref = [msg.pose.position.x, msg.pose.position.y, yaw]

    def odometry_callback(self, msg: Odometry) -> None:
        """Update the current robot state from the /odometry topic."""
        yaw = self._quat_z_to_yaw(
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        )
        self.x_init = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    # ------------------------------------------------------------------ #
    # Dynamics model                                                       #
    # ------------------------------------------------------------------ #

    def system_dynamics(self, x: ca.MX, u: ca.MX, dt: float) -> ca.MX:
        """Unicycle (Dubins-car) forward Euler step.

        Continuous-time model:
            ẋ = v cos θ
            ẏ = v sin θ
            θ̇ = ω
        Discretised with explicit Euler at time step dt.
        """

        def f(x, u):
            return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])

        return x + f(x, u) * dt

    # ------------------------------------------------------------------ #
    # Solver management                                                    #
    # ------------------------------------------------------------------ #

    def _get_solver_path(self, solver_name: str) -> str:
        """Return the absolute path of the compiled .so solver file."""
        pkg_share = FindPackageShare("mpc_local_planner").find("mpc_local_planner")
        compiled_dir = os.path.join(pkg_share, "assets", "casadi_compiled")
        os.makedirs(compiled_dir, exist_ok=True)
        return os.path.join(compiled_dir, f"{solver_name}.so")

    def _compile_nlp_to_so(self, nlp: dict, solver_name: str, so_path: str) -> None:
        """Compile the NLP to a C shared library via CasADi + GCC.

        CasADi first generates optimised C code for the NLP, which is then
        compiled to a position-independent shared library.  Loading the .so
        instead of rebuilding the computational graph at every run yields a
        significant speed improvement for large horizons.

        Args:
            nlp:         CasADi NLP problem dict {"f", "x", "g", "p"}.
            solver_name: Base filename (without extension) for output files.
            so_path:     Absolute path where the .so file should be written.
        """
        # Brief delay to ensure logging infrastructure is fully initialised
        # before the lengthy compilation output begins.
        time.sleep(4)
        sep = "#" * 50
        self.get_logger().info(sep)
        self.get_logger().info(f"Compiling NLP solver: {solver_name}")
        self.get_logger().info(sep)

        # Build an un-compiled CasADi solver to generate the C source
        proto_solver = ca.nlpsol("nlp_solver", "ipopt", nlp)

        c_filename = f"{solver_name}.c"
        with tempfile.TemporaryDirectory() as tmpdir:
            # generate_dependencies() writes to cwd; move immediately to tmpdir
            proto_solver.generate_dependencies(c_filename)
            c_path = os.path.join(tmpdir, c_filename)
            shutil.move(c_filename, c_path)

            result = subprocess.run(
                ["gcc", "-fPIC", "-shared", "-O3", c_path, "-o", so_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"GCC compilation failed for {solver_name}:\n{result.stderr}"
                )

        self.get_logger().info(sep)
        self.get_logger().info(f"Solver compiled successfully → {so_path}")
        self.get_logger().info(sep)

    def _load_solver(self, so_path: str) -> ca.nlpsol:
        """Load a pre-compiled NLP solver from a shared library.

        Uses IPOPT with the HSL MA57 sparse direct linear solver for
        efficient interior-point iterations.
        """
        opts = {
            "ipopt": {
                "print_level": 5,
                "max_wall_time": self.IPOPT_MAX_WALL_TIME,
                "warm_start_init_point": "yes",
                "hsllib": self.HSL_LIB_PATH,
                "linear_solver": "ma57",
            }
        }
        return ca.nlpsol("nlp_solver", "ipopt", so_path, opts)

    def _get_solver(self, nlp: dict, solver_name: str) -> ca.nlpsol:
        """Compile (if requested) and load the NLP solver.

        Args:
            nlp:         CasADi NLP problem dict.
            solver_name: Base filename for the compiled solver.

        Returns:
            Loaded CasADi IPOPT solver ready for repeated evaluation.

        Raises:
            FileNotFoundError: If the compiled .so does not exist and
                               compile_nlp_solver is False.
        """
        so_path = self._get_solver_path(solver_name)

        if self.get_parameter("compile_nlp_solver").value:
            self._compile_nlp_to_so(nlp, solver_name, so_path)

        if not os.path.isfile(so_path):
            raise FileNotFoundError(
                f"Compiled solver not found: {so_path}\n"
                "Launch the node with --ros-args -p compile_nlp_solver:=true "
                "to compile it first."
            )

        return self._load_solver(so_path)

    # ------------------------------------------------------------------ #
    # NLP parameter vector assembly                                        #
    # ------------------------------------------------------------------ #

    def _build_nlp_params(self) -> list:
        """Assemble the numerical parameter vector P for the NLP solver.

        The base implementation produces:
            [x_init, x_ref, vec(predictions[0]), …, vec(predictions[N])]

        Subclasses that add extra parameters (neural-net weights, obstacle
        velocities, etc.) should override this method.

        Returns:
            List of CasADi DM / numpy arrays to be concatenated into P.
        """
        P = [self.x_init, self.x_ref]
        for i in range(self.N + 1):
            P.append(ca.vec(ca.DM(self.predictions[i])))
        return P

    # ------------------------------------------------------------------ #
    # MPC control loop                                                     #
    # ------------------------------------------------------------------ #

    def mpc_callback(self) -> None:
        """Run one MPC step (called at 20 Hz by the ROS2 timer).

        Solves the NLP with warm-starting from the previous solution,
        then publishes the first optimal control, the predicted trajectory,
        and the optimisation wall time.
        """
        P = ca.vertcat(*self._build_nlp_params())

        opt_sol = self.solver(
            x0=self.x_opt,
            lam_x0=self.lam_x_opt,
            lam_g0=self.lam_g_opt,
            lbx=self.lbX,
            ubx=self.ubX,
            lbg=self.lbG,
            ubg=self.ubG,
            p=P,
        )

        # Extract and store solution for warm-starting the next iteration
        self.x_opt = opt_sol["x"].full().flatten()
        self.lam_x_opt = opt_sol["lam_x"]
        self.lam_g_opt = opt_sol["lam_g"]

        # ---- Publish velocity command ----
        # Decision variables are ordered: [x_0, u_0, x_1, u_1, …, x_N]
        # The first control u_0 starts at index nx
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.twist.linear.x = float(self.x_opt[self.nx])       # v [m/s]
        twist_msg.twist.angular.z = float(self.x_opt[self.nx + 1])  # ω [rad/s]
        self.cmd_vel_pub.publish(twist_msg)

        # ---- Publish predicted path (for visualisation) ----
        path_msg = Path()
        path_msg.header.stamp = twist_msg.header.stamp
        path_msg.header.frame_id = "odom"
        stride = self.nx + self.nu  # state + control per step
        for i in range(self.N):
            idx = (i + 1) * stride
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(self.x_opt[idx])
            pose.pose.position.y = float(self.x_opt[idx + 1])
            yaw = float(self.x_opt[idx + 2])
            pose.pose.orientation.z = np.sin(yaw / 2.0)
            pose.pose.orientation.w = np.cos(yaw / 2.0)
            path_msg.poses.append(pose)
        self.predicted_path_pub.publish(path_msg)

        # ---- Publish MPC optimisation time ----
        opt_time_msg = Float32()
        opt_time_msg.data = float(self.solver.stats()["t_wall_total"] * 1000)  # ms
        self.mpc_opt_time_pub.publish(opt_time_msg)

    # ------------------------------------------------------------------ #
    # Abstract interface (subclasses must implement)                       #
    # ------------------------------------------------------------------ #

    def configure_mpc(self) -> None:
        """Build the NLP formulation and load the compiled solver.

        Subclasses must:
          1. Build symbolic variables X, G, J, P using CasADi.
          2. Populate self.lbX, self.ubX, self.lbG, self.ubG.
          3. Call self._get_solver(nlp, solver_name) to compile/load.
          4. Assign self.solver, self.x_opt, self.lam_x_opt, self.lam_g_opt.
          5. Initialise self.predictions to safe defaults.
        """
        raise NotImplementedError("configure_mpc() must be implemented by subclass")

    def local_obstacles_callback(self, msg: Float64MultiArray) -> None:
        """Process incoming local-obstacle observations.

        Subclasses must update self.predictions (shape: [N+1, max_num_obstacles, 2])
        with predicted obstacle positions for each time step in the horizon.
        """
        raise NotImplementedError(
            "local_obstacles_callback() must be implemented by subclass"
        )
