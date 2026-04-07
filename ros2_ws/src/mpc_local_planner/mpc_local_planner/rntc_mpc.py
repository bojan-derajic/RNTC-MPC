"""
RNTC-MPC: Residual Neural Terminal Constraint MPC.

This is the primary contribution of the paper.  The terminal constraint is:

    SDF(x_N) ≥ 1.4 · φ(x_N − x_init; θ)

where φ is a small neural network (main network) whose weights θ are predicted
on-the-fly by a convolutional hypernetwork conditioned on two 2-D signed-distance
fields (SDF) of the predicted obstacle configuration.

The "residual" name comes from the fact that the main network learns to refine
the baseline SDF constraint: the factor 1.4 scales φ so that its output acts as
a safety-region multiplier on top of the raw SDF value.

Architecture:
    Hypernetwork (PyTorch, offline):
        Input   → 2-channel SDF grid (terminal + stage, 100×100)
        Backbone→ 4× [Conv2d → ReLU → MaxPool2d]
        Head    → Linear(2048 → num_params)

    Main network (CasADi, embedded in NLP):
        Input   → [x − x_init, y − y_init, θ]   (robot-relative coordinates)
        Layers  → 3×36-SIN → 3×18-SELU → 3×9-SELU → 1-ELU+1
        Output  → scalar ≥ 1  (safety multiplier)

NLP formulation:
    Decision variables: [x_0, u_0, x_1, u_1, …, x_N]
    Parameters:         [x_init, x_ref, predictions_{0..N}, main_net_weights]
    Objective:          Σ_i (x_i−x_ref)^T Q (x_i−x_ref) + u_i^T R u_i
                        + (x_N−x_ref)^T Q_N (x_N−x_ref)
    Constraints:
        x_0 = x_init
        x_{i+1} = f(x_i, u_i)  ∀i
        ‖obs_{i,j} − x_i[:2]‖ ≥ min_dist  ∀i,j  (stage obstacle avoidance)
        SDF(x_N) − 1.4·φ(x_N) ≥ 0               (residual neural terminal constraint)
"""

import os
import numpy as np
import torch
import rclpy
from launch_ros.substitutions import FindPackageShare
import casadi as ca

from mpc_local_planner.base_mpc import BaseMPC
from mpc_local_planner.models import MainNetwork, Hypernetwork


class RNTC_MPC(BaseMPC):
    """MPC with Residual Neural Terminal Constraint (RNTC).

    The hypernetwork predicts main-network weights from the local SDF, and the
    main network defines the terminal safety region that the NLP must satisfy.
    """

    NODE_NAME = "rntc_mpc"

    # SDF grid parameters (shared with NTC variant)
    SDF_SIZE: float = 8.0     # Physical extent of the local costmap [m]
    SDF_RES: float = 0.08     # Grid resolution [m/cell]

    # Main-network architecture (must match the pre-trained weights)
    MAIN_NET_CONFIG = {
        "input_size": 3,
        "layers": [
            (36, "sin"),
            (36, "sin"),
            (36, "sin"),
            (18, "selu"),
            (18, "selu"),
            (18, "selu"),
            (9, "selu"),
            (9, "selu"),
            (9, "selu"),
            (1, "elu_plus_1"),  # Output ≥ 1, acts as safety multiplier
        ],
    }

    # Scale factor in the residual terminal constraint: SDF ≥ LAMBDA · φ(x_N)
    LAMBDA: float = 1.4

    def configure_mpc(self) -> None:
        """Build the RNTC-MPC NLP and load (or compile) the solver."""
        sdf_num_cells = int(self.SDF_SIZE / self.SDF_RES)

        # ---- Pre-compute the local XY grid for SDF evaluation ----
        # Grid spans [−(size−res)/2, +(size−res)/2] in both axes,
        # expressed in the robot-centred local frame.
        axis = np.linspace(
            -(self.SDF_SIZE - self.SDF_RES) / 2,
            (self.SDF_SIZE - self.SDF_RES) / 2,
            sdf_num_cells,
        )
        self.xy_grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)

        # ---- Define the CasADi main network (embedded in the NLP) ----
        main_net = MainNetwork(self.MAIN_NET_CONFIG)

        # ---- Build symbolic NLP ----
        x_min = [-ca.inf, -ca.inf, -ca.inf]
        x_max = [ca.inf, ca.inf, ca.inf]
        Q = ca.diag(self.Q_DIAG)
        QN = ca.diag(self.QN_DIAG)
        R = ca.diag(self.R_DIAG)

        x_init_sym = ca.MX.sym("x_init", self.nx)
        x_ref_sym = ca.MX.sym("x_ref", self.nx)

        # Obstacle position predictions at each time step (world frame)
        pred_syms = [
            ca.MX.sym(f"predictions_{i}", self.max_num_obstacles, 2)
            for i in range(self.N + 1)
        ]

        X, self.lbX, self.ubX = [], [], []
        G, self.lbG, self.ubG = [], [], []
        J = 0

        for i in range(self.N):
            x = ca.MX.sym(f"x_{i}", self.nx)
            X.append(x)
            self.lbX.extend(x_min)
            self.ubX.extend(x_max)

            # Dynamics constraint: x_0 = x_init, x_{i+1} = f(x_i, u_i)
            if i == 0:
                G.append(x - x_init_sym)
            else:
                G.append(x - self.system_dynamics(x_prev, u_prev, self.dt))  # noqa: F821
            self.lbG.extend(self.nx * [0])
            self.ubG.extend(self.nx * [0])

            # Stage obstacle-avoidance constraints: ‖obs_j − x[:2]‖ ≥ min_dist
            for j in range(self.max_num_obstacles):
                G.append(
                    ca.norm_2(pred_syms[i][j, :].T - x[:2]) - self.min_distance
                )
                self.lbG.append(0.0)
                self.ubG.append(ca.inf)

            u = ca.MX.sym(f"u_{i}", self.nu)
            X.append(u)
            self.lbX.extend(self.U_MIN)
            self.ubX.extend(self.U_MAX)

            J += (x - x_ref_sym).T @ Q @ (x - x_ref_sym) + u.T @ R @ u

            x_prev, u_prev = x, u

        # Terminal state x_N
        x_N = ca.MX.sym(f"x_{self.N}", self.nx)
        X.append(x_N)
        self.lbX.extend(x_min)
        self.ubX.extend(x_max)

        G.append(x_N - self.system_dynamics(x_prev, u_prev, self.dt))
        self.lbG.extend(self.nx * [0])
        self.ubG.extend(self.nx * [0])

        # Residual Neural Terminal Constraint:
        #   min_j ‖obs_N,j − x_N[:2]‖ − min_dist ≥ LAMBDA · φ(x_N; θ)
        terminal_sdf = ca.mmin(
            ca.sqrt(
                ca.sum2(
                    (pred_syms[self.N] - ca.repmat(x_N[:2].T, self.max_num_obstacles, 1))
                    ** 2
                )
            )
            - self.min_distance
        )
        # Main-network input: robot-relative position + heading
        main_net_input = ca.vcat([x_N[:2] - x_init_sym[:2], x_N[2]]).T
        G.append(terminal_sdf - self.LAMBDA * main_net(main_net_input))
        self.lbG.append(0.0)
        self.ubG.append(ca.inf)

        J += (x_N - x_ref_sym).T @ QN @ (x_N - x_ref_sym)

        # ---- Assemble parameter vector P (symbolic) ----
        P = [x_init_sym, x_ref_sym]
        for sym in pred_syms:
            P.append(ca.vec(sym))
        for layer in main_net.layers:
            P.append(ca.vec(layer["weight"]))
            P.append(ca.vec(layer["bias"]))

        nlp = {
            "f": J,
            "x": ca.vertcat(*X),
            "g": ca.vertcat(*G),
            "p": ca.vertcat(*P),
        }

        # ---- Load (or compile) the solver ----
        self.solver = self._get_solver(nlp, f"rntc_mpc_N{self.N}")

        # ---- Load PyTorch hypernetwork ----
        pkg_share = FindPackageShare("mpc_local_planner").find("mpc_local_planner")
        hypernet_path = os.path.join(pkg_share, "hypernet_weights", "hypernet_rntc.pth")

        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"PyTorch device: {self.torch_device}")

        self.hypernet = Hypernetwork(
            input_size=2, output_size=main_net.num_params()
        ).to(self.torch_device)
        self.hypernet.load_state_dict(
            torch.load(hypernet_path, weights_only=True, map_location=self.torch_device)
        )
        self.hypernet.eval()

        # Run a dummy forward pass to initialise main_net_params
        dummy_sdf = torch.ones((1, 2, sdf_num_cells, sdf_num_cells), device=self.torch_device)
        self.main_net_params = self.hypernet(dummy_sdf).detach().cpu().numpy()

        # ---- Initialise warm-start variables ----
        self.predictions = np.full(
            (self.N + 1, self.max_num_obstacles, 2), self.NAN_SENTINEL
        )
        self.x_opt = np.zeros(len(self.lbX))
        self.lam_x_opt = np.zeros(len(self.lbX))
        self.lam_g_opt = np.zeros(len(self.lbG))

    # ------------------------------------------------------------------ #
    # Obstacle processing + hypernetwork inference                        #
    # ------------------------------------------------------------------ #

    def local_obstacles_callback(self, msg) -> None:
        """Update obstacle predictions and infer main-network weights.

        Steps:
          1. Decode incoming observation (positions + velocities).
          2. Linearly propagate each obstacle over the MPC horizon.
          3. Compute two-channel SDF (at terminal and near-terminal time steps)
             in the robot-centred local frame.
          4. Run the hypernetwork to predict main-network weights.
        """
        obs_shape = (msg.layout.dim[0].size, msg.layout.dim[1].size)
        observation = np.array(msg.data).reshape(obs_shape)
        positions = observation[:, :2]
        velocities = observation[:, 2:]

        # Linear obstacle predictions over the horizon
        self.predictions = np.stack(
            [
                positions + velocities * dt
                for dt in np.arange(0, (self.N + 1) * self.dt, self.dt)
            ],
            axis=0,
        )
        # Replace NaN entries (absent obstacles) with a safe sentinel far from
        # the robot's operating area so NLP distance constraints are trivially met
        self.predictions[np.isnan(self.predictions)] = self.NAN_SENTINEL

        # Count only actually observed obstacles for SDF computation
        num_obstacles = int(np.count_nonzero(~np.isnan(positions[:, 0])))

        # Compute two-channel SDF in the local (robot-centred) frame:
        #   channel 0: SDF at terminal step N   (shapes the terminal constraint)
        #   channel 1: SDF at step N−3          (provides context near the horizon)
        large_val = self.SDF_SIZE  # Background SDF value (no obstacles nearby)
        terminal_sdf = [np.full(self.xy_grid.shape[:2], large_val)]
        stage_sdf = [np.full(self.xy_grid.shape[:2], large_val)]
        robot_pos = np.array(self.x_init[:2])

        for i in range(num_obstacles):
            # Translate obstacle prediction into the local frame
            terminal_local = self.predictions[-1, i, :] - robot_pos
            stage_local = self.predictions[-3, i, :] - robot_pos

            terminal_sdf.append(
                np.linalg.norm(self.xy_grid - terminal_local, axis=-1) - self.min_distance
            )
            stage_sdf.append(
                np.linalg.norm(self.xy_grid - stage_local, axis=-1) - self.min_distance
            )

        terminal_sdf = np.min(np.stack(terminal_sdf, axis=-1), axis=-1)
        stage_sdf = np.min(np.stack(stage_sdf, axis=-1), axis=-1)

        # Run the hypernetwork: SDF grid → main-network weights
        sdf_seq = np.stack([terminal_sdf, stage_sdf], axis=0)
        sdf_tensor = torch.tensor(
            sdf_seq[None, ...], dtype=torch.float, device=self.torch_device
        )
        with torch.no_grad():
            self.main_net_params = self.hypernet(sdf_tensor).cpu().numpy()

    # ------------------------------------------------------------------ #
    # NLP parameter vector (adds main-net weights after predictions)      #
    # ------------------------------------------------------------------ #

    def _build_nlp_params(self) -> list:
        """Extend the base parameter vector with main-network weights."""
        P = super()._build_nlp_params()
        P.append(ca.vec(ca.DM(self.main_net_params)))
        return P


def main(args=None):
    rclpy.init(args=args)
    node = RNTC_MPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
