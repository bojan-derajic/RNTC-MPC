"""
NTC-MPC: Neural Terminal Constraint MPC.

This variant directly uses the neural main network output as the terminal
constraint, without the residual SDF multiplier used by RNTC-MPC:

    φ(x_N − x_init; θ) ≥ 0

where φ is a small neural network whose weights θ are predicted by the
hypernetwork from the local SDF.  The main-network output is not bounded
below by 1 (linear final activation), so negative values mean "unsafe" and
the NLP must push the terminal state into a safe region where φ ≥ 0.

This formulation serves as a baseline for the RNTC-MPC comparison.

Architecture differences vs. RNTC-MPC:
    - Main-network final activation: linear  (RNTC uses elu_plus_1)
    - Terminal constraint: φ(x_N) ≥ 0       (RNTC uses SDF ≥ 1.4·φ(x_N))
    - Separate pre-trained weights (hypernet_ntc.pth)
"""

import os
import numpy as np
import torch
import rclpy
from launch_ros.substitutions import FindPackageShare
import casadi as ca

from mpc_local_planner.base_mpc import BaseMPC
from mpc_local_planner.models import MainNetwork, Hypernetwork


class NTC_MPC(BaseMPC):
    """MPC with direct Neural Terminal Constraint (NTC).

    The hypernetwork predicts main-network weights from the local SDF.
    The terminal constraint enforces that the main-network output is non-negative
    at the predicted terminal state.
    """

    NODE_NAME = "ntc_mpc"

    SDF_SIZE: float = 8.0
    SDF_RES: float = 0.08

    # Same hidden architecture as RNTC, but linear output (no lower bound of 1)
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
            (1, "linear"),  # Unbounded output; constraint is output ≥ 0
        ],
    }

    def configure_mpc(self) -> None:
        """Build the NTC-MPC NLP and load (or compile) the solver."""
        sdf_num_cells = int(self.SDF_SIZE / self.SDF_RES)

        axis = np.linspace(
            -(self.SDF_SIZE - self.SDF_RES) / 2,
            (self.SDF_SIZE - self.SDF_RES) / 2,
            sdf_num_cells,
        )
        self.xy_grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)

        main_net = MainNetwork(self.MAIN_NET_CONFIG)

        x_min = [-ca.inf, -ca.inf, -ca.inf]
        x_max = [ca.inf, ca.inf, ca.inf]
        Q = ca.diag(self.Q_DIAG)
        QN = ca.diag(self.QN_DIAG)
        R = ca.diag(self.R_DIAG)

        x_init_sym = ca.MX.sym("x_init", self.nx)
        x_ref_sym = ca.MX.sym("x_ref", self.nx)

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

            if i == 0:
                G.append(x - x_init_sym)
            else:
                G.append(x - self.system_dynamics(x_prev, u_prev, self.dt))  # noqa: F821
            self.lbG.extend(self.nx * [0])
            self.ubG.extend(self.nx * [0])

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

        x_N = ca.MX.sym(f"x_{self.N}", self.nx)
        X.append(x_N)
        self.lbX.extend(x_min)
        self.ubX.extend(x_max)

        G.append(x_N - self.system_dynamics(x_prev, u_prev, self.dt))
        self.lbG.extend(self.nx * [0])
        self.ubG.extend(self.nx * [0])

        # Neural terminal constraint: φ(x_N − x_init; θ) ≥ 0
        main_net_input = ca.vcat([x_N[:2] - x_init_sym[:2], x_N[2]]).T
        G.append(main_net(main_net_input))
        self.lbG.append(0.0)
        self.ubG.append(ca.inf)

        J += (x_N - x_ref_sym).T @ QN @ (x_N - x_ref_sym)

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

        self.solver = self._get_solver(nlp, f"ntc_mpc_N{self.N}")

        # ---- Load PyTorch hypernetwork ----
        pkg_share = FindPackageShare("mpc_local_planner").find("mpc_local_planner")
        hypernet_path = os.path.join(pkg_share, "hypernet_weights", "hypernet_ntc.pth")

        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"PyTorch device: {self.torch_device}")

        self.hypernet = Hypernetwork(
            input_size=2, output_size=main_net.num_params()
        ).to(self.torch_device)
        self.hypernet.load_state_dict(
            torch.load(hypernet_path, weights_only=True, map_location=self.torch_device)
        )
        self.hypernet.eval()

        dummy_sdf = torch.ones((1, 2, sdf_num_cells, sdf_num_cells), device=self.torch_device)
        self.main_net_params = self.hypernet(dummy_sdf).detach().cpu().numpy()

        self.predictions = np.full(
            (self.N + 1, self.max_num_obstacles, 2), self.NAN_SENTINEL
        )
        self.x_opt = np.zeros(len(self.lbX))
        self.lam_x_opt = np.zeros(len(self.lbX))
        self.lam_g_opt = np.zeros(len(self.lbG))

    def local_obstacles_callback(self, msg) -> None:
        """Update obstacle predictions and infer main-network weights from the SDF."""
        obs_shape = (msg.layout.dim[0].size, msg.layout.dim[1].size)
        observation = np.array(msg.data).reshape(obs_shape)
        positions = observation[:, :2]
        velocities = observation[:, 2:]

        self.predictions = np.stack(
            [
                positions + velocities * dt
                for dt in np.arange(0, (self.N + 1) * self.dt, self.dt)
            ],
            axis=0,
        )
        self.predictions[np.isnan(self.predictions)] = self.NAN_SENTINEL

        num_obstacles = int(np.count_nonzero(~np.isnan(positions[:, 0])))

        large_val = self.SDF_SIZE
        terminal_sdf = [np.full(self.xy_grid.shape[:2], large_val)]
        stage_sdf = [np.full(self.xy_grid.shape[:2], large_val)]
        robot_pos = np.array(self.x_init[:2])

        for i in range(num_obstacles):
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

        sdf_seq = np.stack([terminal_sdf, stage_sdf], axis=0)
        sdf_tensor = torch.tensor(
            sdf_seq[None, ...], dtype=torch.float, device=self.torch_device
        )
        with torch.no_grad():
            self.main_net_params = self.hypernet(sdf_tensor).cpu().numpy()

    def _build_nlp_params(self) -> list:
        """Extend the base parameter vector with main-network weights."""
        P = super()._build_nlp_params()
        P.append(ca.vec(ca.DM(self.main_net_params)))
        return P


def main(args=None):
    rclpy.init(args=args)
    node = NTC_MPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
