"""
DCBF-MPC: Discrete-time Control Barrier Function MPC.

Instead of a neural terminal constraint, safety is enforced via a monotonic
decrease condition on the signed-distance function (SDF) along the trajectory.
This is the discrete-time CBF condition:

    SDF(x_i, obs_i) − (1 − γ) · SDF(x_{i-1}, obs_{i-1}) ≥ 0   ∀ i ∈ {1, …, N}

where γ ∈ (0, 1) is the decay rate.  Intuitively, the robot is allowed to get
slightly closer to an obstacle each step, but the SDF may not decrease faster
than the geometric rate (1 − γ).

This formulation is purely geometry-based (no neural network) and serves as a
classical baseline in the comparison.

NLP structure:
    Decision variables: [x_0, u_0, x_1, u_1, …, x_N]
    Parameters:         [x_init, x_ref, predictions_{0..N}]
    Objective:          standard tracking cost
    Constraints:
        x_0 = x_init,  x_{i+1} = f(x_i, u_i)
        DCBF: SDF_i − (1−γ)·SDF_{i-1} ≥ 0  for i = 1 … N  (per obstacle)
"""

import numpy as np
import casadi as ca
import rclpy

from mpc_local_planner.base_mpc import BaseMPC


class DCBF_MPC(BaseMPC):
    """MPC with Discrete-time Control Barrier Function (DCBF) constraints.

    The SDF between consecutive time steps is constrained to decrease no
    faster than the rate determined by the decay parameter gamma.
    """

    NODE_NAME = "dcbf_mpc"

    # CBF decay rate γ ∈ (0,1).  Smaller γ → slower allowed SDF decrease
    # (more conservative).  Larger γ → faster decrease allowed (less conservative).
    GAMMA: float = 0.3

    def configure_mpc(self) -> None:
        """Build the DCBF-MPC NLP and load (or compile) the solver."""
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

            if i > 0:
                # DCBF constraint for each obstacle between steps i-1 and i:
                #   SDF(x_i, obs_i) ≥ (1 − γ) · SDF(x_{i-1}, obs_{i-1})
                for j in range(self.max_num_obstacles):
                    sdf_i = (
                        ca.norm_2(pred_syms[i][j, :].T - x[:2]) - self.min_distance
                    )
                    sdf_prev = (
                        ca.norm_2(pred_syms[i - 1][j, :].T - x_prev[:2])  # noqa: F821
                        - self.min_distance
                    )
                    G.append(sdf_i - (1.0 - self.GAMMA) * sdf_prev)
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

        # DCBF constraint at the terminal step N
        for j in range(self.max_num_obstacles):
            sdf_N = (
                ca.norm_2(pred_syms[self.N][j, :].T - x_N[:2]) - self.min_distance
            )
            sdf_Nm1 = (
                ca.norm_2(pred_syms[self.N - 1][j, :].T - x_prev[:2])
                - self.min_distance
            )
            G.append(sdf_N - (1.0 - self.GAMMA) * sdf_Nm1)
            self.lbG.append(0.0)
            self.ubG.append(ca.inf)

        J += (x_N - x_ref_sym).T @ QN @ (x_N - x_ref_sym)

        P = [x_init_sym, x_ref_sym]
        for sym in pred_syms:
            P.append(ca.vec(sym))

        nlp = {
            "f": J,
            "x": ca.vertcat(*X),
            "g": ca.vertcat(*G),
            "p": ca.vertcat(*P),
        }

        self.solver = self._get_solver(nlp, f"dcbf_mpc_N{self.N}")

        self.predictions = np.full(
            (self.N + 1, self.max_num_obstacles, 2), self.NAN_SENTINEL
        )
        self.x_opt = np.zeros(len(self.lbX))
        self.lam_x_opt = np.zeros(len(self.lbX))
        self.lam_g_opt = np.zeros(len(self.lbG))

    def local_obstacles_callback(self, msg) -> None:
        """Update obstacle predictions by linear propagation over the horizon."""
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


def main(args=None):
    rclpy.init(args=args)
    node = DCBF_MPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
