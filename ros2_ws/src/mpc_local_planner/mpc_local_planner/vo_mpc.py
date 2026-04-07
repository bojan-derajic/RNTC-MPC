"""
VO-MPC: Velocity Obstacle MPC.

The Velocity Obstacle (VO) set for agent A w.r.t. obstacle B is the set of
relative velocities that would lead to a collision within a finite time horizon.
The complementary set defines safe relative velocities.

For a circular obstacle of combined radius r the constraint per step is:

    α ≥ β

where:
    p_rel = obs_j − x[:2]             (relative position vector)
    v_A   = [v cos θ, v sin θ]        (robot absolute velocity in world frame)
    v_rel = v_A − v_obs_j             (relative velocity)
    α     = arccos(p̂_rel · v̂_rel)    (angle between relative position and velocity)
    β     = arcsin(r / ‖p_rel‖)       (half-angle of the VO cone)

Enforcing α ≥ β means the relative velocity vector points outside the VO cone,
i.e., the robot is steering away from a potential collision.

NLP structure:
    Decision variables: [x_0, u_0, x_1, u_1, …, x_N]
    Parameters:         [x_init, x_ref, velocities, predictions_{0..N}]
    Objective:          standard tracking cost
    Constraints:
        x_0 = x_init,  x_{i+1} = f(x_i, u_i)
        VO:  α_i,j ≥ β_i,j  for all stages i and obstacles j
"""

import numpy as np
import casadi as ca
import rclpy

from mpc_local_planner.base_mpc import BaseMPC


class VO_MPC(BaseMPC):
    """MPC with Velocity Obstacle (VO) constraints.

    At each time step the robot's relative velocity w.r.t. each obstacle must
    lie outside the velocity obstacle cone.
    """

    NODE_NAME = "vo_mpc"

    # Small regularisation added to norms to avoid division by zero in CasADi
    _NORM_EPS: float = 1e-6

    # Default obstacle velocity used when no measurement is available.
    # A small non-zero value avoids degenerate geometry in the VO constraint.
    _DEFAULT_OBS_VX: float = -0.01
    _DEFAULT_OBS_VY: float = 0.01

    def configure_mpc(self) -> None:
        """Build the VO-MPC NLP and load (or compile) the solver."""
        x_min = [-ca.inf, -ca.inf, -ca.inf]
        x_max = [ca.inf, ca.inf, ca.inf]
        Q = ca.diag(self.Q_DIAG)
        QN = ca.diag(self.QN_DIAG)
        R = ca.diag(self.R_DIAG)

        x_init_sym = ca.MX.sym("x_init", self.nx)
        x_ref_sym = ca.MX.sym("x_ref", self.nx)

        # Obstacle velocities are parameters (constant over the horizon)
        vel_sym = ca.MX.sym("velocities", self.max_num_obstacles, 2)

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

            u = ca.MX.sym(f"u_{i}", self.nu)
            X.append(u)
            self.lbX.extend(self.U_MIN)
            self.ubX.extend(self.U_MAX)

            # VO constraints for each obstacle at step i
            for j in range(self.max_num_obstacles):
                p_rel = pred_syms[i][j, :].T - x[:2]   # relative position

                # Robot velocity in world frame at step i
                v_robot = ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]))
                v_obs = vel_sym[j, :].T
                v_rel = v_robot - v_obs                  # relative velocity

                # Normalised vectors (regularised to avoid zero-division)
                p_unit = p_rel / (ca.norm_2(p_rel) + self._NORM_EPS)
                v_unit = v_rel / (ca.norm_2(v_rel) + self._NORM_EPS)
                dist = ca.norm_2(p_rel)

                # α: angle between relative position and relative velocity
                dot_clamped = ca.fmax(-1.0 + 1e-3, ca.fmin(1.0 - 1e-3,
                                                             ca.dot(p_unit, v_unit)))
                alpha = ca.acos(dot_clamped)

                # β: half-angle of the VO cone (arcsin of the ratio r / dist)
                sin_clamped = ca.fmax(-1.0 + 1e-3, ca.fmin(1.0 - 1e-3,
                                                             self.min_distance / dist))
                beta = ca.asin(sin_clamped)

                # Safety: relative velocity must point outside the VO cone
                G.append(alpha - beta)
                self.lbG.append(0.0)
                self.ubG.append(ca.inf)

            J += (x - x_ref_sym).T @ Q @ (x - x_ref_sym) + u.T @ R @ u

            x_prev, u_prev = x, u

        x_N = ca.MX.sym(f"x_{self.N}", self.nx)
        X.append(x_N)
        self.lbX.extend(x_min)
        self.ubX.extend(x_max)

        G.append(x_N - self.system_dynamics(x_prev, u_prev, self.dt))
        self.lbG.extend(self.nx * [0])
        self.ubG.extend(self.nx * [0])

        J += (x_N - x_ref_sym).T @ QN @ (x_N - x_ref_sym)

        # Note: vel_sym is placed BEFORE predictions in P (matches mpc_callback order)
        P = [x_init_sym, x_ref_sym, ca.vec(vel_sym)]
        for sym in pred_syms:
            P.append(ca.vec(sym))

        nlp = {
            "f": J,
            "x": ca.vertcat(*X),
            "g": ca.vertcat(*G),
            "p": ca.vertcat(*P),
        }

        self.solver = self._get_solver(nlp, f"vo_mpc_N{self.N}")

        # Initialise obstacle velocities with a small default (avoids zero-norm issues)
        self.velocities = np.full((self.max_num_obstacles, 2), 0.0)
        self.velocities[:, 0] = self._DEFAULT_OBS_VX
        self.velocities[:, 1] = self._DEFAULT_OBS_VY

        self.predictions = np.full(
            (self.N + 1, self.max_num_obstacles, 2), self.NAN_SENTINEL
        )
        self.x_opt = np.zeros(len(self.lbX))
        self.lam_x_opt = np.zeros(len(self.lbX))
        self.lam_g_opt = np.zeros(len(self.lbG))

    def local_obstacles_callback(self, msg) -> None:
        """Update obstacle predictions and measured velocities.

        Obstacles with NaN positions get the default velocity so the VO
        constraint geometry stays well-defined.
        """
        obs_shape = (msg.layout.dim[0].size, msg.layout.dim[1].size)
        observation = np.array(msg.data).reshape(obs_shape)
        positions = observation[:, :2]
        measured_vel = observation[:, 2:]

        self.predictions = np.stack(
            [
                positions + measured_vel * dt
                for dt in np.arange(0, (self.N + 1) * self.dt, self.dt)
            ],
            axis=0,
        )
        self.predictions[np.isnan(self.predictions)] = self.NAN_SENTINEL

        # Reset velocities to the default then overwrite with measurements
        self.velocities = np.full((self.max_num_obstacles, 2), 0.0)
        self.velocities[:, 0] = self._DEFAULT_OBS_VX
        self.velocities[:, 1] = self._DEFAULT_OBS_VY
        valid_mask = ~np.isnan(measured_vel)
        self.velocities[valid_mask] = measured_vel[valid_mask]

    def _build_nlp_params(self) -> list:
        """Extend parameter vector with obstacle velocities (before predictions)."""
        # Order must match configure_mpc: [x_init, x_ref, velocities, predictions...]
        P = [self.x_init, self.x_ref, ca.vec(ca.DM(self.velocities))]
        for i in range(self.N + 1):
            P.append(ca.vec(ca.DM(self.predictions[i])))
        return P


def main(args=None):
    rclpy.init(args=args)
    node = VO_MPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
