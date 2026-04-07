import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class DoubleIntegrator2D(dynamics.ControlAndDisturbanceAffineDynamics):
    """Class that implements 2D double integrator model."""

    def __init__(
        self,
        control_mode: Literal["min", "max"] = "max",
        disturbance_mode: Literal["min", "max"] = "min",
        control_space=None,
        disturbance_space=None,
    ):
        if control_space == None:
            control_space = sets.Box(lo=jnp.array([0.0, 0.0]), hi=jnp.array([0.0, 0.0]))
        if disturbance_space == None:
            disturbance_space = sets.Box(
                lo=jnp.array([0.0, 0.0]), hi=jnp.array([0.0, 0.0])
            )
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        """Implements open loop dynamics of a 2D double integrator"""
        A = jnp.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        return A @ state

    def control_jacobian(self, state, time):
        """Calculates control Jacobian of a 2D double integrator"""
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state, time):
        """Calculates disturbance Jacobian of a 2D double integrator"""
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
