import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class KinematicUnicycle(dynamics.ControlAndDisturbanceAffineDynamics):
    """Class that implements the kinematic unicycle model."""

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
        """Implements open loop dynamics of a kinematic unicycle model"""
        return jnp.zeros(shape=(3))

    def control_jacobian(self, state, time):
        """Calculates control Jacobian of a kinematic unicycle model"""
        theta = state[2]
        return jnp.array([[jnp.cos(theta), 0.0], [jnp.sin(theta), 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state, time):
        """Calculates disturbance Jacobian of a kinematic unicycle model"""
        theta = state[2]
        return jnp.array([[jnp.cos(theta), 0.0], [jnp.sin(theta), 0.0], [0.0, 1.0]])
