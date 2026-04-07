"""
Generate training data for learning-based motion planning with Hamilton-Jacobi reachability.

This module generates datasets of Signed Distance Field (SDF) sequences and
corresponding Hamilton-Jacobi (HJ) value functions for a kinematic unicycle
navigating among dynamic circular obstacles.

The value function encodes the safe reachable set: the set of robot states from
which the robot can avoid all obstacles for all time, given worst-case
disturbances and optimal control. States where vf >= 0 are safe.

Output structure (written under `data_dir`):
    grid/grid.npy        — State-space grid coordinates, shape (Nx, Ny, Nθ, 3)
    sdf/sdf_XXXXXX.npy  — SDF observation sequences, shape (T, Nx, Ny) per sample
    vf/vf_XXXXXX.npy    — HJ value functions, shape (Nx, Ny, Nθ) per sample

External dependency:
    hj_reachability (git submodule) — Hamilton-Jacobi PDE solver and robot dynamics.
    See README.md for setup instructions.
"""

import os
import time
import shutil
from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

import hj_reachability as hj
from hj_reachability.systems import KinematicUnicycle


@jax.jit
def compute_sdf(xy_grid: jnp.ndarray, positions: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """Compute the joint Signed Distance Field (SDF) for a set of circular obstacles.

    For each grid point the SDF value is the distance to the nearest obstacle
    surface (positive outside, negative inside). Multiple obstacles are combined
    by taking the minimum (i.e., the union of obstacle interiors is obstacle-free
    iff the minimum SDF is positive).

    Args:
        xy_grid:   (Nx, Ny, 2) array of 2D grid coordinates.
        positions: (M, 2) array of obstacle centre positions.
        radii:     (M,) array of obstacle radii (includes robot radius inflation).

    Returns:
        (Nx, Ny) array of SDF values over the grid.
    """
    # Broadcast grid points against obstacle centres: shape (Nx, Ny, M, 2)
    diffs = xy_grid[..., None, :] - positions[None, None, :]
    distances = jnp.linalg.norm(diffs, axis=-1)          # (Nx, Ny, M)
    sdf_values = distances - radii[None, None, :]          # signed distance per obstacle
    return jnp.min(sdf_values, axis=-1)                    # union: closest obstacle surface


def get_target_func(
    grid: hj.Grid,
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    radii: jnp.ndarray,
) -> Callable:
    """Build a time-parametrised target function for moving circular obstacles.

    The target function maps a time `t` to the joint SDF of all obstacles at
    that instant, assuming each obstacle moves with constant velocity. It is
    used both to record the SDF history observed by the robot and as the
    dynamic failure set boundary in the HJ PDE solver.

    Args:
        grid:       hj.Grid object defining the state-space discretisation.
        positions:  (M, 2) obstacle positions at t = 0.
        velocities: (M, 2) constant obstacle velocities.
        radii:      (M,) inflated obstacle radii (obstacle radius + robot radius).

    Returns:
        A JIT-compiled callable ``target_func(t) -> (Nx, Ny, 1)`` giving the
        joint SDF at time `t`.
    """
    # Extract the (x, y) slice from the full (x, y, θ) grid; θ is irrelevant
    # for the 2-D obstacle geometry.
    xy_grid = grid.states[..., 0, :2]  # shape (Nx, Ny, 2), θ index fixed at 0

    @partial(jax.jit, static_argnums=())
    def target_func(t: float = 0.0) -> jnp.ndarray:
        """Return joint SDF at time t, extrapolated from initial positions."""
        positions_at_t = positions + t * velocities          # constant-velocity propagation
        sdf = compute_sdf(xy_grid, positions_at_t, radii)
        return sdf[..., None]                                 # add trailing dim for hj API

    return target_func


def clear_directory(directory: str) -> None:
    """Delete all contents of `directory` and recreate it as an empty folder.

    Args:
        directory: Path to the directory to clear.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def generate_data(data_dir: str, clear_old_data: bool = False) -> None:
    """Generate and save SDF sequences and HJ value functions.

    For each sample the function:
    1. Samples a random configuration of 1–4 moving circular obstacles.
    2. Computes the SDF history (T snapshots) for those obstacles.
    3. Solves the HJ PDE backward in time to obtain the safe-set value function.
    4. Saves both arrays alongside a single shared grid file.

    Args:
        data_dir:       Root directory for all output files.
        clear_old_data: If True, wipe and recreate output subdirectories before
                        generating new data. If False, append to existing files.
    """
    sdf_dir  = os.path.join(data_dir, "sdf")
    vf_dir   = os.path.join(data_dir, "vf")
    grid_dir = os.path.join(data_dir, "grid")

    if clear_old_data:
        print("Clearing output directories...")
        for path in [sdf_dir, vf_dir, grid_dir]:
            clear_directory(path)
    else:
        for path in [sdf_dir, vf_dir, grid_dir]:
            os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Hyperparameters
    # ------------------------------------------------------------------ #
    num_samples   = 10000  # total number of (SDF, VF) pairs to generate

    # State-space geometry
    sdf_size      = 8      # side length of the square (x, y) domain in metres
    sdf_num_cells = 100    # grid cells along each spatial axis
    sdf_res       = sdf_size / sdf_num_cells  # spatial resolution (m/cell)
    # θ uses 30 cells; the dimension is periodic so boundary effects are avoided.

    # SDF observation sequence
    sdf_seq_size  = 4      # number of SDF snapshots per sample
    sdf_seq_step  = 0.2    # time between consecutive snapshots (s)

    # Obstacle configuration
    robot_radius  = 0.4    # robot body radius (m), used to inflate obstacle radii
    max_num_obst  = 4      # maximum number of obstacles per sample
    # Obstacle counts are skewed toward more obstacles (harder scenarios).
    obstacle_count_probs = [0.1, 0.2, 0.3, 0.4]

    # Obstacle motion
    theta_bounds  = (-np.pi, np.pi)  # uniform distribution over all headings
    speed_bounds  = (1.0, 1.0)       # constant speed of 1 m/s (fixed for now)
    radius_bounds = (0.5, 0.5)       # obstacle radius before inflation (m)

    # HJ solver time horizon: integrate backward from t=6 s to t=0 s
    initial_time  = 6.0   # start time of backward reachability (s)
    target_time   = 0.0   # end time; value function is evaluated at this time

    # ------------------------------------------------------------------ #
    # Grid and dynamics (created once, shared across all samples)
    # ------------------------------------------------------------------ #

    # 3-D state space: (x, y, θ) with θ periodic on [-π, π]
    # The spatial bounds are inset by half a cell to centre the outer nodes.
    half_extent = (sdf_size - sdf_res) / 2
    state_space = hj.sets.Box(
        lo=jnp.array([-half_extent, -half_extent, -jnp.pi]),
        hi=jnp.array([ half_extent,  half_extent,  jnp.pi]),
    )

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=state_space,
        shape=(sdf_num_cells, sdf_num_cells, 30),
        periodic_dims=2,  # θ dimension is periodic
        boundary_conditions=(
            hj.boundary_conditions.extrapolate,  # x: zero-gradient at boundary
            hj.boundary_conditions.extrapolate,  # y: zero-gradient at boundary
            hj.boundary_conditions.periodic,     # θ: wrap around ±π
        ),
    )

    # Control space: [angular velocity, linear acceleration] bounded at ±0.5
    control_space = hj.sets.Box(
        lo=jnp.array([-0.5, -0.5]),
        hi=jnp.array([ 0.5,  0.5]),
    )

    # Disturbance space: external position disturbance bounded at ±0.1 m/s
    disturbance_space = hj.sets.Box(
        lo=jnp.array([-0.1, -0.1]),
        hi=jnp.array([ 0.1,  0.1]),
    )

    # Kinematic unicycle: maximises control (avoidance) against minimising disturbance
    dynamics = KinematicUnicycle(
        control_mode="max",
        control_space=control_space,
        disturbance_mode="min",
        disturbance_space=disturbance_space,
    )

    # Save the grid once; all samples share the same discretisation
    np.save(os.path.join(grid_dir, "grid.npy"), grid.states)

    # Time points at which SDF observations are recorded (relative to target_time=0)
    sdf_seq_times = np.array(
        [target_time - j * sdf_seq_step for j in range(sdf_seq_size)]
    )

    # ------------------------------------------------------------------ #
    # Sample generation loop
    # ------------------------------------------------------------------ #
    print(f"Generating {num_samples} samples...")

    # Pre-sample obstacle counts for the full dataset so that the distribution
    # across samples is exactly as specified.
    obstacle_counts = np.random.choice(
        range(1, max_num_obst + 1),
        size=num_samples,
        p=obstacle_count_probs,
    )

    for i in range(num_samples):
        num_obstacles = obstacle_counts[i]

        # Random initial positions (uniformly distributed in the domain)
        positions = np.random.uniform(-sdf_size / 2, sdf_size / 2, (num_obstacles, 2))

        # Random constant velocities: speed drawn from speed_bounds, direction uniform
        speeds = np.random.uniform(*speed_bounds, num_obstacles)
        thetas = np.random.uniform(*theta_bounds, num_obstacles)
        velocities = np.column_stack([speeds * np.cos(thetas), speeds * np.sin(thetas)])

        # Inflate obstacle radii to account for the robot body (Minkowski sum)
        radii = np.random.uniform(*radius_bounds, num_obstacles) + robot_radius

        target_func = get_target_func(
            grid=grid,
            positions=jnp.array(positions),
            velocities=jnp.array(velocities),
            radii=jnp.array(radii),
        )

        # Record SDF snapshots across the observation window
        sdf_history = [np.array(target_func(t)[..., 0]) for t in sdf_seq_times]
        sdf = np.stack(sdf_history, axis=0)  # shape: (sdf_seq_size, Nx, Ny)
        np.save(os.path.join(sdf_dir, f"sdf_{i:06d}.npy"), sdf)

        # Configure the HJ PDE solver with the dynamic obstacle set as the
        # failure-set boundary (value function = min(vf, SDF) at each step)
        solver_settings = hj.SolverSettings.with_accuracy(
            accuracy="high",
            hamiltonian_postprocessor=hj.solver.identity,
            value_postprocessor=hj.solver.dynamic_target_func(target_func),
        )

        # Initialise the value function at the initial time using the SDF;
        # replicate over the θ dimension since obstacle geometry is 2-D only.
        initial_values = target_func(initial_time).repeat(grid.shape[-1], axis=-1)

        # Solve the HJ PDE backward from initial_time to target_time
        vf = hj.step(
            solver_settings=solver_settings,
            dynamics=dynamics,
            grid=grid,
            time=initial_time,
            values=initial_values,
            target_time=target_time,
        )

        np.save(os.path.join(vf_dir, f"vf_{i:06d}.npy"), vf)

        if (i + 1) % 5 == 0 or i == num_samples - 1:
            print(f"  {i + 1:>6}/{num_samples} samples completed")

        # Periodically clear JAX's compiled-function cache to prevent unbounded
        # memory growth caused by re-tracing with different array shapes/values.
        if i % 5 == 4:
            jax.clear_caches()

    print("Data generation complete.")


if __name__ == "__main__":
    # Use the first available GPU; set before any JAX initialisation
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_dir = "./data/dynamic_env/circles"

    t_start = time.time()
    generate_data(data_dir=data_dir, clear_old_data=True)
    elapsed = time.time() - t_start

    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"Total time: {hours:02d}h {minutes:02d}m")
