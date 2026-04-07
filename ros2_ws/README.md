# RNTC-MPC: Residual Neural Terminal Constraint MPC

Evaluation framework for the **Residual Neural Terminal Constraint MPC (RNTC-MPC)** method for dynamic obstacle avoidance with mobile robots.

The framework runs a Jackal robot in Gazebo Sim (Harmonic / Jazzy) and benchmarks five collision-avoidance MPC variants across 100 randomly generated dynamic obstacle scenarios.

---

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Architecture Details](#architecture-details)

---

## Overview

The robot (Jackal, unicycle kinematics) must navigate from **start (−2, 0)** to **goal (13, 0)** [m] while avoiding 6 moving cylindrical obstacles in a 16 × 10 m arena.

Each MPC variant solves a Nonlinear Programme (NLP) at 20 Hz with:
- **Horizon lengths** N ∈ {5, 10, 20, 30}
- **Sampling time** dt = 0.1 s
- **Solver**: IPOPT + HSL MA57

All NLPs are compiled ahead of time to C shared libraries via CasADi + GCC for fast online evaluation.

---

## Package Structure

```
ros2_ws/src/
├── simulation_bringup/        # Gazebo world, robot URDF, ROS 2 launch file
│   ├── launch/jackal_robot.launch.py
│   ├── urdf/jackal_robot.urdf.xacro
│   ├── worlds/{empty,warehouse,house}.world
│   └── config/{control,ekf,ros_gz_bridge}.yaml
│
├── mpc_local_planner/         # Five MPC variants
│   ├── mpc_local_planner/
│   │   ├── base_mpc.py        # Shared base class (ROS2 interface, solver utils)
│   │   ├── rntc_mpc.py        # Residual Neural Terminal Constraint MPC ★
│   │   ├── ntc_mpc.py         # Neural Terminal Constraint MPC
│   │   ├── dcbf_mpc.py        # Discrete-time CBF MPC
│   │   ├── vo_mpc.py          # Velocity Obstacle MPC
│   │   ├── sdf_mpc.py         # SDF distance-constraint MPC (baseline)
│   │   └── models/
│   │       ├── main_network.py   # CasADi symbolic main network
│   │       └── hypernetwork.py   # PyTorch hypernetwork (CNN)
│   └── hypernet_weights/
│       ├── hypernet_rntc.pth  # Pre-trained weights for RNTC
│       └── hypernet_ntc.pth   # Pre-trained weights for NTC
│
├── dynamic_env/               # Automated evaluation loop + obstacle manager
│   ├── dynamic_env/dynamic_env.py
│   └── assets/
│       ├── scenarios/scenarios.json   # 100 pre-generated test scenarios
│       ├── sdf_models/                # Pre-generated Gazebo obstacle SDF files
│       └── xacro_models/              # Obstacle xacro template
│
└── visualize_vf/              # Optional: real-time value-function visualiser
    └── visualize_vf/visualize_vf.py
```

---

## Prerequisites

### System
- Ubuntu 24.04
- ROS 2 Jazzy (desktop-full)
- Gazebo Harmonic (`gz-sim`)
- GCC (for CasADi NLP compilation)
- Python ≥ 3.10

### Python packages
```
casadi==3.6.7
torch==2.6.0
```
See `requirements.txt` for the full list.

### HSL MA57 Solver (required)

The IPOPT solver inside CasADi uses the **HSL MA57** sparse direct linear solver for best performance.  This requires a **free academic licence** from the Numerical Algorithms Group (NAG):

1. Register and download the source archive at:  
   <https://licences.stfc.ac.uk/product/coin-hsl>

2. Extract the archive into `ThirdParty-HSL/` following the instructions at:  
   <https://github.com/coin-or-tools/ThirdParty-HSL>

3. Build and install the library:
   ```bash
   ./scripts/install_hsl_lib.sh
   ```
   This runs `./configure && make && sudo make install` in `ThirdParty-HSL/`
   and installs `libcoinhsl.so` to `/usr/local/lib/`.

---

## Installation

### Option 1 — Docker (recommended)

```bash
# Build the image
./scripts/build_docker_image.sh

# Run experiments inside the container
./scripts/run_sim_experiments.sh
```

The `run_sim_experiments.sh` script mounts the workspace, installs the HSL library, and launches the full evaluation pipeline.

### Option 2 — Native

```bash
# 1. Install ROS 2 Jazzy and system dependencies
#    (see Dockerfile for the full apt list)

# 2. Install Python dependencies
pip3 install --break-system-packages -r requirements.txt

# 3. Install HSL MA57 (see above)
./scripts/install_hsl_lib.sh

# 4. Build the workspace
cd ~/ros2_ws
colcon build --symlink-install

# 5. Source the workspace
source install/setup.bash
```

---

## Usage

### Launch the simulation

```bash
ros2 launch simulation_bringup jackal_robot.launch.py
```

**Launch arguments:**

| Argument      | Default | Description                                  |
| ------------- | ------- | -------------------------------------------- |
| `world_model` | `empty` | Gazebo world (`empty`, `warehouse`, `house`) |
| `gazebo_gui`  | `false` | Show the Gazebo GUI                          |
| `rviz`        | `false` | Launch RViz for visualisation                |

Example with GUI:
```bash
ros2 launch simulation_bringup jackal_robot.launch.py gazebo_gui:=true rviz:=true
```

The launch file automatically starts the `dynamic_env` node (after a 10 s delay for Gazebo to initialise), which then runs the full evaluation loop.

### Run a single MPC variant manually

First compile the NLP solver (one-time per variant × horizon):
```bash
ros2 run mpc_local_planner rntc_mpc \
  --ros-args -p compile_nlp_solver:=true \
             -p horizon_length:=20 \
             -p use_sim_time:=true
```

Once the `.so` file exists in `share/mpc_local_planner/assets/casadi_compiled/`,
subsequent runs load it directly:
```bash
ros2 run mpc_local_planner rntc_mpc \
  --ros-args -p horizon_length:=20 \
             -p use_sim_time:=true
```

### Visualise the value function (optional)

```bash
ros2 run visualize_vf visualize_vf
```

Publishes `/vf` (mono8 Image, 100×100) showing the local-frame RNTC value function.
View it in RViz with the Image display or using `rqt_image_view`.

### Regenerate obstacle SDF models

Only needed if the obstacle geometry changes:
```bash
ros2 run dynamic_env dynamic_env \
  --ros-args -p generate_sdf_models:=true
```

---

## Results

Evaluation results are written to `./simulation_results/` as JSON files named:
```
{world_model}_{robot_model}_{mpc_type}_{N}.json
```

Each file contains:
```json
{
  "world_model":       "empty",
  "robot_model":       "jackal_robot",
  "mpc_type":          "rntc_mpc",
  "N":                 "20",
  "success":           [1, 0, 1, ...],
  "success_rate":      0.92,
  "travel_time":       [15.3, null, 18.2, ...],
  "mean_travel_time":  16.8,
  "mean_mpc_opt_time": [25.3, null, ...],
  "timed_out":         [0, 0, 1, ...],
  "timed_out_rate":    0.05,
  "position_log":      [[[x, y], ...], ...]
}
```

---

## Architecture Details

### ROS 2 Topics

| Topic                | Type                         | Direction      | Description                             |
| -------------------- | ---------------------------- | -------------- | --------------------------------------- |
| `/odometry`          | `nav_msgs/Odometry`          | Gazebo → ROS   | Robot pose (via EKF)                    |
| `/cmd_vel`           | `geometry_msgs/TwistStamped` | ROS → Gazebo   | MPC velocity command                    |
| `/local_obstacles`   | `std_msgs/Float64MultiArray` | Env → MPC      | 4×4 matrix [px, py, vx, vy]             |
| `/goal_pose`         | `geometry_msgs/PoseStamped`  | Env → MPC      | Navigation goal                         |
| `/predicted_path`    | `nav_msgs/Path`              | MPC → RViz     | Predicted trajectory over horizon       |
| `/mpc_opt_time`      | `std_msgs/Float32`           | MPC → Env      | Solve time [ms]                         |
| `/contact`           | `ros_gz_interfaces/Contacts` | Gazebo → ROS   | Collision detection                     |
| `/obstacle_pose`     | `geometry_msgs/PoseArray`    | Gazebo → ROS   | All entity poses (ground, robot, obst.) |
| `/cmd_vel_000`…`009` | `geometry_msgs/Twist`        | Env → Gazebo   | Per-obstacle velocity commands          |
| `/vf`                | `sensor_msgs/Image`          | VF node → RViz | Value-function image (optional)         |

### NLP Decision Variable Layout

The CasADi NLP interleaves state and control variables:
```
[x_0 (nx), u_0 (nu), x_1 (nx), u_1 (nu), ..., x_{N-1} (nx), u_{N-1} (nu), x_N (nx)]
```
with `nx = 3` (x, y, yaw) and `nu = 2` (linear velocity, angular velocity).

### Solver Compilation

At first run with `compile_nlp_solver:=true`, CasADi generates optimised C code
for the NLP and GCC compiles it to a `.so` shared library stored in:
```
share/mpc_local_planner/assets/casadi_compiled/{variant}_N{horizon}.so
```
Compilation takes 45–90 seconds. Subsequent runs load the `.so` directly.

---

## Glossary

| Term     | Definition                                 |
| -------- | ------------------------------------------ |
| **RNTC** | Residual Neural Terminal Constraint        |
| **NTC**  | Neural Terminal Constraint                 |
| **DCBF** | Discrete-time Control Barrier Function     |
| **VO**   | Velocity Obstacle                          |
| **SDF**  | Signed Distance Function                   |
| **MPC**  | Model Predictive Control                   |
| **NLP**  | Nonlinear Programme                        |
| **HSL**  | Harwell Subroutine Library (provides MA57) |
