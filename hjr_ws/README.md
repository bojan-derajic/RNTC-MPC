# HJR Workspace

Workspace for applying Hamilton-Jacobi (HJ) reachability analysis to generate data necessary for the RNTC-MPC method.

The tool produces paired datasets of:
- **SDF sequences** — snapshots of the joint Signed Distance Field of moving
  circular obstacles observed by the robot over a short time window.
- **HJ value functions** — the solution to the backward-reachable HJ PDE,
  encoding the safe set for a kinematic unicycle under worst-case disturbances.

---

## How it works

For each sample the generator:

1. Samples a random configuration of 1–4 circular obstacles with random
   positions and constant velocities.
2. Records `T = 4` SDF snapshots of the obstacle field at evenly spaced times.
3. Solves the HJ PDE backward from `t = 6 s` to `t = 0 s` with the dynamic
   obstacle set as the failure-set boundary, yielding the value function.
4. Saves the SDF sequence and value function as NumPy arrays.

The robot is modelled as a **kinematic unicycle** on a 3-D state space
`(x, y, θ)` with control bounds `±0.5` and external disturbances `±0.1 m/s`.
The spatial domain is an 8 m × 8 m square discretised into 100 × 100 cells;
the heading dimension uses 30 cells with periodic boundary conditions.

---

## Dependencies

### External dependency — `hj_reachability` (git submodule)

The HJ PDE solver and robot dynamics models are provided by the
`hj_reachability` package, included as a git submodule:

```
hj_reachability/   ← do NOT edit; this is an external repository
```

Repository: <https://github.com/bojan-derajic/hj_reachability>

Initialise it after cloning this repo:

```bash
git submodule update --init
```

### Python packages

All Python dependencies are pinned in `requirements.txt`.  Key packages:

| Package          | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `jax` + `jaxlib` | JIT-compiled numerical computation (GPU via CUDA 12) |
| `numpy`          | Array manipulation and `.npy` file I/O               |
| `matplotlib`     | Visualisation                                        |
| `flax`, `optax`  | Neural-network training (downstream use)             |
| `scipy`          | Scientific utilities                                 |

---

## Setup

### Option A — VS Code Dev Container (recommended)

The repository ships with a pre-configured Dev Container that gives you a
GPU-enabled Python environment with all dependencies installed.

1. Install [Docker](https://docs.docker.com/get-docker/) and the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Open the repository in VS Code with the
   [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   extension installed.
3. Press `F1` → **Dev Containers: Reopen in Container**.
   VS Code will build the image and install `hj_reachability` automatically.

### Option B — Docker (headless / CI)

```bash
# 1. Initialise the submodule
git submodule update --init

# 2. Build the Docker image (only needed once, or after requirements change)
./scripts/build_docker_image.sh

# 3. Generate data (runs inside the container, writes to ./data/ on the host)
./scripts/generate_data.sh
```

### Option C — Local Python environment

```bash
# 1. Initialise the submodule
git submodule update --init

# 2. Create and activate a virtual environment (Python 3.13 recommended)
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install hj_reachability in editable mode
pip install -e hj_reachability

# 5. Generate data
python src/generate_data.py
```

> **Note:** JAX's CUDA 12 wheels are installed automatically via
> `requirements.txt`.  If you are running on CPU only, replace the
> `jax-cuda12-*` lines with `jax[cpu]`.

---

## Usage

```bash
python src/generate_data.py
```

Output is written to `./data/dynamic_env/circles/`:

```
data/dynamic_env/circles/
├── grid/
│   └── grid.npy          # Shared grid coordinates, shape (Nx, Ny, Nθ, 3)
├── sdf/
│   ├── sdf_000000.npy    # SDF sequence for sample 0, shape (T, Nx, Ny)
│   ├── sdf_000001.npy
│   └── ...               # 10 000 files total
└── vf/
    ├── vf_000000.npy     # Value function for sample 0, shape (Nx, Ny, Nθ)
    ├── vf_000001.npy
    └── ...               # 10 000 files total
```

Where `Nx = Ny = 100`, `Nθ = 30`, and `T = 4`.

To adjust the number of samples or other hyperparameters, edit the
constants at the top of the `generate_data()` function in
[`src/generate_data.py`](src/generate_data.py).

---

## Generated visualisations

The `visuals/` directory contains example outputs:

| File                                                          | Description                            |
| ------------------------------------------------------------- | -------------------------------------- |
| `visuals/animations/dynamic_sdf.gif`                          | SDF evolving as obstacles move         |
| `visuals/animations/moving_obstacles.gif`                     | Obstacle trajectories                  |
| `visuals/animations/kinematic_unicycle_and_moving_circle.gif` | Robot navigating a moving obstacle     |
| `visuals/figures/two_obstacles_joint_vf.png`                  | Value function slice for two obstacles |

---

## Project structure

```
.
├── .devcontainer/
│   └── devcontainer.json       # VS Code Dev Container configuration
├── scripts/
│   ├── build_docker_image.sh   # Build the Docker image
│   └── generate_data.sh        # Run data generation inside Docker
├── src/
│   └── generate_data.py        # Main data-generation script
├── visuals/
│   ├── animations/             # Example GIF animations
│   └── figures/                # Example static figures
├── hj_reachability/            # External dependency (git submodule — do not edit)
├── Dockerfile                  # Container definition
├── requirements.txt            # Pinned Python dependencies
└── README.md
```
