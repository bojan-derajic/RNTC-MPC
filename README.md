# RNTC-MPC

> This repo contains an unofficial implementation of the **Residual Neural Terminal Constraint MPC (RNTC-MPC)** method for safe navigation of mobile robots in dynamic environments, as presented at CoRL 2025.

---

## Paper

**Residual Neural Terminal Constraints for Model Predictive Control**  
B. Derajic et al., *Proceedings of Machine Learning Research (CoRL 2025)*, vol. 305, 2025.

| Resource                   | Link                                                                                                      |
| -------------------------- | --------------------------------------------------------------------------------------------------------- |
| CoRL 2025 PMLR Proceedings | [derajic25a.pdf](https://raw.githubusercontent.com/mlresearch/v305/main/assets/derajic25a/derajic25a.pdf) |
| arXiv preprint             | [arXiv:2508.03428](https://arxiv.org/pdf/2508.03428v1)                                                    |

---

## Overview

RNTC-MPC learns a *residual correction* on top of a signed-distance-function (SDF) term to approximate the Hamilton-Jacobi (HJ) value function for a kinematic unicycle robot. At runtime the approximated value function is used as a **terminal safety constraint** inside a Model Predictive Control (MPC) optimisation problem, enabling safe robot navigation.

The repository is split into three independent workspaces that together cover the full pipeline — from offline data generation and model training to closed-loop simulation evaluation.

---

## Repository Structure

```
RNTC-MPC/
├── hjr_ws/        # HJ reachability data generation
├── pytorch_ws/    # Hypernetwork model training
└── ros2_ws/       # ROS 2 / Gazebo simulation & evaluation
```

---

## Workspaces

### 1. `hjr_ws` — HJ Reachability Data Generation

[![README](https://img.shields.io/badge/README-hjr__ws-blue)](hjr_ws/README.md)

Generates paired training datasets using Hamilton-Jacobi reachability analysis:

- **SDF sequences** — snapshots of the joint Signed Distance Field of 1–4 moving circular obstacles observed over a short time window.
- **HJ value functions** — solutions to the backward-reachable HJ PDE encoding the safe set for a kinematic unicycle under worst-case disturbances.

The spatial domain is an 8 m × 8 m square (100 × 100 × 30 cells), solved with JAX on GPU via the [`hj_reachability`](https://github.com/bojan-derajic/hj_reachability) library (included as a git submodule). By default, 10 000 samples are generated.

See [hjr_ws/README.md](hjr_ws/README.md) for setup and usage instructions.

---

### 2. `pytorch_ws` — Hypernetwork Training

[![README](https://img.shields.io/badge/README-pytorch__ws-blue)](pytorch_ws/README.md)

Trains the RNTC hypernetwork model in PyTorch using the data produced by `hjr_ws`:

- A **CNN hypernetwork** maps an SDF sequence to the weights of a small MLP on the fly.
- The **MLP (MainNetwork)** predicts a residual correction on top of the current-time SDF slice to approximate the HJ value function.
- An NTC baseline (direct value-function prediction without the residual) is also provided for comparison.
- Training is tracked with **MLflow** and supports three loss functions: MSE, RWMSE, and CME.

See [pytorch_ws/README.md](pytorch_ws/README.md) for architecture details, configuration reference, and training instructions.

---

### 3. `ros2_ws` — ROS 2 Simulation & Evaluation

[![README](https://img.shields.io/badge/README-ros2__ws-blue)](ros2_ws/README.md)

Evaluates five MPC variants for dynamic obstacle avoidance in Gazebo Harmonic / ROS 2 Jazzy:

| Variant    | Description                                            |
| ---------- | ------------------------------------------------------ |
| `rntc_mpc` | Residual Neural Terminal Constraint MPC (**proposed**) |
| `ntc_mpc`  | Neural Terminal Constraint MPC                         |
| `dcbf_mpc` | Discrete-time Control Barrier Function MPC             |
| `vo_mpc`   | Velocity Obstacle MPC                                  |
| `sdf_mpc`  | SDF distance-constraint MPC (baseline)                 |

A Jackal robot must navigate from start (−2, 0) to goal (13, 0) [m] while avoiding 6 moving cylindrical obstacles. The NLP is solved at 20 Hz using IPOPT + HSL MA57, compiled ahead of time to a C shared library via CasADi. Results are reported over 100 pre-generated scenarios.

See [ros2_ws/README.md](ros2_ws/README.md) for prerequisites, installation, and usage instructions.

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{derajic25a,
  title     = {Residual Neural Terminal Constraints for Model Predictive Control},
  author    = {Derajic, Bojan and others},
  booktitle = {Proceedings of The 9th Conference on Robot Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {305},
  year      = {2025},
  publisher = {PMLR},
  url       = {https://raw.githubusercontent.com/mlresearch/v305/main/assets/derajic25a/derajic25a.pdf}
}
```

A machine-readable citation file is provided at [CITATION.cff](CITATION.cff).

---

## License

This project is released under the [MIT License](LICENSE).
