# RNTC-MPC Hypernetwork Training

PyTorch workspace for training the hypernetwork model used in RNTC-MPC (Model Predictive Control with a residual neural terminal constraint).

---

## Overview

RNTC-MPC ues a neural approximation of the HJ value function (VF) as the MPC terminal constraint.
Rather than predicting the VF directly, the RNTC model learns a *residual correction* on top of
a signed-distance-function (SDF) term:

```
vf_pred = sdf_term − residual(grid_coords)
```

where `sdf_term` is the SDF slice (channel 0) and `residual` is the output of a
small MLP whose **weights are generated on-the-fly** by a CNN **hypernetwork** conditioned on
the SDF sequence.  This lets a single network generalise to arbitrary obstacle
configurations at inference time. During the training, this estimation virtually happens at the current time, but during the deployment, it happens at the end of the MPC prediction horizon.

A non-residual **NTC** baseline (direct VF prediction) is also included for comparison.

---

## Architecture

```
SDF input  (B, C, H, W)
     │
     ▼
┌─────────────────────────────────┐
│  Hypernetwork (CNN)             │   Conv 5×5 → Conv 5×5 → Conv 3×3 → Conv 3×3
│  backbone + linear head         │   → Flatten → Linear(2048 → #params)
└─────────────────────────────────┘
     │  flat parameter vector  (B, P)
     ▼
┌─────────────────────────────────┐
│  MainNetwork (MLP)              │   set_params() injects the hypernetwork output
│  DynamicMultilinear layers      │   as weights/biases before each forward pass
└─────────────────────────────────┘
     │  residual  (B, N, 1)
     ▼
vf_pred = sdf_term − residual
```

**Key modules:**

| Module               | Location                                   | Description                               |
| -------------------- | ------------------------------------------ | ----------------------------------------- |
| `Hypernetwork`       | `src/custom/models/hypernetwork.py`        | CNN → flat parameter vector               |
| `MainNetwork`        | `src/custom/models/main_network.py`        | Dynamic MLP (weights from hypernetwork)   |
| `DynamicMultilinear` | `src/custom/layers/dynamic_multilinear.py` | Linear layer with injected weights        |
| `Sin`                | `src/custom/activations/sin.py`            | Sine activation (SIREN-style)             |
| `RWMSELoss`          | `src/custom/metrics/rwmse_loss.py`         | Radially-weighted MSE (boundary emphasis) |
| `CMELoss`            | `src/custom/metrics/cme_loss.py`           | Combined MSE + Exponential sign-agreement |
| `IoU`                | `src/custom/metrics/iou.py`                | Safe-region intersection over union       |
| `ConfusionMatrix`    | `src/custom/metrics/confusion_matrix.py`   | TP/FP/FN/TN for safe/unsafe regions       |
| `NumpyDataset`       | `src/custom/datasets/numpy_dataset.py`     | Paired `.npy` file dataset                |
| `train_test_split`   | `src/utils/train_test_split.py`            | Random train/test split utility           |

---

## Project Structure

```
workspace/
├── data/                          # Data root (populate before training)
│   ├── sdf/                       # Raw SDF inputs  (.npy, one file per sample)
│   ├── vf/                        # Raw VF targets  (.npy, one file per sample)
│   ├── grid/
│   │   └── grid.npy               # Evaluation grid coordinates
│   ├── train/                     # Created automatically by train_test_split
│   │   ├── input/
│   │   └── target/
│   └── test/
│       ├── input/
│       └── target/
├── mlruns/                        # MLflow experiment store
├── scripts/
│   ├── build_docker_image.sh      # Build the Docker image
│   ├── start_mlflow_server.sh     # Launch the MLflow tracking UI
│   └── start_model_training.sh    # Run training inside Docker
├── src/
│   ├── train_rntc.py              # ← RNTC training entry point
│   ├── train_ntc.py               # ← NTC baseline training entry point
│   ├── custom/
│   │   ├── activations/           # Custom activation functions
│   │   ├── datasets/              # Dataset loaders
│   │   ├── layers/                # Custom neural-network layers
│   │   ├── metrics/               # Loss functions and evaluation metrics
│   │   └── models/                # Hypernetwork and MainNetwork
│   └── utils/
│       └── train_test_split.py    # Data-split utility
├── Dockerfile
├── .devcontainer/devcontainer.json
└── requirements.txt
```

---

## Data Format

Each sample consists of one `.npy` input file and one `.npy` target file with matching names.

| Array           | Shape          | Description                                     |
| --------------- | -------------- | ----------------------------------------------- |
| **Input (SDF)** | `(C, H, W)`    | `C` SDF channels — one per time snapshot        |
| **Target (VF)** | `(H, W, …, 1)` | Ground-truth value function on the spatial grid |
| **Grid**        | `(H, W, …, D)` | Grid of `D`-dimensional evaluation coordinates  |

> **SDF channel convention:** Channel 0 is the current-time SDF slice and is
> used as the `sdf_term` in the residual formula.  Additional channels
> (e.g. channel 1 at `t = −0.4 s`) serve as context for the hypernetwork.

**If you have raw data in `data/sdf/` and `data/vf/`**, set `split_data=True` and
`test_ratio=0.2` in the training script to auto-split before the first run.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place your `.npy` files in:

```
data/
├── sdf/          # input SDF files (shape C×H×W each)
├── vf/           # target VF files (shape H×W×…×1 each)
└── grid/
    └── grid.npy  # evaluation grid (shape H×W×…×D)
```

Or, if train/test splits already exist, ensure `data/train/` and `data/test/` are populated.

### 3. Train the RNTC model

```bash
cd /workspace
python src/train_rntc.py
```

To train the NTC baseline:

```bash
python src/train_ntc.py
```

### 4. Monitor training with MLflow

```bash
bash scripts/start_mlflow_server.sh
# Then open http://localhost:8080 in your browser
```

---

## Docker / Dev Container

Build the image:

```bash
bash scripts/build_docker_image.sh
```

Run training in a container (GPU-enabled):

```bash
bash scripts/start_model_training.sh
```

The VS Code Dev Container (`.devcontainer/devcontainer.json`) mounts the workspace and
enables GPU passthrough (`--gpus=all`) for seamless development.

---

## Configuration Reference

All training options are passed to the `train()` function.  Edit the `__main__` block of
`train_rntc.py` (or `train_ntc.py`) to change them.

| Parameter          | Type        | Description                                                   |
| ------------------ | ----------- | ------------------------------------------------------------- |
| `data_dir`         | `str`       | Root data directory path                                      |
| `input_name`       | `str`       | Sub-directory name for raw SDF inputs (default `"sdf"`)       |
| `target_name`      | `str`       | Sub-directory name for raw VF targets (default `"vf"`)        |
| `main_net_config`  | `dict`      | `{"input_size": int, "layers": [(size, act), …]}`             |
| `batch_size`       | `int`       | Training batch size                                           |
| `num_epochs`       | `int`       | Total training epochs                                         |
| `lr`               | `float`     | Adam learning rate                                            |
| `lr_sched_config`  | `dict`      | `{"milestones": [int, …], "gamma": float}`                    |
| `split_data`       | `bool`      | Auto-split raw data before training                           |
| `test_ratio`       | `float`     | Fraction of data for test (used when `split_data=True`)       |
| `loss_func_name`   | `str`       | `"MSE"`, `"RWMSE"`, or `"CME"`                                |
| `loss_func_params` | `dict`      | Extra kwargs for the loss class (e.g. `{"gamma": 0.2}`)       |
| `device`           | `str`       | PyTorch device (`"cuda:0"`, `"cpu"`, …)                       |
| `dynamics`         | `str`       | Label for the MLflow experiment (e.g. `"Kinematic Unicycle"`) |
| `sdf_idx`          | `list[int]` | SDF channel indices fed to the hypernetwork                   |

### Main network config example

```python
main_net_config = {
    "input_size": 3,          # x, y, θ  (kinematic unicycle)
    "layers": [
        (36, "sin"),           # SIREN-style periodic layers
        (36, "sin"),
        (36, "sin"),
        (18, "relu"),          # ReLU refinement
        (18, "relu"),
        (18, "relu"),
        (9,  "relu"),
        (9,  "relu"),
        (9,  "relu"),
        (1,  "softplus"),      # non-negative scalar output
    ],
}
```

Supported activations: `linear`, `relu`, `elu`, `selu`, `softplus`, `sigmoid`, `tanh`, `sin`.

### Loss functions

| Name    | Formula                                                              | Use case                          |
| ------- | -------------------------------------------------------------------- | --------------------------------- |
| `MSE`   | `mean((pred − target)²)`                                             | Baseline regression               |
| `RWMSE` | `mean(w(target) · (pred − target)²)` where `w(t) = 1 + α·exp(−β·t²)` | Emphasise boundary accuracy       |
| `CME`   | `γ·MSE + (1−γ)·mean(exp(−pred·target))`                              | Balance accuracy + sign agreement |

For RNTC the recommended default is **CME with γ = 0.2**.

---

## Metrics

Training logs the following metrics to MLflow at every epoch:

| Metric                     | Description                                |
| -------------------------- | ------------------------------------------ |
| `train_loss` / `test_loss` | Mean batch loss                            |
| `train_iou` / `test_iou`   | Safe-region IoU ∈ [0, 1] (higher = better) |

After training, artefacts are logged:

- `visuals/vf_pred_test_NNN.png` — side-by-side SDF / true VF / predicted VF for 50 test samples  
- `confusion_matrix.png` — normalised TP/FP/FN/TN for safe/unsafe region classification  
- `hypernet_weights/` — PyTorch state-dict checkpoints (every 5 epochs + final)

---

## Notes

- The hypernetwork's CNN backbone assumes a **112 × 112 or 100 × 100** SDF input resolution
  (both produce a 2048-dimensional flattened feature vector through four conv+pool stages).  
  If your SDF images have a different resolution, update the `Linear(2048, …)` head
  in `Hypernetwork.__init__` to match the actual flattened backbone output size.
- Set `split_data=False` when `data/train/` and `data/test/` already exist to skip re-splitting.
- The CME warm-start (MSE for epoch 0) prevents divergence from the large exponential term
  before the network produces sensible predictions.
