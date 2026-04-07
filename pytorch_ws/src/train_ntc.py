"""Training script for the Neural Terminal Constraint (NTC) baseline model.

Unlike the RNTC variant, the NTC model directly predicts the value function
without a residual SDF term::

    vf_pred = main_net(grid_coords)

This script is otherwise identical to ``train_rntc.py`` and serves as a
non-residual baseline for comparison.

Usage::

    python src/train_ntc.py

Results are logged to MLflow (``./mlruns``).  Launch the UI with::

    mlflow ui --backend-store-uri ./mlruns
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in containers / headless
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import mlflow

# Allow running the script directly from the repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))

from utils import train_test_split
from custom.models import MainNetwork, Hypernetwork
from custom.datasets import NumpyDataset
from custom.metrics import RWMSELoss, CMELoss, IoU, ConfusionMatrix


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_loss(loss_func_name: str, loss_func_params: dict) -> nn.Module:
    """Instantiate the requested loss function."""
    if loss_func_name == "MSE":
        return nn.MSELoss()
    elif loss_func_name == "RWMSE":
        return RWMSELoss(**loss_func_params)
    elif loss_func_name == "CME":
        return CMELoss(**loss_func_params)
    else:
        raise ValueError(
            f"Unknown loss function '{loss_func_name}'. "
            "Choose from: 'MSE', 'RWMSE', 'CME'."
        )


def _log_confusion_matrix(
    hypernet: nn.Module,
    main_net: nn.Module,
    test_dataloader: DataLoader,
    grid_flat: torch.Tensor,
    sdf_idx: list,
) -> plt.Figure:
    """Compute mean confusion matrix over the test set and return a figure."""
    conf_mat_func = ConfusionMatrix()
    batches = []

    with torch.no_grad():
        for sdf, vf in test_dataloader:
            main_net_params = hypernet(sdf[:, sdf_idx, ...])
            main_net.set_params(main_net_params)
            vf_pred = main_net(grid_flat).view_as(vf)
            batches.append(conf_mat_func(vf_pred, vf))

    conf_mat = torch.stack(batches, dim=-1).mean(dim=-1).numpy()

    fig, ax = plt.subplots()
    ax.matshow(conf_mat, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (normalised)")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Safe (+)", "Unsafe (-)"])
    ax.set_yticklabels(["Safe (+)", "Unsafe (-)"])
    for (r, c), val in np.ndenumerate(conf_mat):
        ax.text(c, r, f"{val:.3f}", ha="center", va="center", color="black")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    input_name: str,
    target_name: str,
    main_net_config: dict,
    batch_size: int,
    num_epochs: int,
    lr: float,
    lr_sched_config: dict,
    split_data: bool = True,
    test_ratio: float = None,
    loss_func_name: str = "MSE",
    loss_func_params: dict = None,
    device: str = "cuda",
    dynamics: str = None,
    sdf_idx: list = None,
) -> None:
    """Train the NTC (non-residual) hypernetwork baseline model.

    Args:
        data_dir:         Path to the root data directory.
        input_name:       Name of the raw input sub-directory (e.g. ``'sdf'``).
        target_name:      Name of the raw target sub-directory (e.g. ``'vf'``).
        main_net_config:  Architecture config for ``MainNetwork``.
        batch_size:       Number of samples per training batch.
        num_epochs:       Total number of training epochs.
        lr:               Initial learning rate for Adam.
        lr_sched_config:  Dict with ``'milestones'`` (list[int]) and ``'gamma'``
                          (float) for ``MultiStepLR``.
        split_data:       If ``True``, perform a random train/test split before
                          training.  Requires ``test_ratio``.
        test_ratio:       Fraction of data reserved for testing (required when
                          ``split_data=True``).
        loss_func_name:   One of ``'MSE'``, ``'RWMSE'``, ``'CME'``.
        loss_func_params: Extra keyword arguments forwarded to the loss class.
                          Pass ``{}`` or ``None`` for MSE.
        device:           PyTorch device string.
        dynamics:         Human-readable dynamics label used in MLflow (e.g.
                          ``'Kinematic Unicycle'``).
        sdf_idx:          List of SDF channel indices fed to the hypernetwork.
    """
    if loss_func_params is None:
        loss_func_params = {}

    # ------------------------------------------------------------------
    # Optional data split
    # ------------------------------------------------------------------
    if split_data:
        if test_ratio is None:
            raise ValueError(
                "'test_ratio' must be provided when 'split_data=True'"
            )
        if not (0.0 < test_ratio < 1.0):
            raise ValueError("'test_ratio' must be in the open interval (0, 1)")
        train_test_split(data_dir, input_name, target_name, test_ratio)

    # ------------------------------------------------------------------
    # Datasets and data loaders
    # ------------------------------------------------------------------
    train_dataset = NumpyDataset(data_dir=data_dir, device=device, train=True,  preload_data=False)
    test_dataset  = NumpyDataset(data_dir=data_dir, device=device, train=False, preload_data=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Evaluation grid: flatten spatial dims → (N, coord_dim)
    grid_path = os.path.join(data_dir, "grid", "grid.npy")
    grid = torch.tensor(np.load(grid_path), dtype=torch.float32, device=device)
    grid_flat = grid.reshape(-1, grid.shape[-1])

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    main_net = MainNetwork(main_net_config).to(device)
    main_net_num_params = main_net.num_params()

    hypernet = Hypernetwork(
        input_size=len(sdf_idx),
        output_size=main_net_num_params,
    ).to(device)
    hypernet_num_params = sum(p.numel() for p in hypernet.parameters())

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler, metric
    # ------------------------------------------------------------------
    loss_func    = _build_loss(loss_func_name, loss_func_params)
    optimizer    = optim.Adam(hypernet.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_sched_config["milestones"],
        gamma=lr_sched_config["gamma"],
    )
    iou_func = IoU()

    # ------------------------------------------------------------------
    # MLflow experiment  (NTC, not RNTC)
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(f"NTC ({dynamics})")

    with mlflow.start_run(run_name=loss_func_name):
        mlflow.log_artifacts("./src", "src")
        mlflow.log_params({
            "dynamics":              dynamics,
            "input":                 input_name,
            "target":                target_name,
            "loss_function":         loss_func_name,
            "optimizer":             optimizer.__class__.__name__,
            "num_epochs":            num_epochs,
            "batch_size":            batch_size,
            "lr":                    lr,
            "lr_scheduler":          lr_sched_config,
            "main_net_config":       main_net_config,
            "main_net_num_params":   main_net_num_params,
            "hypernet_num_params":   hypernet_num_params,
            "grid_shape":            list(grid.shape),
            "sdf_idx":               sdf_idx,
        })
        for k, v in loss_func_params.items():
            mlflow.log_param(f"{loss_func_name}_{k}", v)

        # ==============================================================
        # Training loop
        # ==============================================================
        for epoch in range(num_epochs):

            # ── Training phase ─────────────────────────────────────────
            hypernet.train()
            train_losses, train_ious = [], []

            for sdf, vf in train_loader:
                main_net_params = hypernet(sdf[:, sdf_idx, ...])
                main_net.set_params(main_net_params)

                # Direct value-function prediction (no residual)
                vf_pred = main_net(grid_flat).view_as(vf)

                if epoch == 0 and loss_func_name == "CME":
                    # Warm-start: plain MSE avoids CME instability in epoch 0
                    loss = nn.MSELoss()(vf_pred, vf)
                else:
                    loss = loss_func(vf_pred, vf)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.detach().cpu().item())
                train_ious.append(iou_func(vf_pred.detach(), vf).cpu().item())

            lr_scheduler.step()

            # ── Evaluation phase ───────────────────────────────────────
            hypernet.eval()
            test_losses, test_ious = [], []

            with torch.no_grad():
                for sdf, vf in test_loader:
                    main_net_params = hypernet(sdf[:, sdf_idx, ...])
                    main_net.set_params(main_net_params)
                    vf_pred = main_net(grid_flat).view_as(vf)

                    test_losses.append(loss_func(vf_pred, vf).cpu().item())
                    test_ious.append(iou_func(vf_pred, vf).cpu().item())

            # ── Logging ────────────────────────────────────────────────
            mean_train_loss = float(np.mean(train_losses))
            mean_test_loss  = float(np.mean(test_losses))
            mean_train_iou  = float(np.mean(train_ious))
            mean_test_iou   = float(np.mean(test_ious))

            print(
                f"Epoch {epoch:3d}/{num_epochs - 1} | "
                f"Train loss: {mean_train_loss:.6f} | Test loss: {mean_test_loss:.6f} | "
                f"Train IoU: {mean_train_iou:.4f} | Test IoU: {mean_test_iou:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": mean_train_loss,
                "test_loss":  mean_test_loss,
                "train_iou":  mean_train_iou,
                "test_iou":   mean_test_iou,
            }, step=epoch)

            if epoch % 5 == 0:
                mlflow.pytorch.log_state_dict(hypernet.state_dict(), "hypernet_weights")

        mlflow.pytorch.log_state_dict(hypernet.state_dict(), "hypernet_weights")

        # ==============================================================
        # Post-training artefacts
        # ==============================================================
        hypernet.eval()

        # ── Prediction visualisations ──────────────────────────────────
        with torch.no_grad():
            n_vis = min(50, len(test_dataset))
            vis_indices = np.random.choice(len(test_dataset), size=n_vis, replace=False)

            for vis_num, test_idx in enumerate(vis_indices):
                sdf, vf = test_dataset[test_idx]

                main_net_params = hypernet(sdf[None, sdf_idx, ...])
                main_net.set_params(main_net_params)
                vf_pred = main_net(grid_flat).view_as(vf)

                sdf_plot  = sdf[0].cpu()            # channel-0 SDF slice (H, W)
                vf_plot   = vf.cpu()[..., 0]
                pred_plot = vf_pred.cpu()[..., 0]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(sdf_plot)
                axes[0].contour(sdf_plot, levels=[0], colors="k")
                axes[0].set_title("SDF (channel 0)")
                axes[0].axis("off")

                axes[1].imshow(vf_plot)
                axes[1].contour(sdf_plot, levels=[0], colors="k")
                axes[1].contour(vf_plot,  levels=[0], colors="r")
                axes[1].set_title("True VF")
                axes[1].axis("off")

                axes[2].imshow(pred_plot)
                axes[2].contour(sdf_plot,  levels=[0], colors="k")
                axes[2].contour(vf_plot,   levels=[0], colors="r")
                axes[2].contour(pred_plot, levels=[0], colors="orange")
                axes[2].set_title("Predicted VF")
                axes[2].axis("off")

                plt.tight_layout()
                mlflow.log_figure(fig, f"visuals/vf_pred_test_{vis_num:03d}.png")
                plt.close(fig)

        # ── Confusion matrix ──────────────────────────────────────────
        conf_fig = _log_confusion_matrix(
            hypernet, main_net, test_loader, grid_flat, sdf_idx
        )
        mlflow.log_figure(conf_fig, "confusion_matrix.png")
        plt.close(conf_fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dynamics = "Kinematic Unicycle"

    main_net_config = {
        "input_size": 3,          # 3-D grid coordinates (x, y, θ)
        "layers": [
            (36, "sin"),
            (36, "sin"),
            (36, "sin"),
            (18, "relu"),
            (18, "relu"),
            (18, "relu"),
            (9, "relu"),
            (9, "relu"),
            (9, "relu"),
            (1, "softplus"),
        ],
    }

    lr_sched_config = {
        "milestones": [85, 95],
        "gamma": 0.1,
    }

    train(
        data_dir="./data",
        input_name="sdf",
        target_name="vf",
        main_net_config=main_net_config,
        batch_size=40,
        num_epochs=100,
        lr=1e-4,
        lr_sched_config=lr_sched_config,
        split_data=False,          # Set True + test_ratio to auto-split raw data
        test_ratio=0.2,
        loss_func_name="MSE",
        loss_func_params={},
        device=device,
        dynamics=dynamics,
        sdf_idx=[0, 1],            # SDF channels: t=0 and t=-0.4 s
    )
