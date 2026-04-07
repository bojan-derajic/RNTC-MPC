import torch
from torch import nn, Tensor


class IoU(nn.Module):
    """Intersection over Union (IoU) for safe-region prediction.

    Treats the positive level-set (value > 0) as the predicted safe region and
    computes the fraction of overlap between the predicted and ground-truth safe
    regions::

        IoU = |{input > 0} ∩ {target > 0}|
              ─────────────────────────────
              |{input > 0} ∪ {target > 0}|

    A small epsilon (1e-8) is added to the denominator to avoid division by
    zero in the degenerate case where both sets are empty.

    Returns a scalar tensor in [0, 1]; higher is better.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        intersection = ((input > 0.0) & (target > 0.0)).sum().float()
        union = ((input > 0.0) | (target > 0.0)).sum().float()
        return intersection / (union + 1e-8)
