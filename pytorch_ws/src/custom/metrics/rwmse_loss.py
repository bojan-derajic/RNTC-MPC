import torch
from torch import nn, Tensor


class RWMSELoss(nn.Module):
    """Radially Weighted Mean-Squared-Error loss.

    Assigns higher weight to samples near the zero level-set of the target
    (i.e. near the decision boundary), which is critical for accurate
    safe/unsafe region prediction in value-function learning.

    The per-element weight is::

        w(t) = 1 + alpha * exp(-beta * t²)

    which peaks at ``t = 0`` (boundary) and decays away from it.

    Loss::

        L = mean( w(target) * (input - target)² )

    Args:
        alpha: Scales the boundary emphasis (default 1.0).
               Higher values increase the relative weight near ``target = 0``.
        beta:  Controls the width of the boundary region (default 1.0).
               Higher values create a narrower region of emphasis.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight = 1.0 + self.alpha * torch.exp(-self.beta * target.square())
        return (weight * (input - target).square()).mean()
