import torch
import torch.nn.functional as F
from torch import nn, Tensor


class CMELoss(nn.Module):
    """Combined MSE and Exponential (CME) loss.

    Blends a standard MSE term (for regression accuracy) with an exponential
    sign-agreement term (for safety-region correctness)::

        L = gamma * MSE(input, target)
          + (1 - gamma) * mean( exp(-input * target) )

    The exponential term is minimised when ``input`` and ``target`` have the
    same sign — i.e. both predict safe (> 0) or both predict unsafe (≤ 0).
    It penalises sign disagreements exponentially, making it particularly
    useful for learning value-function zero level-sets.

    Note:
        When ``gamma = 1`` the loss reduces to plain MSE.
        When ``gamma = 0`` only the sign-agreement term is used.
        A value around ``0.2`` balances accuracy with safety-boundary sharpness.

    Args:
        gamma: Mixing coefficient in [0, 1] (default 1.0).
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse_term = F.mse_loss(input, target)
        exp_term = torch.exp(-input * target).mean()
        return self.gamma * mse_term + (1.0 - self.gamma) * exp_term
