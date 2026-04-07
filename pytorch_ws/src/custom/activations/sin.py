import torch
from torch import nn, Tensor


class Sin(nn.Module):
    """Sine activation function: f(x) = sin(x).

    Useful for Implicit Neural Representations (INR / SIREN-style networks)
    where periodic activations help learn high-frequency signals.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)
