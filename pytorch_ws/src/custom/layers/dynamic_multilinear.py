import torch
from torch import nn, Tensor


class DynamicMultilinear(nn.Module):
    """Linear layer whose weights and biases are set externally by a hypernetwork.

    Unlike a standard ``nn.Linear``, this layer holds no trainable parameters of
    its own.  Instead, ``set_params`` must be called before every forward pass to
    inject a batch of weight/bias tensors produced by a hypernetwork.  This
    allows a single ``MainNetwork`` instance to behave differently for each
    sample in a batch — each sample gets its own parameter set.

    Shape conventions (after ``set_params``):
        weight : (batch, out_features, in_features)
        bias   : (batch, 1, out_features)

    The ``forward`` method exploits PyTorch's batched ``matmul`` broadcasting so
    that a shared input grid of shape ``(N, in_features)`` is evaluated under all
    ``batch`` parameter sets simultaneously, producing output of shape
    ``(batch, N, out_features)``.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Placeholders — real values are injected via set_params() before use.
        # Registered as buffers so they move correctly with .to(device), though
        # they will be replaced by set_params() each forward cycle.
        self.register_buffer(
            "weight", torch.zeros(1, out_features, in_features)
        )
        self.register_buffer(
            "bias", torch.zeros(1, 1, out_features)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Compute batched linear transform: out = x @ W^T + b.

        Args:
            x: Input tensor of shape ``(N, in_features)`` — the shared
               evaluation grid for all parameter sets in the batch.

        Returns:
            Tensor of shape ``(batch, N, out_features)``.
        """
        # weight: (batch, out, in) -> transpose -> (batch, in, out)
        # x:      (N, in)  [broadcast as (1, N, in) -> (batch, N, in)]
        # result: (batch, N, out)
        return torch.matmul(x, self.weight.transpose(-1, -2)) + self.bias

    def num_params(self) -> int:
        """Return the total number of parameters per sample (weight + bias)."""
        return self.in_features * self.out_features + self.out_features

    def set_params(self, params: Tensor) -> None:
        """Inject a batch of flattened weight+bias values produced by the hypernetwork.

        Args:
            params: Tensor of shape ``(batch, num_params())`` where the first
                    ``in_features * out_features`` values are the weight and the
                    remaining ``out_features`` values are the bias.
        """
        weight_n = self.in_features * self.out_features
        self.weight = params[:, :weight_n].reshape(-1, self.out_features, self.in_features)
        self.bias = params[:, weight_n:].reshape(-1, 1, self.out_features)
