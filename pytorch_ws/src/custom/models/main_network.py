from torch import nn, Tensor

from custom.activations import Sin
from custom.layers import DynamicMultilinear


def _make_activation(name: str) -> nn.Module:
    """Return a new activation module instance for the given name.

    A new instance is created on each call so that every layer in the network
    owns an independent module (important for any stateful activations added
    in the future).

    Args:
        name: One of ``'linear'``, ``'relu'``, ``'elu'``, ``'selu'``,
              ``'softplus'``, ``'sigmoid'``, ``'tanh'``, ``'sin'``.

    Returns:
        The corresponding ``nn.Module`` activation.

    Raises:
        KeyError: If *name* is not in the supported set.
    """
    activations = {
        "linear":   nn.Identity,
        "relu":     nn.ReLU,
        "elu":      nn.ELU,
        "selu":     nn.SELU,
        "softplus": nn.Softplus,
        "sigmoid":  nn.Sigmoid,
        "tanh":     nn.Tanh,
        "sin":      Sin,
    }
    if name not in activations:
        raise KeyError(
            f"Unknown activation '{name}'. "
            f"Choose from: {sorted(activations.keys())}"
        )
    return activations[name]()


class MainNetwork(nn.Module):
    """MLP whose weights are provided dynamically by a hypernetwork.

    The network is built from ``DynamicMultilinear`` layers, which hold no
    trainable parameters themselves.  Before each forward pass the caller must
    invoke ``set_params`` to inject the batch of weight/bias tensors generated
    by the paired ``Hypernetwork``.

    Config format::

        {
            "input_size": <int>,          # number of input features
            "layers": [                   # list of (out_features, activation) tuples
                (36, "sin"),
                (36, "sin"),
                ...
                (1, "softplus"),
            ]
        }

    Args:
        config: Architecture specification dictionary (see above).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model = nn.Sequential()

        for i, (out_features, act_name) in enumerate(config["layers"]):
            in_features = config["input_size"] if i == 0 else config["layers"][i - 1][0]
            self.model.add_module(f"layer_{i}", DynamicMultilinear(in_features, out_features))
            self.model.add_module(f"activation_{i}", _make_activation(act_name))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the MLP on input *x*.

        ``set_params`` must be called before this method to load the current
        batch of hypernetwork-generated parameters.

        Args:
            x: Input tensor of shape ``(N, input_size)`` — typically the
               flattened evaluation grid.

        Returns:
            Tensor of shape ``(batch, N, out_features_last)``.
        """
        return self.model(x)

    def num_params(self) -> int:
        """Return the total number of dynamic parameters across all layers."""
        return sum(
            self.model.get_submodule(f"layer_{i}").num_params()
            for i in range(len(self.config["layers"]))
        )

    def set_params(self, params: Tensor) -> None:
        """Distribute a flat parameter vector from the hypernetwork to each layer.

        Args:
            params: Tensor of shape ``(batch, num_params())`` — the concatenated
                    weights and biases for every layer, ordered layer-by-layer.
        """
        pointer = 0
        for i in range(len(self.config["layers"])):
            layer = self.model.get_submodule(f"layer_{i}")
            n = layer.num_params()
            layer.set_params(params[:, pointer: pointer + n])
            pointer += n
