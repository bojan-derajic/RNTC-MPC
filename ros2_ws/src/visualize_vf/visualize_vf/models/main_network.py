import torch
from torch import nn, Tensor


def get_activation_function(activation_function: str):
    """Returns the activation function based on the string parameter."""

    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "sigmoid": nn.Sigmoid(),
        "sin": Sin(),
    }
    if activation_function not in activations:
        raise ValueError(f"Unsupported activation function: {activation_function}")
    return activations[activation_function]


class Sin(nn.Module):
    """Applies sine function element-wise."""

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)


class DynamicMultilinear(nn.Module):
    """Linear transformation layer with multiple sets of dynamic parameters."""

    def __init__(self, weight: Tensor = None, bias: Tensor = None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        if self.weight is None or self.bias is None:
            raise AttributeError("Parameters of the module are not set.")
        return torch.matmul(input, self.weight) + self.bias

    def set_params(self, weight: Tensor, bias: Tensor = None) -> None:
        self.weight = weight
        self.bias = bias


class MainNetwork(nn.Module):
    """Dynamically parametrized model which predicts value function for given coordinates."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model = nn.Sequential()
        for i in range(len(config["num_hidden_units"])):
            self.model.add_module(f"layer_{i}", DynamicMultilinear())
            if i < 3:
                self.model.add_module(
                    f"activation_{i}",
                    Sin(),
                )
            else:
                self.model.add_module(
                    f"activation_{i}",
                    get_activation_function(config["activation_function"]),
                )
        self.model.add_module("output_layer", DynamicMultilinear())

    def forward(self, input: Tensor) -> Tensor:
        return -(1 + torch.nn.functional.elu(self.model(input)))

    def set_params(self, params: Tensor) -> None:
        offset = 0
        in_features = self.config["input_size"]
        for i in range(len(self.config["num_hidden_units"])):
            out_features = self.config["num_hidden_units"][i]
            weight_num_params = in_features * out_features
            weight = params[:, offset : offset + weight_num_params].reshape(
                (-1, in_features, out_features)
            )
            offset += weight_num_params
            bias_num_params = out_features
            bias = params[:, offset : offset + bias_num_params].reshape(
                (-1, 1, out_features)
            )
            offset += bias_num_params
            self.model.get_submodule(f"layer_{i}").set_params(weight, bias)
            in_features = out_features
        out_features = self.config["output_size"]
        weight_num_params = in_features * out_features
        weight = params[:, offset : offset + weight_num_params].reshape(
            (-1, in_features, out_features)
        )
        offset += weight_num_params
        bias_num_params = out_features
        bias = params[:, offset : offset + bias_num_params].reshape(
            (-1, 1, out_features)
        )
        self.model.get_submodule("output_layer").set_params(weight, bias)

    def num_params(self) -> int:
        count = 0
        in_features = self.config["input_size"]
        for i in range(len(self.config["num_hidden_units"])):
            out_features = self.config["num_hidden_units"][i]
            count += in_features * out_features + out_features
            in_features = out_features
        out_features = self.config["output_size"]
        count += in_features * out_features + out_features
        return count
