import casadi as ca


def get_func(func_name: str):
    """Returns a function implemented using CasADi based on the function name."""

    def linear(x):
        return x

    def relu(x):
        return ca.fmax(0, x)

    def elu(x):
        return ca.if_else(x >= 0, x, (ca.exp(x) - 1))

    def elu_plus_1(x):
        return elu(x) + 1

    def selu(x):
        scale = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        return scale * (ca.fmax(0, x) + ca.fmin(0, alpha * (ca.exp(x) - 1)))

    def softplus(x):
        return ca.log(1 + ca.exp(x))

    def sigmoid(x):
        return 1 / (1 + ca.exp(-x))

    func_bank = {
        "linear": linear,
        "relu": relu,
        "elu": elu,
        "elu_plus_1": elu_plus_1,
        "selu": selu,
        "softplus": softplus,
        "sigmoid": sigmoid,
        "tanh": ca.tanh,
        "sin": ca.sin,
    }
    if func_name not in func_bank.keys():
        raise KeyError(f"Invalid function name: '{func_name}'")
    return func_bank[func_name]


class MainNetwork:
    def __init__(self, config: dict):
        self.layers = []
        for i in range(len(config["layers"])):
            input_size = config["input_size"] if i == 0 else config["layers"][i - 1][0]
            output_size = config["layers"][i][0]
            self.layers.append(
                {
                    "weight": ca.MX.sym(f"weight_{i:02}", input_size, output_size),
                    "bias": ca.MX.sym(f"bias_{i:02}", 1, output_size),
                    "act_func": get_func(config["layers"][i][1]),
                }
            )

    def __call__(self, x):
        for layer in self.layers:
            x = ca.mtimes(x, layer["weight"]) + layer["bias"]
            x = layer["act_func"](x)
        return x

    def num_params(self):
        num_params = 0
        for layer in self.layers:
            num_params += layer["weight"].shape[1] * (layer["weight"].shape[0] + 1)
        return num_params
