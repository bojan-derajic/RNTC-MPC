import torch.nn as nn
from torch import Tensor


class Hypernetwork(nn.Module):
    """Convolutional hypernetwork that produces parameters for the main network.

    The hypernetwork takes a batch of 2-D inputs (e.g. signed-distance-field
    slices with one or more channels) and maps them to a flat vector of weights
    and biases for the ``MainNetwork``.  A standard CNN backbone extracts
    spatial features before a single linear head projects them to the required
    parameter count.

    Architecture summary (for 112×112 or 100×100 spatial input)::

        Conv2d(input_size →  16, 5×5, valid) → ReLU → MaxPool 2×2
        Conv2d(         16 →  32, 5×5, valid) → ReLU → MaxPool 2×2
        Conv2d(         32 →  64, 3×3, valid) → ReLU → MaxPool 2×2
        Conv2d(         64 → 128, 3×3, valid) → ReLU → MaxPool 2×2
        Flatten  →  2048
        Linear(2048 → output_size)

    Note:
        The flattened backbone dimension (2048) is fixed and assumes the input
        spatial resolution is **112×112 or 100×100** (both produce a 4×4 spatial
        map after four conv+pool stages, giving 128 × 4 × 4 = 2048 features).
        If you use a different input resolution, adjust the ``Linear`` input
        dimension to match the actual flattened backbone output size.

    Args:
        input_size:  Number of input channels (e.g. 2 for two SDF snapshots).
        output_size: Total number of parameters to produce for the main network
                     (obtained via ``MainNetwork.num_params()``).
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        n = 16  # base channel width; doubles with each stage

        self.backbone = nn.Sequential(
            # Stage 1: 64×64 → 30×30 → 15×15
            nn.Conv2d(input_size, n, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: 15×15 → 11×11 → 5×5
            nn.Conv2d(n, 2 * n, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: 5×5 → 3×3 → 1×1  (with 64 channels → 64*1*1 = 64)
            nn.Conv2d(2 * n, 4 * n, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

                # Stage 4: 9×9 → 7×7 → 4×4  (for 112×112 / 100×100 input)
            nn.Conv2d(4 * n, 8 * n, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),  # → (batch, 2048) for 112×112 or 100×100 input
        )

        # Linear projection: backbone features → main-network parameter vector
        self.head = nn.Linear(2048, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of 2-D SDF images to a flat parameter vector.

        Args:
            x: Tensor of shape ``(batch, input_size, H, W)``.

        Returns:
            Tensor of shape ``(batch, output_size)`` — the flattened
            weights and biases for the main network.
        """
        return self.head(self.backbone(x))
