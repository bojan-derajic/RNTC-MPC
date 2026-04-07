import torch.nn as nn

class Hypernetwork(nn.Module):
    """Predicts parameters of the main network based on a 2D input."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        n = 16
        self.backbone = nn.Sequential(
            nn.Conv2d(input_size, n, (5, 5), (1, 1), "valid", bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            
            nn.Conv2d(n, 2 * n, (5, 5), (1, 1), "valid", bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(2 * n, 4 * n, (3, 3), (1, 1), "valid", bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(4 * n, 8 * n, (3, 3), (1, 1), "valid", bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(2048, output_size),
        )

    def forward(self, input):
        input = self.backbone(input)
        return self.head(input)