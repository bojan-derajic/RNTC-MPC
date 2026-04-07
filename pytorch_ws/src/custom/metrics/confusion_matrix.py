import torch
from torch import nn, Tensor


class ConfusionMatrix(nn.Module):
    """Normalised confusion matrix for binary safe/unsafe region classification.

    Interprets ``input > 0`` as "predicted safe" (Positive) and
    ``target > 0`` as "truly safe" (Positive).  Each cell is normalised by
    the total number of elements so that the four values sum to 1.

    Returns a 2×2 tensor::

        [[TP, FN],
         [FP, TN]]

    where:
        TP — predicted safe,  truly safe   (correct, safe)
        FN — predicted unsafe, truly safe  (miss: unsafe when actually safe)
        FP — predicted safe,  truly unsafe (false alarm: safe when actually unsafe)
        TN — predicted unsafe, truly unsafe (correct, unsafe)

    Note:
        FP (false alarm / over-prediction of the safe set) is the most
        safety-critical error: the controller believes it is safe when it is not.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n = input.numel()
        TP = ((input > 0) & (target > 0)).sum().float() / n
        FN = ((input <= 0) & (target > 0)).sum().float() / n
        FP = ((input > 0) & (target <= 0)).sum().float() / n
        TN = ((input <= 0) & (target <= 0)).sum().float() / n
        return torch.stack([torch.stack([TP, FN]), torch.stack([FP, TN])])
