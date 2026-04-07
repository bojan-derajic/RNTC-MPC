import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """PyTorch Dataset that loads paired input/target ``.npy`` files.

    Expected directory layout::

        data_dir/
        ├── train/
        │   ├── input/    # one .npy file per sample (e.g. SDF data)
        │   └── target/   # one .npy file per sample (e.g. value function)
        └── test/
            ├── input/
            └── target/

    Files inside each ``input/`` and ``target/`` folder are paired by their
    sorted order, so filenames must align.

    Args:
        data_dir:     Root data directory containing ``train/`` and ``test/``
                      sub-directories.
        device:       PyTorch device string (``'cpu'``, ``'cuda:0'``, …).
        train:        If ``True``, load from ``train/``; otherwise from ``test/``.
        preload_data: If ``True``, load all samples into memory on construction
                      (fast iteration, high RAM usage).  If ``False``, load
                      each sample on demand (slow iteration, low RAM usage).
    """

    def __init__(
        self,
        data_dir: str,
        device: str,
        train: bool = True,
        preload_data: bool = False,
    ) -> None:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: '{data_dir}'")

        split = "train" if train else "test"
        self.input_dir = os.path.join(data_dir, split, "input")
        self.target_dir = os.path.join(data_dir, split, "target")

        for d in (self.input_dir, self.target_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Expected directory not found: '{d}'")

        self.device = device
        self.input_filenames = sorted(os.listdir(self.input_dir))
        self.target_filenames = sorted(os.listdir(self.target_dir))

        if len(self.input_filenames) != len(self.target_filenames):
            raise ValueError(
                f"Mismatch between input ({len(self.input_filenames)}) and "
                f"target ({len(self.target_filenames)}) file counts."
            )

        # Optional eager pre-load into device memory
        self.preload_data = preload_data
        if preload_data:
            self._inputs: list[Tensor] = []
            self._targets: list[Tensor] = []
            for i in range(len(self)):
                self._inputs.append(self._load_input(i))
                self._targets.append(self._load_target(i))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.input_filenames)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if self.preload_data:
            return self._inputs[index], self._targets[index]
        return self._load_input(index), self._load_target(index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_npy(self, path: str) -> Tensor:
        """Load a ``.npy`` file and return a float32 tensor on ``self.device``."""
        return torch.from_numpy(np.load(path)).float().to(self.device)

    def _load_input(self, index: int) -> Tensor:
        return self._load_npy(os.path.join(self.input_dir, self.input_filenames[index]))

    def _load_target(self, index: int) -> Tensor:
        return self._load_npy(os.path.join(self.target_dir, self.target_filenames[index]))
