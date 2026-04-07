import os
import shutil
import numpy as np


def train_test_split(
    data_dir: str,
    input_name: str,
    target_name: str,
    test_ratio: float = 0.2,
) -> None:
    """Randomly split raw data into ``train/`` and ``test/`` sub-directories.

    Reads ``.npy`` files from ``data_dir/<input_name>/`` and
    ``data_dir/<target_name>/``, shuffles them, and copies them into::

        data_dir/
        ├── train/
        │   ├── input/
        │   └── target/
        └── test/
            ├── input/
            └── target/

    Any pre-existing ``train/`` or ``test/`` directories are removed and
    recreated from scratch.

    File ordering is preserved across input and target via ``sorted()``, so
    the i-th input file corresponds to the i-th target file.

    Args:
        data_dir:    Root data directory.
        input_name:  Name of the sub-directory that holds raw input files.
        target_name: Name of the sub-directory that holds raw target files.
        test_ratio:  Fraction of samples to place in the test split (0–1).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: '{data_dir}'")

    input_dir = os.path.join(data_dir, input_name)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: '{input_dir}'"
        )

    target_dir = os.path.join(data_dir, target_name)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Target directory not found: '{target_dir}'"
        )

    input_filenames = sorted(os.listdir(input_dir))
    target_filenames = sorted(os.listdir(target_dir))

    if len(input_filenames) != len(target_filenames):
        raise ValueError(
            f"Mismatch: {len(input_filenames)} input files vs "
            f"{len(target_filenames)} target files."
        )

    data_size = len(input_filenames)
    indices = np.random.permutation(data_size)

    train_size = int((1.0 - test_ratio) * data_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]  # may be larger than test_ratio * n by 1

    # Build output directory structure (remove stale splits if they exist)
    for split in ("train", "test"):
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(os.path.join(split_dir, "input"))
        os.makedirs(os.path.join(split_dir, "target"))

    train_input_dir  = os.path.join(data_dir, "train", "input")
    train_target_dir = os.path.join(data_dir, "train", "target")
    test_input_dir   = os.path.join(data_dir, "test",  "input")
    test_target_dir  = os.path.join(data_dir, "test",  "target")

    for i in train_indices:
        shutil.copy(os.path.join(input_dir,  input_filenames[i]),  train_input_dir)
        shutil.copy(os.path.join(target_dir, target_filenames[i]), train_target_dir)

    for i in test_indices:
        shutil.copy(os.path.join(input_dir,  input_filenames[i]),  test_input_dir)
        shutil.copy(os.path.join(target_dir, target_filenames[i]), test_target_dir)

    print(f"Split complete  —  total: {data_size} samples")
    print(f"  Train : {len(train_indices):4d}  ({100 * len(train_indices) / data_size:.1f}%)")
    print(f"  Test  : {len(test_indices):4d}  ({100 * len(test_indices)  / data_size:.1f}%)")
