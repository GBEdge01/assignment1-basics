from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input/target token batches for next-token prediction."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D array")

    num_possible_starting_indices = len(dataset) - context_length
    if num_possible_starting_indices <= 0:
        raise ValueError("dataset must be longer than context_length")

    starts = np.random.randint(0, num_possible_starting_indices, size=batch_size)
    offsets = np.arange(context_length)
    x = dataset[starts[:, None] + offsets[None, :]]
    y = dataset[starts[:, None] + offsets[None, :] + 1]

    return torch.from_numpy(x.astype(np.int64, copy=False)).to(device), torch.from_numpy(
        y.astype(np.int64, copy=False)
    ).to(device)
