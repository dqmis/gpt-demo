import torch

from typing import Tuple


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    device: torch.DeviceObjType,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
