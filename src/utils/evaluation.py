import torch
from typing import Dict

from src.models.base import Model
from src.utils.data import get_batch


@torch.no_grad()
def estimate_loss(
    model: Model,
    eval_iters: int,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    device: torch.DeviceObjType,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, device, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
