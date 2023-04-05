import torch
from typing import List

from src.models.base import Model


@torch.no_grad()
def estimate_loss(
    model: Model,
    val_dataloader: torch.Tensor,
    device: torch.DeviceObjType,
) -> float:
    model.eval()

    losses: List[float] = []
    for batch in val_dataloader:
        xb, yb = batch
        xb = xb.to(device)
        yb = yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)
