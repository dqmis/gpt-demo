from typing import List

import torch

from src.data.preprocessing.tokenziner import Tokenizer
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


def generate_text_from_model(
    model: Model, tokenizer: Tokenizer, context: torch.Tensor, max_new_tokens: int = 500
) -> str:
    return "".join(
        tokenizer.decode(
            model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
        )
    )
