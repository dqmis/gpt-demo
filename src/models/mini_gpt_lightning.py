from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch import optim

from src.models.base import Model
from src.models.mini_gpt import MiniGPT


class MiniGPTLightning(pl.LightningModule, Model):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._device = device
        self.save_hyperparameters()
        self.model = MiniGPT(vocab_size, block_size, n_embd, n_head, n_layer)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x, y)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        x.to(self._device)
        y.to(self._device)
        _, loss = self(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        x.to(self._device)
        y.to(self._device)
        _, loss = self(x, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        return self.model.generate(idx, max_new_tokens)
