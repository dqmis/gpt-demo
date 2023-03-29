import torch
import torch.nn as nn
from torch.nn import functional as F

from src.modules.attention import Block

from src.models.base import Model

from typing import Optional


class MiniGPT(Model):
    def __init__(
        self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.token_embdedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embdedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)],
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        (
            B,
            T,
        ) = idx.shape
        tok_emb = self.token_embdedding_table(idx)
        pos_emb = self.position_embdedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            idx_cropped = idx[:, -self.block_size :]
            logits, _ = self(idx_cropped)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
