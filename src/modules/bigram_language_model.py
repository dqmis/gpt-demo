from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.token_embedding_table(
            idx
        )  # (batch_size, block_size, vocab_size) -> (B,T,C)

        # If targets is None, we are in inference mode
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)  # (batch_size * block_size, vocab_size)
        targets = targets.view(-1)  # (batch_size * block_size)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens given a sequence of tokens.

        Args:
            idx: (batch_size, block_size) tensor containing the tokens
            max_new_tokens: maximum number of tokens to generate

        Returns:
            (batch_size, block_size + max_new_tokens) tensor containing the original
            tokens and the generated tokens
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
