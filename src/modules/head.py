import torch
from torch import nn
from torch.functional import F


class Head(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        print(head_size, n_embd, block_size)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        print(x.shape, "CIA", self.key.weight.shape)
        k = self.key(x)
        q = self.query(x)
        # compute attention
        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # periforming the weighted aggregation
        v = self.value(x)
        out: torch.Tensor = wei @ v
        return out
