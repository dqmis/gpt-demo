# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.data.preprocessing.tokenziner import Tokenizer


def get_dataset(path_to_text: str) -> str:
    with open(path_to_text, "r", encoding="utf-8") as f:
        text = f.read()
    return text


class TokensDatataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, text: str, block_size: int) -> None:
        self._data = tokenizer.encode(text)
        self._block_size = block_size

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._data[index : index + self._block_size]  # noqa
        y = self._data[index + 1 : index + self._block_size + 1]  # noqa

        if len(x) < self._block_size:
            x += [0] * (self._block_size - len(x))
        if len(y) < self._block_size:
            y += [0] * (self._block_size - len(y))

        return torch.tensor(x), torch.tensor(y)
