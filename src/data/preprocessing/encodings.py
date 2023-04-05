from src.data.preprocessing.tokenziner import Tokenizer
import torch

from typing import List, Tuple


def build_dataset(
    words: List[str], tokenizer: Tokenizer, block_size: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w:
            ix = tokenizer.encode(ch)
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y
