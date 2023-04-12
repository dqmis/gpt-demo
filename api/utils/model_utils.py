import pickle

import torch

from src.data.preprocessing.tokenziner import Tokenizer
from src.models import MiniGPTLightning


def load_model(checkpoints_path: str) -> MiniGPTLightning:
    model = MiniGPTLightning.load_from_checkpoint(checkpoints_path)
    model.to("cpu")
    model.eval()
    return model


def load_tokenizer(checkpoints_path: str) -> Tokenizer:
    with open(checkpoints_path, "rb") as f:
        return pickle.load(f)


def parse_str_to_context(text: str, tokenizer: Tokenizer) -> torch.Tensor:
    context = torch.unsqueeze(
        torch.tensor(tokenizer.encode(text), dtype=torch.long, device="cpu"),
        0,
    )
    return context


def parse_tokens_to_str(tokens: torch.Tensor, tokenizer: Tokenizer) -> str:
    return "".join(tokenizer.decode(tokens[0].tolist()))
