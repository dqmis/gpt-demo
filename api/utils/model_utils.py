import pickle

import torch

from src.data.preprocessing.tokenziner import Tokenizer
from src.models import MiniGPTLightning


def load_model(
    model_name: str, checkpoints_path: str = "./artifacts"
) -> MiniGPTLightning:
    model = MiniGPTLightning.load_from_checkpoint(
        f"{checkpoints_path}/{model_name}.ckpt"
    )
    model.to("cpu")
    model.eval()
    return model


def load_tokenizer(checkpoints_path: str = "./artifacts/tokenizer.pkl") -> Tokenizer:
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
