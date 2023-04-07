from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from api.utils.model_utils import (
    load_model,
    load_tokenizer,
    parse_str_to_context,
    parse_tokens_to_str,
)

MODEL = load_model("./models/model-epoch=00-val_loss=3.05.ckpt")
TOKENIZER = load_tokenizer("./models/tokenizer.pkl")

# loading model
app = FastAPI()


class Input(BaseModel):
    text: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"health": "alive"}


@app.post("/generate_text")
def generate_text(input: Input) -> Dict[str, str]:
    context = parse_str_to_context(input.text, TOKENIZER)
    output = MODEL.generate(context, 512)
    return {"output": parse_tokens_to_str(output, TOKENIZER)}
