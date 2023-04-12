FROM python:3.10.11-slim

ENV POETRY_VERSION=1.3.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1

RUN pip install twine \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app
COPY ./pyproject.toml ./poetry.lock ./

RUN poetry install --without dev

COPY ./api ./api
COPY ./artifacts/tokenizer.pkl ./artifacts/tokenizer.pkl
COPY ./artifacts/model-epoch=01-val_loss=1.94.ckpt ./artifacts/model-epoch=01-val_loss=1.94.ckpt
COPY ./src ./src

ENV MODEL_NAME model-epoch=01-val_loss=1.94

CMD ["poetry", "run", "uvicorn" ,"api.main:app"]
