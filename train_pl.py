from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.data.dataset import get_dataset
from src.data.loading import split_data_into_loaders
from src.data.preprocessing.tokenziner import MiniGPTTokenizer
from src.helpers.wandb_helpers import parse_list_to_wandb_table
from src.models import MiniGPTLightning
from src.utils.evaluation import generate_text_from_model


@hydra.main(version_base=None, config_path="runs", config_name="config")
def train(raw_cfg: DictConfig):
    run_name = str(datetime.now())
    wandb.init(name=run_name, project="gpt", config=raw_cfg["run"]["run"])

    cfg = raw_cfg["run"]["run"]

    torch.manual_seed(1337)

    # loading and splitting data into train and validation
    data = get_dataset(f"./data/{cfg.dataset}.txt")
    data = data[:1000]

    # initializing and fitting the tokenizer
    tokenizer = MiniGPTTokenizer()
    tokenizer.fit(data)

    # splitting data and loading into dataloaders
    train_dataloader, val_dataloader = split_data_into_loaders(
        data, tokenizer, 0.1, cfg.block_size, cfg.batch_size
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./models/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode="min",
    )

    early_stopping_callback = EarlyStopping("val_loss", patience=2)

    # initializing the model
    model = MiniGPTLightning(
        tokenizer.vocab_size,
        cfg.block_size,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_layer,
        cfg.device,
    )
    model = model.to(cfg.device)

    # training the model using pl trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=cfg.epochs,
        logger=WandbLogger(),
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # loading the best model
    model.load_from_checkpoint(
        checkpoint_callback.best_model_path,
    )

    print(checkpoint_callback.best_model_path)

    # generating output from model
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    generated_text = generate_text_from_model(
        model, tokenizer, context, max_new_tokens=500
    )

    wandb.log({"generated_text": parse_list_to_wandb_table([[generated_text]])})
    wandb.finish()


if __name__ == "__main__":
    train()
