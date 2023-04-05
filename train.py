import torch
import hydra
from src.models import MiniGPT
from torch.utils.data import DataLoader
from src.utils.evaluation import estimate_loss
from omegaconf import DictConfig

from tqdm import tqdm

from typing import List

from src.data.preprocessing.tokenziner import MiniGPTTokenizer
from src.data.dataset import get_dataset, ShakespeareTokensDatataset

from matplotlib import pyplot as plt


@hydra.main(version_base=None, config_path="runs", config_name="config")
def train(raw_cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = raw_cfg["run"]["run"]

    torch.manual_seed(1337)

    # loading and splitting data into train and validation
    raw_data = get_dataset("./data/tinyshakespeare.txt")

    # cropping part of data for testing purposes
    raw_data = raw_data[:1000]

    # initializing and fitting the tokenizer
    tokenizer = MiniGPTTokenizer()
    tokenizer.fit(raw_data)

    n = int(0.9 * len(raw_data))  # first 90% will be train, rest val
    train_data = raw_data[:n]
    val_data = raw_data[n:]

    # initializing train and validation datasets
    train_dataset = ShakespeareTokensDatataset(tokenizer, train_data, cfg.block_size)
    val_dataset = ShakespeareTokensDatataset(tokenizer, val_data, cfg.block_size)

    # creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # initializing the model
    model = MiniGPT(
        tokenizer.vocab_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer
    )
    model = model.to(device)

    print(sum(p.numel() for p in model.parameters()), "parameters")

    # TRAINIGN LOOP
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    train_losses = []
    val_losses = []

    for iter in range(cfg.epochs):
        # sample a batch of data
        iter_train_losses: List[float] = []
        for batch in tqdm(train_dataloader):
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            # evaluate the loss
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iter_train_losses.append(loss.item())
        val_loss = estimate_loss(model, val_dataloader, device)
        train_loss = sum(iter_train_losses) / len(iter_train_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("train_loss: ", train_loss)
        print("val_loss: ", val_loss)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.savefig("out.png")


if __name__ == "__main__":
    train()
