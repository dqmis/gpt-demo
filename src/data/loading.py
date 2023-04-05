from typing import Tuple

from torch.utils.data import DataLoader

from src.data.dataset import TokensDatataset
from src.data.preprocessing.tokenziner import Tokenizer


def split_data_into_loaders(
    data: str, tokenizer: Tokenizer, val_size: float, block_size: int, batch_size: int
) -> Tuple[DataLoader]:
    n = int((1 - val_size) * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # initializing train and validation datasets
    train_dataset = TokensDatataset(tokenizer, train_data, block_size)
    val_dataset = TokensDatataset(tokenizer, val_data, block_size)

    # creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
