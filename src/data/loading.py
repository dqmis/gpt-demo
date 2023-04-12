from typing import Tuple, List
from itertools import chain

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from src.data.dataset import TokensDatataset
from src.data.preprocessing.tokenziner import Tokenizer


def chunks(lst: List[any], n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def list_to_str(ls: List[List[str]]) -> str:
    return "\n".join(list(chain.from_iterable(ls)))


def split_data(
    data: str, split_lines_count: int = 4, val_size: float = 0.1
) -> Tuple[str, str]:
    batched_data = [i for i in chunks(data.splitlines(), split_lines_count)]
    train_data, val_data = train_test_split(
        batched_data, test_size=val_size, random_state=42
    )

    return list_to_str(train_data), list_to_str(val_data)


def split_data_into_loaders(
    data: str, tokenizer: Tokenizer, val_size: float, block_size: int, batch_size: int
) -> Tuple[DataLoader]:
    train_data, val_data = split_data(data, val_size=val_size)
    # initializing train and validation datasets
    train_dataset = TokensDatataset(tokenizer, train_data, block_size)
    val_dataset = TokensDatataset(tokenizer, val_data, block_size)

    # creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
