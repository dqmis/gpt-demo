from typing import Tuple, List, Any
from itertools import chain

from torch.utils.data import DataLoader

from src.data.dataset import TokensDatataset
from src.data.preprocessing.tokenziner import Tokenizer

from sklearn.model_selection import train_test_split


def _chunks(lst: List[Any], chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def _list_to_str(lst: List[List[str]]) -> str:
    return "\n".join(list(chain.from_iterable(lst)))


def _split_data(
    data: str, split_lines_count: int = 4, test_size: float = 0.1
) -> Tuple[str, str]:
    batched_data: List[List[str]] = [
        i for i in _chunks(data.splitlines(), split_lines_count)
    ]
    train_data, test_data = train_test_split(
        batched_data, test_size=test_size, random_state=412
    )
    return _list_to_str(train_data), _list_to_str(test_data)


def split_data_into_loaders(
    data: str, tokenizer: Tokenizer, val_size: float, block_size: int, batch_size: int
) -> Tuple[DataLoader]:
    train_data, val_data = _split_data(data, test_size=val_size)

    # initializing train and validation datasets
    train_dataset = TokensDatataset(tokenizer, train_data, block_size)
    val_dataset = TokensDatataset(tokenizer, val_data, block_size)

    # creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
