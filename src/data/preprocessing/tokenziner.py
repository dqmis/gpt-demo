from typing import Dict, List
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def fit(self, x: str) -> None:
        pass

    @abstractmethod
    def encode(self, x: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, x: List[int]) -> str:
        pass

    @property
    def vocab_size(self) -> int:
        return len(self._characters)

    @property
    def characters(self) -> List[str]:
        return self._characters


class MiniGPTTokenizer(Tokenizer):
    """
    Character level tokenizer for UTF-8 (almost) alphabet
    """

    def __init__(self, cased: bool = True) -> None:
        self._stoi: Dict[str, int] = {}
        self._itos: Dict[str, int] = {}

        self.cased = cased
        self._characters: List[str] = None

    def fit(self, x: str, cased: bool = True) -> None:
        if not self.cased:
            self._characters = sorted(list(set(x.lower())))
        else:
            self._characters = sorted(list(set(x)))

        self._stoi = {ch: i for i, ch in enumerate(self._characters)}
        self._itos = {i: ch for i, ch in enumerate(self._characters)}

    def encode(self, x: str) -> List[int]:
        if not self.cased:
            x = x.lower()

        # return list(map(lambda s: self._stoi[s], x))
        return [self._stoi[s] for s in x]

    def decode(self, x: List[int]) -> str:
        return "".join([self._itos[s] for s in x])
