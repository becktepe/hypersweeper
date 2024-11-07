from abc import ABC, abstractmethod
from .hypersweeper.utils import Info, Result

class HyperAdapter(ABC):
    @abstractmethod
    def ask(self) -> tuple[Info, bool]:
        pass

    @abstractmethod
    def tell(self, info: Info, result: Result) -> None:
        pass