from abc import ABC, abstractmethod


class DataClass(ABC):
    @abstractmethod
    def concepts(self) -> list[str]:
        raise NotImplementedError
