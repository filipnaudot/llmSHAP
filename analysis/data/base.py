from abc import ABC, abstractmethod


class DataClass(ABC):
    @abstractmethod
    def concepts(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, concept_index: int | None = None) -> str | list[str]:
        raise NotImplementedError
