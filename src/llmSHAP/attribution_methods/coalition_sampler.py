from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Set, Tuple, List
from itertools import combinations
from math import factorial
import random


class CoalitionSampler(ABC):
    @abstractmethod
    def __call__(self, feature: str, variable_keys: List[str]) -> Iterable[Tuple[Set[str], float]]: ...


class FullEnumerationSampler(CoalitionSampler):
    def __init__(self, num_players: int):
        self._num_players = num_players
        self._factorial_cache = {k: factorial(k) for k in range(num_players + 1)}

    def __call__(self, feature: str, keys: List[str]):
        features = [key for key in keys if key != feature]
        num_players = len(keys)

        for coalition_size in range(len(features) + 1):
            weight = self._factorial_cache[coalition_size] * self._factorial_cache[num_players - coalition_size - 1] / self._factorial_cache[self._num_players]
            for coalition in combinations(features, coalition_size):
                yield set(coalition), weight