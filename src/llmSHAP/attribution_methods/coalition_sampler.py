from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import combinations
from math import factorial
import random

from llmSHAP.types import Iterable, Set, Dict, Tuple, List


class CoalitionSampler(ABC):
    @abstractmethod
    def __call__(self, feature: str, variable_keys: List[str]) -> Iterable[Tuple[Set[str], float]]: ...


class CounterfactualSampler(CoalitionSampler):
    def __init__(self):
        pass

    def __call__(self, feature: str, keys: List[str]):
        features = [key for key in keys if key != feature]
        feature_set = set(features)
        for f in features:
            yield feature_set - {f}, 1.0


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


class SlidingWindowSampler(CoalitionSampler):
    def __init__(self, ordered_keys: List[str], w_size: int, stride: int = 1):
        assert w_size >= 1, "w_size must be >= 1"
        self.ordered_keys = ordered_keys
        self.w_size = w_size
        self.stride = stride

        self.windows: List[List[str]] = []
        for start in range(0, len(ordered_keys), stride):
            window = ordered_keys[start:start + w_size]
            if len(window) == 0: break
            self.windows.append(window)

        self.feature2wins: Dict[str, List[int]] = {k: [] for k in ordered_keys}
        for index, window in enumerate(self.windows):
            for k in window: self.feature2wins[k].append(index)

        self._fact = {k: factorial(k) for k in range(w_size + 1)}

    def __call__(self, feature: str, variable_keys: List[str]):
        window_ids = self.feature2wins.get(feature, [])
        if not window_ids: return

        avg_factor = 1.0 / len(window_ids)
        for win_id in window_ids:
            window = self.windows[win_id]
            window_features = [k for k in window if k != feature]
            outside = set(variable_keys) - set(window)

            for coalition_size in range(len(window_features) + 1):
                weight = (self._fact[coalition_size] * self._fact[len(window) - coalition_size - 1] / self._fact[len(window)]) * avg_factor
                for coalition in combinations(window_features, coalition_size):
                    final_set = set(coalition) | outside
                    yield final_set, weight