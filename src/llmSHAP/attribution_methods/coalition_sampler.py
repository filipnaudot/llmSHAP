from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import combinations
from math import ceil, comb, factorial
import random

from llmSHAP.types import Index, Iterable, Set, Dict, Tuple, List


class CoalitionSampler(ABC):
    @abstractmethod
    def __call__(self, feature: Index, variable_keys: List[Index]) -> Iterable[Tuple[Set[Index], float]]: ...


class CounterfactualSampler(CoalitionSampler):
    def __init__(self):
        pass

    def __call__(self, feature: Index, keys: List[Index]):
        coalition = {k for k in keys if k != feature}
        yield coalition, 1.0


class FullEnumerationSampler(CoalitionSampler):
    def __init__(self, num_players: int):
        self._num_players = num_players
        self._factorial_cache = {k: factorial(k) for k in range(num_players + 1)}

    def __call__(self, feature: Index, keys: List[Index]):
        features = [key for key in keys if key != feature]
        num_players = len(keys)

        for coalition_size in range(len(features) + 1):
            weight = self._factorial_cache[coalition_size] * self._factorial_cache[num_players - coalition_size - 1] / self._factorial_cache[self._num_players]
            for coalition in combinations(features, coalition_size):
                yield set(coalition), weight


class SlidingWindowSampler(CoalitionSampler):
    def __init__(self, ordered_keys: List[Index], w_size: int, stride: int = 1):
        assert w_size >= 1, "w_size must be >= 1"
        self.ordered_keys = ordered_keys
        self.w_size = w_size
        self.stride = stride

        self.windows: List[List[Index]] = []
        for start in range(0, len(ordered_keys) - w_size + 1, stride):
            window = ordered_keys[start:start + w_size]
            if len(window) == 0: break
            self.windows.append(window)

        self.feature2wins: Dict[Index, List[int]] = {k: [] for k in ordered_keys}
        for index, window in enumerate(self.windows):
            for key in window: self.feature2wins[key].append(index)

        self._factorials = {key: factorial(key) for key in range(w_size + 1)}

    def __call__(self, feature: Index, non_permanent_keys: List[Index]):
        window_ids = self.feature2wins.get(feature, [])
        if not window_ids: return

        avg_factor = 1.0 / len(window_ids)
        for win_id in window_ids:
            window = self.windows[win_id]
            window_features = [key for key in window if key != feature]
            outside = set(non_permanent_keys) - set(window)

            for coalition_size in range(len(window_features) + 1):
                weight = (self._factorials[coalition_size] * self._factorials[len(window) - coalition_size - 1] / self._factorials[len(window)]) * avg_factor
                for coalition in combinations(window_features, coalition_size):
                    final_set = set(coalition) | outside
                    yield final_set, weight


class StratifiedSampler(CoalitionSampler):
    def __init__(self, sampling_ratio: float, seed: int | None = None):
        assert 0 < sampling_ratio <= 1, "sampling_ratio must be in (0,1]"
        self.rng = random.Random(seed)
        self.sampling_ratio = sampling_ratio


    def _sample_coalitions(self,
                           others: List[Index],
                           coalition_size: int,
                           sample_count: int,
                           total_count: int) -> Iterable[Set[Index]]:
        if sample_count == total_count:
            for coalition in combinations(others, coalition_size):
                yield set(coalition)
            return
        drawn: set[frozenset[Index]] = set()
        while len(drawn) < sample_count:
            coalition = frozenset(self.rng.sample(others, coalition_size))
            drawn.add(coalition)
        for coalition in drawn:
            yield set(coalition)


    def __call__(self, feature: Index, keys: List[Index]):
        others = [key for key in keys if key != feature]
        num_strata = len(others) + 1
        for coalition_size in range(num_strata):
            total_count = comb(len(others), coalition_size)
            sample_count = ceil(self.sampling_ratio * total_count)
            weight = 1.0 / (num_strata * sample_count)
            for coalition in self._sample_coalitions(others, coalition_size, sample_count,total_count):
                yield coalition, weight
