import pytest
from math import ceil, comb

from llmSHAP.attribution_methods import StratifiedSampler


def test_invalid_sampling_ratio_raises():
    with pytest.raises(AssertionError):
        StratifiedSampler(0)
    with pytest.raises(AssertionError):
        StratifiedSampler(1.1)
    with pytest.raises(AssertionError):
        StratifiedSampler(-0.1)


def test_single_feature_yields_empty_coalition():
    sampler = StratifiedSampler(0.5, seed=123)
    assert list(sampler("A", ["A"])) == [(set(), 1.0)] # type: ignore


def test_samples_each_stratum_and_weights_sum_to_one():
    keys = ["A", "B", "C", "D", "E"]
    sampler = StratifiedSampler(0.5, seed=42)
    results = list(sampler("A", keys)) # type: ignore
    counts_by_size = {
        size: sum(len(coalition) == size for coalition, _ in results)
        for size in range(len(keys))
    }
    for size, count in counts_by_size.items():
        total_count = comb(len(keys) - 1, size)
        assert count == ceil(0.5 * total_count)
    assert sum(weight for _, weight in results) == pytest.approx(1.0)


def test_target_alone_and_target_left_out_cases_are_always_included():
    keys = ["A", "B", "C", "D", "E"]
    sampler = StratifiedSampler(0.1, seed=0)
    results = list(sampler("A", keys)) # type: ignore
    coalitions = [coalition for coalition, _ in results]
    assert set() in coalitions
    assert {"B", "C", "D", "E"} in coalitions


def test_each_stratum_has_equal_total_weight():
    keys = ["A", "B", "C", "D", "E"]
    sampler = StratifiedSampler(0.5, seed=5)
    results = list(sampler("A", keys)) # type: ignore
    for size in range(len(keys)):
        total_weight = sum(weight for coalition, weight in results if len(coalition) == size)
        assert total_weight == pytest.approx(1 / len(keys))


def test_deterministic_with_seed():
    keys = ["A", "B", "C", "D", "E", "F"]
    sampler1 = StratifiedSampler(0.3, seed=123)
    sampler2 = StratifiedSampler(0.3, seed=123)
    assert list(sampler1("A", keys)) == list(sampler2("A", keys)) # type: ignore
