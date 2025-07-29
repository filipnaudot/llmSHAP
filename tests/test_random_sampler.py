import pytest
from math import factorial
from collections import Counter

from llmSHAP.attribution_methods import RandomSampler



def test_invalid_sampling_ratio_raises():
    with pytest.raises(AssertionError):
        RandomSampler(0)
    with pytest.raises(AssertionError):
        RandomSampler(1)
    with pytest.raises(AssertionError):
        RandomSampler(-0.1)


def test_no_others_yields_nothing():
    sampler = RandomSampler(0.5, seed=123)
    keys = ["A"]
    result = list(sampler("A", keys))
    assert result == []


def test_leave_one_out_always_included():
    keys = ["A", "B", "C"]
    sampler = RandomSampler(0.5, seed=42)
    results = list(sampler("A", keys))

    # First len(others) yields must be leave‑one‑outs.
    others = [k for k in keys if k != "A"]
    leave_one_out_sets = [set(others[:i] + others[i + 1 :]) for i in range(len(others))]
    sampled_sets = [coalition for coalition, _ in results]

    for loo in leave_one_out_sets:
        assert loo in sampled_sets


def test_sampled_sets_do_not_include_leave_one_outs():
    keys = ["A", "B", "C", "D"]
    sampler = RandomSampler(0.5, seed=0)
    results = list(sampler("A", keys))

    others = [k for k in keys if k != "A"]
    leave_one_out_sets = {frozenset(others[:i] + others[i + 1:]) for i in range(len(others))}
    sampled_sets = [frozenset(c) for c, _ in results]

    # Leave‑one‑outs appear, but extra sampled sets cannot be in leave_one_out_sets.
    extras = sampled_sets[len(leave_one_out_sets):]
    for c in extras:
        assert c not in leave_one_out_sets


def test_weight_matches_expected_kernel_for_leave_one_outs():
    keys = ["A", "B", "C"]
    sampler = RandomSampler(0.5, seed=1)
    results = list(sampler("A", keys))

    # The first len(others) results are leave-one-outs.
    for coalition, weight in results[:2]:
        subset_size = len(coalition)
        total_players = len(keys)
        expected_weight = factorial(subset_size) * factorial(total_players - subset_size - 1) / factorial(total_players)
        assert weight == pytest.approx(expected_weight)


def test_weight_is_adjusted_for_sample_probability():
    # Force sampling_ratio so that we sample 1 from total_remaining=1
    keys = ["A", "B", "C", "D"]
    sampler = RandomSampler(0.5, seed=5)
    others = [k for k in keys if k != "A"]
    total_remaining = (1 << len(others)) - 2 - len(others)
    results = list(sampler("A", keys))

    extras = results[len(others):]
    assert extras
    coalition, weight = extras[0]

    subset_size = len(coalition)
    total_players = len(keys)
    kernel_weight = factorial(subset_size) * factorial(total_players - subset_size - 1) / factorial(total_players)
    selection_prob = max(1, int(0.5 * total_remaining)) / total_remaining
    expected_weight = kernel_weight / selection_prob
    assert weight == pytest.approx(expected_weight)


def test_deterministic_with_seed():
    keys = ["A", "B", "C", "D"]
    sampler1 = RandomSampler(0.3, seed=123)
    sampler2 = RandomSampler(0.3, seed=123)

    results1 = list(sampler1("A", keys))
    results2 = list(sampler2("A", keys))

    assert results1 == results2