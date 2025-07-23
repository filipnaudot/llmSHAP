import math
import pytest
from llmSHAP.attribution_methods import FullEnumerationSampler

FEATURES = [
    "feat1", "feat2", "feat3", "feat4", "feat5",
    "feat6", "feat7", "feat8", "feat9", "feat10"
]
N = len(FEATURES)

def shapley_weight(subset_size: int, total_players: int) -> float:
    return (
        math.factorial(subset_size)
        * math.factorial(total_players - subset_size - 1)
        / math.factorial(total_players)
    )

@pytest.mark.parametrize("target", FEATURES)
def test_full_enumeration_sampler_correct_counts_and_weights(target):
    sampler = FullEnumerationSampler(num_players=N)
    coalitions = list(sampler(target, FEATURES))

    assert len(coalitions) == 2 ** (N - 1)

    total_weight = 0.0
    seen = set()
    others = set(FEATURES) - {target}
    for coalition_set, weight in coalitions:
        assert coalition_set.issubset(others)

        s = len(coalition_set)
        expected_w = shapley_weight(s, N)
        assert pytest.approx(weight, rel=1e-12) == expected_w

        total_weight += weight
        seen.add(frozenset(coalition_set))

    # No duplicates.
    assert len(seen) == len(coalitions)
    # Total weights sum to 1.
    assert pytest.approx(total_weight, rel=1e-12) == 1.0