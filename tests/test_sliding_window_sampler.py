import pytest
from llmSHAP.attribution_methods import SlidingWindowSampler

FEATURES = [
    "feat1", "feat2", "feat3", "feat4", "feat5",
    "feat6", "feat7", "feat8", "feat9", "feat10"
]
WINDOW_SIZE = 4
STRIDE = 2

@pytest.mark.parametrize("target", FEATURES)
def test_sliding_window_sampler_windows_cover_and_weights(target):
    sampler = SlidingWindowSampler(
        ordered_keys=FEATURES,
        w_size=WINDOW_SIZE,
        stride=STRIDE
    )
    coalitions = list(sampler(target, FEATURES))

    assert coalitions, "No coalitions generated for feature in windows"

    total_weight = 0.0
    window_ids = sampler.feature2wins[target]
    assert window_ids, "Feature not found in any sliding window"

    for coalition_set, weight in coalitions:
        total_weight += weight

        # Target must NOT be in coalition.
        assert target not in coalition_set

        # Outside-of-window features must always be included.
        assert any(
            (set(FEATURES) - set(sampler.windows[w_id])).issubset(coalition_set)
            for w_id in window_ids
        )

    # Correct numnber of coalitions.
    window_ids = sampler.feature2wins[target]
    expected_count = sum(
        2 ** (len(sampler.windows[w_id]) - 1)
        for w_id in window_ids
    )
    assert len(coalitions) == expected_count

    # Total weights sum to 1.
    assert pytest.approx(total_weight, rel=1e-12) == 1.0