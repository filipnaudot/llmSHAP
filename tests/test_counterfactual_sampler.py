import pytest
from llmSHAP.attribution_methods.coalition_sampler import CounterfactualSampler

FEATURES = [
    "feat1", "feat2", "feat3", "feat4", "feat5",
    "feat6", "feat7", "feat8", "feat9", "feat10"
]

@pytest.mark.parametrize("target", FEATURES)
def test_counterfactual_sampler_yields_full_minus_one(target):
    sampler = CounterfactualSampler()
    results = list(sampler(target, FEATURES))

    assert len(results) == 1

    coalition_set, weight = results[0]
    assert weight == 1.0

    expected = set(FEATURES) - {target}
    assert coalition_set == expected