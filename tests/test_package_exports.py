def test_stratified_sampler_is_exported_from_package_root():
    from llmSHAP import StratifiedSampler

    assert StratifiedSampler.__name__ == "StratifiedSampler"
