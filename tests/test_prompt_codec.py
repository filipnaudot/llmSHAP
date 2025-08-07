import pytest
from llmSHAP import Generation, PromptCodec, BasicPromptCodec


class MockDataHandler:
    def __init__(self): pass
    def to_string(self, indexes): return "Mock message"


def test_prompt_codec_is_abstract():
    # Should not be able to instantiate the abstract base class.
    with pytest.raises(TypeError): PromptCodec()

def test_build_prompt_includes_system_and_user_roles():
    mock_datahandler = MockDataHandler()
    codec = BasicPromptCodec(system="SYSTEM_MSG")
    prompt = codec.build_prompt(mock_datahandler, indexes=[1])

    assert prompt == [
        {"role": "system", "content": "SYSTEM_MSG"},
        {"role": "user",   "content": "Mock message"},
    ]

def test_build_prompt_default_system_is_empty_string():
    mock_datahandler = MockDataHandler()
    codec = BasicPromptCodec()  # System defaults to "".
    prompt = codec.build_prompt(mock_datahandler, indexes=[1])
    assert prompt[0]["role"] == "system"
    assert prompt[0]["content"] == ""
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"] == "Mock message"

def test_parse_generation_wraps_output_in_generation():
    codec = BasicPromptCodec()
    generation = codec.parse_generation("model-response")
    assert isinstance(generation, Generation)
    assert generation.output == "model-response"

def test_parse_generation_handles_empty_string():
    generation = BasicPromptCodec().parse_generation("")
    assert generation.output == ""