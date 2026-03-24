import os
import pytest
from dotenv import load_dotenv

from llmSHAP.llm import OpenAIInterface


def _require_live_openai_setup() -> None:
    pytest.importorskip("openai")
    pytest.importorskip("dotenv")
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")


def _hello_prompt() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Reply with exactly: hello"}]


def _assert_response(response: str) -> None:
    assert isinstance(response, str)
    assert response.strip()
    assert "hello" in response.lower()


def test_generate_gpt_5_4_temperature_zero_live() -> None:
    _require_live_openai_setup()
    llm = OpenAIInterface(model_name="gpt-5.4", temperature=0.0, max_tokens=16)
    response = llm.generate(_hello_prompt())
    _assert_response(response)


def test_generate_gpt_5_4_temperature_point_eight_live() -> None:
    _require_live_openai_setup()
    llm = OpenAIInterface(model_name="gpt-5.4", temperature=0.8, max_tokens=16)
    response = llm.generate(_hello_prompt())
    _assert_response(response)


def test_generate_gpt_5_4_reasoning_low_live() -> None:
    _require_live_openai_setup()
    llm = OpenAIInterface(model_name="gpt-5.4", temperature=0.0, reasoning="low", max_tokens=16)
    response = llm.generate(_hello_prompt())
    _assert_response(response)


def test_generate_gpt_5_2_temperature_zero_live() -> None:
    _require_live_openai_setup()
    llm = OpenAIInterface(model_name="gpt-5.2", temperature=0.0, max_tokens=16)
    response = llm.generate(_hello_prompt())
    _assert_response(response)


def test_generate_gpt_5_2_temperature_point_eight_live() -> None:
    _require_live_openai_setup()
    llm = OpenAIInterface(model_name="gpt-5.2", temperature=0.8, max_tokens=16)
    response = llm.generate(_hello_prompt())
    _assert_response(response)


def test_generate_gpt_5_2_reasoning_medium_live() -> None:
    _require_live_openai_setup()
    llm = OpenAIInterface(model_name="gpt-5.2", temperature=0.0, reasoning="medium", max_tokens=16)
    response = llm.generate(_hello_prompt())
    _assert_response(response)
