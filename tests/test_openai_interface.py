import os
from dataclasses import dataclass

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from llmSHAP import DataHandler, Generation, PromptCodec, ShapleyAttribution, TFIDFCosineSimilarity
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
    llm = OpenAIInterface(model_name="gpt-5.4", temperature=0.0, reasoning="low", max_tokens=128)
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
    llm = OpenAIInterface(model_name="gpt-5.2", temperature=0.0, reasoning="medium", max_tokens=128)
    response = llm.generate(_hello_prompt())
    _assert_response(response)


def test_generate_structured_output_with_custom_prompt_codec_live() -> None:
    _require_live_openai_setup()
    class LoanDecision(BaseModel):
        rationale: list[str]
        recommendation: str

    @dataclass
    class LoanGeneration(Generation):
        rationale: str
        recommendation: str

    class RecommendationCodec(PromptCodec):
        def build_prompt(self, data_handler: DataHandler, indexes) -> list[dict[str, str]]:
            return [{"role": "system", "content": "You are a careful loan reviewer. Explain the decision briefly and give a recommendation."},
                    {"role": "user", "content": data_handler.to_string(indexes)}]
        def parse_generation(self, model_output) -> Generation:
            rationale = "\n".join(model_output.rationale)
            return LoanGeneration(output="\n".join([rationale, model_output.recommendation]), rationale=rationale, recommendation=model_output.recommendation)

    class RecommendationSimilarity(TFIDFCosineSimilarity):
        def __call__(self, g1: Generation, g2: Generation) -> float:
            assert isinstance(g1, LoanGeneration)
            assert isinstance(g2, LoanGeneration)
            return self._cached(g1.rationale, g2.rationale) * float(g1.recommendation == g2.recommendation)

    result = ShapleyAttribution(
        model=OpenAIInterface(model_name="gpt-4.1-mini", text_format=LoanDecision),
        data_handler=DataHandler({
            "task": "Should this applicant be approved for a loan?",
            "income": "Annual income: $145,000.",
            "debt": "Debt-to-income ratio: 10%.",
            "history": "Many missed payments in the last 7 years.",
        }, permanent_keys={"task"}),
        prompt_codec=RecommendationCodec(),
        use_cache=True,
        verbose=False,
        value_function=RecommendationSimilarity(),
    ).attribution()
    assert isinstance(result.output, str)
    assert result.output.strip()
    assert result.attribution
