import pytest
from llmSHAP.llm import OpenAIInterface


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

@pytest.fixture
def llm() -> OpenAIInterface:
    return OpenAIInterface("gpt-4o-mini")

def test_generate(llm):
    pytest.skip("TODO")