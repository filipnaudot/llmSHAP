import pytest
from llmSHAP.llm import OpenAIInterface


    
@pytest.fixture
def llm() -> OpenAIInterface:
    return OpenAIInterface("gpt-4o-mini")

def test_name_returns_str(llm):
    value = llm.name()
    assert isinstance(value, str)
    assert value == "gpt-4o-mini"

def test_is_local_returns_bool(llm):
    value = llm.is_local()
    assert isinstance(value, bool)
    assert value == False

def test_generate(llm):
    pytest.skip("TODO")