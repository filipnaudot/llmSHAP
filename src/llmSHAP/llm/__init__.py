from llmSHAP.types import TYPE_CHECKING
from typing import overload

__all__ = ["OpenAIInterface", "LangChainInterface", "DummyLLM"]

if TYPE_CHECKING:
    from .llm_interface import LLMInterface
    from .openai import OpenAIInterface
    from .langchain import LangChainInterface
    from .dummy import DummyLLM

    @overload
    def __getattr__(name: str) -> type[LLMInterface]: ...
    @overload
    def __getattr__(name: str) -> type[OpenAIInterface]: ...
    @overload
    def __getattr__(name: str) -> type[LangChainInterface]: ...
    @overload
    def __getattr__(name: str) -> type[DummyLLM]: ...


def __getattr__(name: str):
    if name == "LLMInterface":
        from .llm_interface import LLMInterface
        return LLMInterface
    if name == "OpenAIInterface":
        from .openai import OpenAIInterface
        return OpenAIInterface
    if name == "LangChainInterface":
        from .langchain import LangChainInterface
        return LangChainInterface
    if name == "DummyLLM":
        from .dummy import DummyLLM
        return DummyLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
