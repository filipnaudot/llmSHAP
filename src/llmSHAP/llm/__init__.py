from llmSHAP.types import TYPE_CHECKING
from typing import overload

__all__ = ["OpenAIInterface", "LangChainInterface"]

if TYPE_CHECKING:
    from .openai import OpenAIInterface
    from .langchain import LangChainInterface

    @overload
    def __getattr__(name: str) -> type[OpenAIInterface]: ...
    @overload
    def __getattr__(name: str) -> type[LangChainInterface]: ...

def __getattr__(name: str):
    if name == "OpenAIInterface":
        from .openai import OpenAIInterface
        return OpenAIInterface
    if name == "LangChainInterface":
        from .langchain import LangChainInterface
        return LangChainInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")