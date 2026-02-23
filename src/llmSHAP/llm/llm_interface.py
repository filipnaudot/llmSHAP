from abc import ABC, abstractmethod

from llmSHAP.types import Any, Optional

class LLMInterface(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: Any,
        tools: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
    ) -> str:
        pass
    