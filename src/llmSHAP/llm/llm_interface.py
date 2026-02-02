from abc import ABC, abstractmethod

from llmSHAP.types import Prompt, Any, Optional

class LLMInterface(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        tools: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
    ) -> str:
        pass
    