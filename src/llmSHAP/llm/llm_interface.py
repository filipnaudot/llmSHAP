from abc import ABC, abstractmethod

from llmSHAP.types import Prompt, Any, Optional

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: Prompt, tools: Optional[list[Any]] = None) -> str:
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_local(slef) -> bool:
        pass

    @abstractmethod
    def cleanup(self):
        pass