from abc import ABC, abstractmethod

from llmSHAP.types import Prompt

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: Prompt, max_tokens: int) -> str:
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