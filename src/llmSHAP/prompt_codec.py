from abc import ABC, abstractmethod

from llmSHAP.types import IndexSelection, Prompt

from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation



class PromptCodec(ABC):
    @abstractmethod
    def build_prompt(self, data_handler: DataHandler, indexes: IndexSelection) -> Prompt:
        """(Encode) Build prompt to send to the model."""
        raise NotImplementedError

    @abstractmethod
    def parse_generation(self, model_output: str) -> Generation:
        """(Decode) Parse model generation into a structured result."""
        raise NotImplementedError


class BasicPromptCodec(PromptCodec):
    def __init__(self, system: str = ""):
        self.system: str = system
    
    def build_prompt(self, data_handler: DataHandler, indexes: IndexSelection) -> Prompt:
        return [
            {"role": "system", "content": self.system},
            {"role": "user",   "content": data_handler.to_string(indexes)}
        ]
    
    def parse_generation(self, model_output: str) -> Generation:
        return Generation(output=model_output)