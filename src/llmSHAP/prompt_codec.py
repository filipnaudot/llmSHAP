from abc import ABC, abstractmethod

from llmSHAP.types import IndexSelection, Any

from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation



class PromptCodec(ABC):
    @abstractmethod
    def build_prompt(self, data_handler: DataHandler, indexes: IndexSelection) -> Any:
        """(Encode) Build prompt to send to the model."""
        raise NotImplementedError

    @abstractmethod
    def parse_generation(self, model_output: str) -> Generation:
        """(Decode) Parse model generation into a structured result."""
        raise NotImplementedError

    def get_tools(self, data_handler: DataHandler, indexes: IndexSelection) -> list[Any]:
        """Retreive the available tools at the given indexes.
           Defaults to an emty list.
        """
        return []

    def get_images(self, data_handler: DataHandler, indexes: IndexSelection) -> list[Any]:
        """Retreive the available images at the given indexes.
           Defaults to an emty list.
        """
        return []


class BasicPromptCodec(PromptCodec):
    def __init__(self, system: str = ""):
        self.system: str = system
    
    def build_prompt(self, data_handler: DataHandler, indexes: IndexSelection) -> Any:
        return [
            {"role": "system", "content": self.system},
            {"role": "user",   "content": data_handler.to_string(indexes)}
        ]
    
    def get_tools(self, data_handler: DataHandler, indexes: IndexSelection) -> list[Any]:
        return data_handler.tool_list(indexes)

    def get_images(self, data_handler: DataHandler, indexes: IndexSelection) -> list[Any]:
        return data_handler.image_list(indexes)
    
    def parse_generation(self, model_output: str) -> Generation:
        return Generation(output=model_output)