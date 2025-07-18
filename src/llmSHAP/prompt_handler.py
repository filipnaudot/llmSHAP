from abc import ABC, abstractmethod

from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation
from llmSHAP.types import IndexSelection



class PromptHandler(ABC):
    """
    Abstract base class for building structured prompts and parsing model outputs.
    """

    @abstractmethod
    def build_prompt(self, data_handler: DataHandler, indexes: IndexSelection) -> str:
        """
        Construct and return a prompt string based on the given data and index(es).

        Args:
            data_handler (DataHandler): Provides access to data required for the prompt.
            indexes (IndexSelection): Index or iterable of indices specifying data entries.

        Returns:
            str: The formatted prompt ready for LLM input.
        """
        pass

    @abstractmethod
    def parse_generation(self, model_output: str) -> Generation:
        """
        Parse and return the structured generation from the model output.

        Args:
            model_output (str): The raw output from the language model.

        Returns:
            Generation: The structured representation of the generation.
        """
        pass