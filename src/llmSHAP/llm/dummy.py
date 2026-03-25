import random as rand
import string
import time

from llmSHAP.types import Optional, Any
from llmSHAP.llm.llm_interface import LLMInterface


class DummyLLM(LLMInterface):
    """
    Lightweight deterministic backend for local benchmarking.

    It matches the constructor shape of ``OpenAIInterface`` so callers can
    swap implementations without changing benchmark setup code.
    """

    def __init__(self,
                 *,
                 model_name: str,
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 reasoning: Optional[str] = None,
                 sleep_seconds: float = 0.02,
                 random: bool = False,
                 response_text: str = "DUMMY_RESPONSE",
                 **_: Any,):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.sleep_seconds = sleep_seconds
        self.random = random
        self.response_text = response_text

    def generate(
        self,
        prompt: Any,
        tools: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
    ) -> str:
        time.sleep(self.sleep_seconds)
        if self.random: return "".join(rand.choices(string.ascii_letters + string.digits, k=10))
        return self.response_text
