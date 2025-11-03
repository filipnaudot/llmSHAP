import gc
import os

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

from llmSHAP.types import Prompt, Optional
from llmSHAP.llm.llm_interface import LLMInterface



class OpenAIInterface(LLMInterface):
    def __init__(self, model_name: str, temperature: float = 0, max_tokens: int = 512, seed: Optional[int] = None):
        self.client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self, prompt: Prompt) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.seed is not None: kwargs["seed"] = self.seed
        response = self.client.chat.completions.create(**kwargs) # type: ignore[arg-type]

        return response.choices[0].message.content or ""

    def is_local(self): return False

    def name(self): return self.model_name

    def cleanup(self):
        gc.collect()