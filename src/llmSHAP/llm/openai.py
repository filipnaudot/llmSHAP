import gc
import os

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

from llmSHAP.types import Prompt
from llmSHAP.llm.llm_interface import LLMInterface



class OpenAIInterface(LLMInterface):
    def __init__(self, model_name: str):
        self.client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def generate(self,
                 prompt: Prompt,
                 *,
                 max_tokens: int = 512,
                 temperature:int = 0,
                 seed:int = 42):
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
        return response.choices[0].message.content

    def is_local(self): return False

    def name(self): return self.model_name

    def cleanup(self):
        del self.client
        self.client = None
        del self.model_name
        self.model_name = None
        gc.collect()