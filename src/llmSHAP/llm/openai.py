import gc, os

from llmSHAP.types import Prompt, Optional
from llmSHAP.llm.llm_interface import LLMInterface

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class OpenAIInterface(LLMInterface):
    def __init__(self,
                 model_name: str,
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 seed: Optional[int] = None):
        if not _HAS_OPENAI:
            raise ImportError(
                "OpenAIInterface requires the 'openai' extra.\n"
                "Install with: pip install llmSHAP[openai]"
            ) from None
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("OPENAI_API_KEY is not set. Set it (e.g. in your .env) before using OpenAIInterface.")
        self.client: OpenAI = OpenAI(api_key=api_key)
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