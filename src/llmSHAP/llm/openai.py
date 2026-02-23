import gc, os, mimetypes

from llmSHAP.types import Optional, Any
from llmSHAP.image import Image
from llmSHAP.llm.llm_interface import LLMInterface

class OpenAIInterface(LLMInterface):
    def __init__(self,
                 model_name: str,
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 seed: Optional[int] = None,
                 reasoning: Optional[str] = None,):
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
        except ImportError:
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
        self.reasoning = {"effort": reasoning}

    def generate(
        self,
        prompt: Any,
        tools: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
    ) -> str:
        if images: prompt = self._attach_images(prompt, images)
        kwargs = dict(
            model=self.model_name,
            input=prompt,
            max_output_tokens=self.max_tokens,
        )
        if self.reasoning is not None: kwargs["reasoning"] = self.reasoning # type: ignore[arg-type]
        elif self.model_name in {"gpt-5.1", "gpt-5.2"}: kwargs["temperature"] = self.temperature # type: ignore[arg-type]
        response = self.client.responses.create(**kwargs) # type: ignore[arg-type]
        return response.output_text or ""

    def _attach_images(self, prompt: Any, images: list[Any]) -> Any:
        content_blocks: list[dict[str, Any]] = []
        for item in images:
            if isinstance(item, Image):
                if item.url:
                    content_blocks.append({"type": "input_image", "image_url": item.url})
                elif item.image_path:
                    mime = mimetypes.guess_type(item.image_path)[0] or "image/png"
                    content_blocks.append({"type": "input_image", "image_url": item.data_url(mime)})
        
        if not content_blocks: return prompt
        
        updated_prompt: Any = []
        attached = False
        for message in prompt:
            if not attached and message.get("role") == "user":
                text = message.get("content", "")
                updated_prompt.append({"role": "user", "content": [{"type": "input_text", "text": text}, *content_blocks]}) # type: ignore
                attached = True
            else:
                updated_prompt.append(message)
        return updated_prompt if attached else prompt