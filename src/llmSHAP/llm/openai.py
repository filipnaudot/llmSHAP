import mimetypes
import os
import random
import time

from llmSHAP.types import Optional, Any
from llmSHAP.image import Image
from llmSHAP.llm.llm_interface import LLMInterface

class OpenAIInterface(LLMInterface):
    """
        OpenAI Responses API interface with llmSHAP-managed retry behavior.

        Retries are handled entirely by llmSHAP. 
        The underlying ``OpenAI`` client is
        constructed with ``max_retries=1``, otherwise the retry budget and backoff are controlled
        by ``max_retries``, ``backoff_base``, and ``backoff_max`` on this interface.

        Requests use an explicit default timeout of ``600.0`` seconds (10 minutes) rather
        than inheriting the OpenAI SDK's default timeout.

        :param model_name: OpenAI model identifier to use for generation.
        :param temperature: Sampling temperature. Set to None (default) to omit the parameter for models that do not support temperature.
        :param max_tokens: Maximum number of output tokens to request.
        :param reasoning: Optional reasoning effort for reasoning-capable models.
        :param max_retries: Number of llmSHAP-managed retries after the initial request.
        :param timeout: Request timeout in seconds passed to the underlying OpenAI client.
        :param backoff_base: Base delay in seconds for exponential backoff.
        :param backoff_max: Maximum backoff delay in seconds.
    """
    def __init__(self,
                 *,
                 model_name: str,
                 temperature: Optional[float] = None,
                 max_tokens: int = 512,
                 reasoning: Optional[str] = None,
                 max_retries: int = 5,
                 timeout: float = 600.0,
                 backoff_base: float = 1.0,
                 backoff_max: float = 30.0,):
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
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it (e.g. in your .env) before using OpenAIInterface.")
        self.client: OpenAI = OpenAI(api_key=api_key, max_retries=1, timeout=timeout)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = {"effort": reasoning} if reasoning is not None else None
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max


    def generate(self, prompt: Any, tools: Optional[list[Any]] = None, images: Optional[list[Any]] = None,) -> str:
        if images: prompt = self._attach_images(prompt, images)
        kwargs = {
            "model": self.model_name,
            "input": prompt,
            "max_output_tokens": self.max_tokens,
        }
        if self.reasoning is not None:
            kwargs["reasoning"] = self.reasoning # type: ignore[assignment]
        elif self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return self._generate_with_retries(kwargs)


    def _generate_with_retries(self, kwargs: dict[str, Any]) -> str:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.responses.create(**kwargs) # type: ignore[arg-type]
                return response.output_text or ""
            except RateLimitError as exc:
                if self._is_quota_exhausted(exc):
                    raise RuntimeError(self._format_error(
                        "OpenAI quota exhausted", attempt=attempt, detail=self._extract_error_message(exc),)) from exc
                if attempt >= self.max_retries:
                    raise RuntimeError(self._format_error(
                        "OpenAI rate limit exceeded after retries", attempt=attempt, detail=self._extract_error_message(exc),)) from exc
                time.sleep(self._backoff_seconds(attempt))
            except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(self._format_error(
                        "OpenAI request failed after retries", attempt=attempt, detail=self._extract_error_message(exc),)) from exc
                time.sleep(self._backoff_seconds(attempt))
        raise RuntimeError(self._format_error("OpenAI request failed", attempt=self.max_retries))


    def _backoff_seconds(self, attempt: int) -> float:
        base_delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        return base_delay * (0.5 + random.random())


    def _is_quota_exhausted(self, exc: Exception) -> bool:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                error_type = error.get("type")
                if code == "insufficient_quota" or error_type == "insufficient_quota":
                    return True
        code = getattr(exc, "code", None)
        return code == "insufficient_quota"


    def _extract_error_message(self, exc: Exception) -> str:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
        return str(exc)


    def _format_error(self, message: str, attempt: int, detail: Optional[str] = None) -> str:
        attempts = attempt + 1
        if detail: return f"{message} for model '{self.model_name}' after {attempts} attempt(s): {detail}"
        return f"{message} for model '{self.model_name}' after {attempts} attempt(s)."


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
