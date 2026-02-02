from typing import Any, Optional, Callable
import mimetypes

from llmSHAP.types import Prompt, Type
from llmSHAP.image import Image
from llmSHAP.llm.llm_interface import LLMInterface

try:
    from langchain_core.messages import (
        BaseMessage, HumanMessage, AIMessage, SystemMessage
    )
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False



class LangChainInterface(LLMInterface):
    def __init__(
        self,
        chat_model: Any,
        name: Optional[str] = None,
        tool_factory: Optional[Callable[[list[Any]], Any]] = None,
    ):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "LangChainInterface requires langchain-core.\n"
                "Install with: pip install langchain-core"
            ) from None
        self.chat_model = chat_model
        self._name = name or getattr(chat_model, "model_name", chat_model.__class__.__name__)
        self._tool_factory = tool_factory

    def generate(
        self,
        prompt: Prompt,
        tools: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
    ) -> str:
        messages = self._prompt_to_messages(prompt, images=images)
        model = self.chat_model
        if tools:
            if self._tool_factory is not None:
                model = self._tool_factory(tools)
            elif hasattr(model, "bind_tools"):
                try:
                    model = model.bind_tools(tools)
                except Exception:
                    model = self.chat_model
        try:
            result = model.invoke(messages)
        except Exception as exc:
            try:
                result = model.invoke({"messages": messages})
            except Exception:
                raise exc
            if isinstance(result, dict) and result.get("messages"):
                last = result["messages"][-1]
                return getattr(last, "content", str(last)) or ""
        return getattr(result, "content", str(result)) or ""
    
    def _prompt_to_messages(self, prompt: Prompt, images: Optional[list[Any]] = None):
        role_map: dict[str, Type[BaseMessage]] = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        content_blocks = [
            {"type": "image_url", "image_url": {"url": image.url}}
            if image.url else
            {"type": "image_url", "image_url": {"url": image.data_url(mimetypes.guess_type(image.image_path)[0] or "image/png")}} # type: ignore
            for image in images or [] if isinstance(image, Image) and (image.url or image.image_path)
        ]
        
        messages = []
        for item in prompt:
            message_class = role_map.get(item.get("role", "user")) or HumanMessage
            if content_blocks and item.get("role") == "user":
                text = item.get("content", "")
                messages.append(message_class(content=[{"type": "text", "text": text}, *content_blocks]))
                content_blocks = [] # Reset so the images do not get duplicated
            else:
                messages.append(message_class(content=item.get("content", "")))
        return messages
