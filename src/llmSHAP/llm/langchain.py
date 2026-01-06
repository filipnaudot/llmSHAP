from typing import Any, Optional

from llmSHAP.types import Prompt, Type
from llmSHAP.llm.llm_interface import LLMInterface

try:
    from langchain_core.messages import (
        BaseMessage, HumanMessage, AIMessage, SystemMessage
    )
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False



class LangChainInterface(LLMInterface):
    def __init__(self, chat_model: Any, name: Optional[str] = None, is_local: bool = False):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "LangChainInterface requires langchain-core.\n"
                "Install with: pip install langchain-core"
            ) from None
        self.chat_model = chat_model
        self._name = name or getattr(chat_model, "model_name", chat_model.__class__.__name__)
        self._is_local = is_local

    def generate(self, prompt: Prompt) -> str:
        messages = self._prompt_to_messages(prompt)
        try:
            result = self.chat_model.invoke(messages)
        except Exception as exc:
            try:
                result = self.chat_model.invoke({"messages": messages})
            except Exception:
                raise exc
            if isinstance(result, dict) and result.get("messages"):
                last = result["messages"][-1]
                return getattr(last, "content", str(last)) or ""
        return getattr(result, "content", str(result)) or ""
    
    def _prompt_to_messages(self, prompt: Prompt):
        role_map: dict[str, Type[BaseMessage]] = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        messages = []
        for item in prompt:
            msg_cls = role_map.get(item.get("role", "user")) or HumanMessage
            messages.append(msg_cls(content=item.get("content", "")))
        return messages

    def is_local(self) -> bool:
        return self._is_local

    def name(self) -> str:
        return self._name

    def cleanup(self):
        pass