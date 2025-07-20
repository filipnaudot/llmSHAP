from dataclasses import dataclass
from llmSHAP.types import Optional


@dataclass
class Generation:
    output: str
    score: Optional[int] = None