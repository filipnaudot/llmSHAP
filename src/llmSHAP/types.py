from typing import Dict, Set, Any, Iterable, Union, Optional, List, Tuple, ClassVar


# ------------------------------------------------------------------ #
# TypeAlias
# ------------------------------------------------------------------ #
Index          = int
IndexSelection = Union[Index, Iterable[Index]]
DataMapping    = Dict[Any, Any]
ResultMapping  = Dict[str, Dict[str, float]]
Prompt         = list[dict[str, str]]