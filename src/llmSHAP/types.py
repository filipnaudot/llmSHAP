from typing import (
    TYPE_CHECKING,
    Type,
    Dict,
    Set,
    Any,
    Iterable,
    Union,
    Optional,
    List,
    Tuple,
    ClassVar,
    Callable,
)


# ------------------------------------------------------------------ #
# TypeAlias
# ------------------------------------------------------------------ #
Index          = int
IndexSelection = Union[Index, Iterable[Index]]
DataMapping    = Dict[Any, Any]
ResultMapping  = Dict[str, Dict[str, float]]
