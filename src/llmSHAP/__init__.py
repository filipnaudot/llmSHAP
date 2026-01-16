from llmSHAP.types import TYPE_CHECKING
from typing import overload

__all__ = [
    "DataHandler",
    "PromptCodec",
    "BasicPromptCodec",
    "Generation",
    "ValueFunction",
    "TFIDFCosineSimilarity",
    "EmbeddingCosineSimilarity",
    "ShapleyAttribution",
    "Attribution",
    "Image",
]

if TYPE_CHECKING:
    from .data_handler import DataHandler
    from .prompt_codec import PromptCodec, BasicPromptCodec
    from .generation import Generation
    from .value_functions import ValueFunction, TFIDFCosineSimilarity, EmbeddingCosineSimilarity
    from .attribution_methods.shapley_attribution import ShapleyAttribution
    from .attribution import Attribution
    from .image import Image

    @overload
    def __getattr__(name: str) -> type[DataHandler]: ...
    @overload
    def __getattr__(name: str) -> type[PromptCodec]: ...
    @overload
    def __getattr__(name: str) -> type[BasicPromptCodec]: ...
    @overload
    def __getattr__(name: str) -> type[Generation]: ...
    @overload
    def __getattr__(name: str) -> type[ValueFunction]: ...
    @overload
    def __getattr__(name: str) -> type[TFIDFCosineSimilarity]: ...
    @overload
    def __getattr__(name: str) -> type[EmbeddingCosineSimilarity]: ...
    @overload
    def __getattr__(name: str) -> type[ShapleyAttribution]: ...
    @overload
    def __getattr__(name: str) -> type[Attribution]: ...
    @overload
    def __getattr__(name: str) -> type[Image]: ...

def __getattr__(name: str):
    if name == "DataHandler":
        from .data_handler import DataHandler
        return DataHandler
    if name in {"PromptCodec", "BasicPromptCodec"}:
        from .prompt_codec import PromptCodec, BasicPromptCodec
        return PromptCodec if name == "PromptCodec" else BasicPromptCodec
    if name == "Generation":
        from .generation import Generation
        return Generation
    if name in {"ValueFunction", "TFIDFCosineSimilarity", "EmbeddingCosineSimilarity"}:
        from .value_functions import ValueFunction, TFIDFCosineSimilarity, EmbeddingCosineSimilarity
        return {
            "ValueFunction": ValueFunction,
            "TFIDFCosineSimilarity": TFIDFCosineSimilarity,
            "EmbeddingCosineSimilarity": EmbeddingCosineSimilarity,
        }[name]
    if name == "ShapleyAttribution":
        from .attribution_methods.shapley_attribution import ShapleyAttribution
        return ShapleyAttribution
    if name == "Attribution":
        from .attribution import Attribution
        return Attribution
    if name == "Image":
        from .image import Image
        return Image
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")