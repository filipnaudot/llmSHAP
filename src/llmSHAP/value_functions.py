from abc import ABC, abstractmethod
from functools import lru_cache

from llmSHAP.types import TYPE_CHECKING, ClassVar, Optional, Any
from llmSHAP.generation import Generation

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer



class ValueFunction(ABC):
    @abstractmethod
    def __call__(self, base_generation: Generation, coalition_generation: Generation) -> float:
        """
        Takes the base (reference / grand-coalition) generation with a
        coalition-specific generation. This allows the user to either
        compare them or focus only on the coalition specific generation.

        Parameters
        ----------
        base:
            The generation from the *full* / reference context. You may ignore
            this if your metric only depends on the coalition.
        coalition:
            The generation produced from a specific coalition (subset of
            features).

        Returns
        -------
        float
            A scalar score.
        """
        raise NotImplementedError


#########################################################
# Basic TFIDF-based Cosine Similarity Funciton.
#########################################################
class TFIDFCosineSimilarity(ValueFunction):
    _vectorizer: ClassVar[Optional["TfidfVectorizer"]] = None
    _cosine_similarity: ClassVar[Optional[Any]] = None

    def __init__(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError(
                "TFIDFCosineSimilarity requires the 'scikit-learn'.\n"
                "Install with: pip install scikit-learn"
            ) from None
        if TFIDFCosineSimilarity._vectorizer is None:
            print(f"Initializing TfidfVectorizer...")
            TFIDFCosineSimilarity._vectorizer = TfidfVectorizer()
            TFIDFCosineSimilarity._cosine_similarity = cosine_similarity

    def __call__(self, g1: Generation, g2: Generation) -> float:
        return self._cached(g1.output, g2.output)
    
    @lru_cache(maxsize=2_000)
    def _cached(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0
        assert self._vectorizer is not None
        assert type(self)._cosine_similarity is not None
        vectors = self._vectorizer.fit_transform([string1, string2])
        return float(type(self)._cosine_similarity(vectors)[0, 1])


#########################################################
# Embedding-Based Similarity Funciton.
#########################################################
class EmbeddingCosineSimilarity(ValueFunction):
    _model: ClassVar[Optional["SentenceTransformer"]] = None
    _util: ClassVar[Optional[Any]] = None

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if EmbeddingCosineSimilarity._model is None:
            try:
                from sentence_transformers import SentenceTransformer, util
            except ImportError:
                raise ImportError(
                    "EmbeddingCosineSimilarity requires the 'embeddings' extra.\n"
                    "Install with: pip install llmSHAP[embeddings]"
                ) from None
            print(f"Loading sentence transformer model {model_name}...")
            EmbeddingCosineSimilarity._model = SentenceTransformer(model_name)
            EmbeddingCosineSimilarity._util = util

    def __call__(self, g1: Generation, g2: Generation) -> float:
        return self._cached(g1.output, g2.output)
    
    @lru_cache(maxsize=2_000)
    def _cached(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0
        assert self._model is not None
        assert self._util is not None
        embeddings = self._model.encode([string1, string2], convert_to_tensor=True)
        return float(self._util.cos_sim(embeddings[0], embeddings[1]))