from abc import ABC, abstractmethod
from functools import lru_cache
import os

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
        return float(type(self)._cosine_similarity(vectors)[0, 1]) # type: ignore


#########################################################
# Embedding-Based Similarity Funciton.
#########################################################
class EmbeddingCosineSimilarity(ValueFunction):
    """
    Embedding-based cosine similarity between two generations.

    This value function supports two backends:
    1. Local sentence-transformers model (default).
    2. OpenAI-compatible embeddings API when ``api_url_endpoint`` is provided.

    Parameters
    ----------
    model_name:
        Embedding model identifier.
        - Local mode default: ``sentence-transformers/all-MiniLM-L6-v2``.
        - API mode: if left as the local default, it is automatically mapped to
          ``text-embedding-3-small``.
    api_url_endpoint:
        Optional base URL for an OpenAI-compatible API endpoint
        (for example ``https://api.openai.com/v1`` or a self-hosted proxy).
        When set, local sentence-transformers are not initialized.

    Environment
    -----------
    OPENAI_API_KEY:
        Required only when ``api_url_endpoint`` is provided.

    Notes
    -----
    - Returns ``0.0`` if either compared output is empty/whitespace.
    - Uses an internal LRU cache to avoid recomputing repeated pairs.
    """
    DEFAULT_LOCAL_EMBEDDING_MODEL: ClassVar[str] = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_API_EMBEDDING_MODEL: ClassVar[str] = "text-embedding-3-small"
    _model: ClassVar[Optional["SentenceTransformer"]] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_url_endpoint: Optional[str] = None,
    ):
        self._api_client: Optional[Any] = None
        resolved_model_name = model_name or self.DEFAULT_LOCAL_EMBEDDING_MODEL
        self._api_model_name: str = resolved_model_name

        if api_url_endpoint:
            try:
                from openai import OpenAI
                from dotenv import load_dotenv
            except ImportError:
                raise ImportError(
                    "EmbeddingCosineSimilarity with api_url_endpoint requires the 'openai' extra.\n"
                    "Install with: pip install llmSHAP[openai]"
                ) from None

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Set it before using api_url_endpoint.")

            if resolved_model_name == self.DEFAULT_LOCAL_EMBEDDING_MODEL:
                self._api_model_name = self.DEFAULT_API_EMBEDDING_MODEL
            self._api_client = OpenAI(api_key=api_key, base_url=api_url_endpoint)
            return

        if EmbeddingCosineSimilarity._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "EmbeddingCosineSimilarity requires the 'embeddings' extra.\n"
                    "Install with: pip install llmSHAP[embeddings]"
                ) from None
            print(f"Loading sentence transformer model {resolved_model_name}...")
            EmbeddingCosineSimilarity._model = SentenceTransformer(resolved_model_name)

    def __call__(self, g1: Generation, g2: Generation) -> float:
        return self._cached(g1.output, g2.output)
    
    @lru_cache(maxsize=2_000)
    def _cached(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0
        if self._api_client is not None:
            response = self._api_client.embeddings.create(model=self._api_model_name, input=[string1, string2])
            embedding1 = response.data[0].embedding
            embedding2 = response.data[1].embedding
            return self._cosine_from_vectors(embedding1, embedding2)
        assert self._model is not None
        embeddings = self._model.encode([string1, string2], convert_to_numpy=True)
        return self._cosine_from_vectors(embeddings[0], embeddings[1])

    @staticmethod
    def _cosine_from_vectors(vector1: Any, vector2: Any) -> float:
        import numpy as np
        array1 = np.asarray(vector1, dtype=float)
        array2 = np.asarray(vector2, dtype=float)
        if array1.shape != array2.shape:
            raise ValueError("Embedding vectors must have the same length.")
        dot = float(np.dot(array1, array2))
        norm1 = float(np.linalg.norm(array1))
        norm2 = float(np.linalg.norm(array2))
        if norm1 == 0.0 or norm2 == 0.0: return 0.0
        return dot / (norm1 * norm2)
