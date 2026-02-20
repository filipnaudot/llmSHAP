from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
import math
import os
import re

from llmSHAP.types import TYPE_CHECKING, ClassVar, Optional, Any
from llmSHAP.generation import Generation

if TYPE_CHECKING:
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
    """
    Minimal TF-IDF cosine similarity between two generation outputs.

    Notes
    -----
    - Computes TF-IDF on the compared pair only (2-document corpus).
    - Returns `0.0` if either text is empty/whitespace.
    - Tokenization uses the regex `(?u)\\b\\w\\w+\\b`:
      includes 2+ character word tokens and splits on punctuation.

    Example
    -------
    `"hello, world!"` -> `["hello", "world"]`
    `"state-of-the-art"` -> `["state", "of", "the", "art"]`
    `"a b c test"` -> `["test"]`
    """
    _token_pattern: ClassVar[re.Pattern[str]] = re.compile(r"(?u)\b\w\w+\b")
    _DOCUMENT_COUNT = 2

    def __call__(self, g1: Generation, g2: Generation) -> float:
        return self._cached(g1.output, g2.output)
    
    @lru_cache(maxsize=2_000)
    def _cached(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0

        term_counts_document_1 = Counter(self._token_pattern.findall(string1.lower()))
        term_counts_document_2 = Counter(self._token_pattern.findall(string2.lower()))
        if not term_counts_document_1 or not term_counts_document_2: return 0.0

        all_terms = set(term_counts_document_1) | set(term_counts_document_2)
        term_idf: dict[str, float] = {}
        for term in all_terms:
            document_frequency = int(term in term_counts_document_1) + int(term in term_counts_document_2)
            term_idf[term] = math.log((1.0 + self._DOCUMENT_COUNT) / (1.0 + document_frequency)) + 1.0

        term_tfidf_document_1 = {term: float(term_count) * term_idf[term] 
                                 for term, term_count in term_counts_document_1.items()}
        term_tfidf_document_2 = {term: float(term_count) * term_idf[term]
                                 for term, term_count in term_counts_document_2.items()}

        if len(term_tfidf_document_1) > len(term_tfidf_document_2):
            term_tfidf_document_1, term_tfidf_document_2 = term_tfidf_document_2, term_tfidf_document_1

        dot_product = sum(tfidf_weight * term_tfidf_document_2.get(term, 0.0)
                          for term, tfidf_weight in term_tfidf_document_1.items())
        
        norm_document_1 = math.sqrt(sum(weight * weight for weight in term_tfidf_document_1.values()))
        norm_document_2 = math.sqrt(sum(weight * weight for weight in term_tfidf_document_2.values()))
        if norm_document_1 == 0.0 or norm_document_2 == 0.0: return 0.0
        return dot_product / (norm_document_1 * norm_document_2)



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
