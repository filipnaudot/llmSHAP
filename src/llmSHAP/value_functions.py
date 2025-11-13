from abc import ABC, abstractmethod
from functools import lru_cache

from llmSHAP.types import ClassVar, Optional


class ValueFunction(ABC):
    @abstractmethod
    def __call__(self, s1: str, s2: str) -> float:
        pass


#########################################################
# Basic TFIDF-based Cosine Similarity Funciton.
#########################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFCosineSimilarity(ValueFunction):
    _vectorizer: ClassVar[TfidfVectorizer | None] = None

    def __init__(self):
        if TFIDFCosineSimilarity._vectorizer is None:
            print(f"Initializing TfidfVectorizer...")
            TFIDFCosineSimilarity._vectorizer = TfidfVectorizer()

    @lru_cache(maxsize=2_000)
    def __call__(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0
        assert self._vectorizer is not None
        vectors = self._vectorizer.fit_transform([string1, string2])
        return float(cosine_similarity(vectors)[0, 1])


#########################################################
# Embedding-Based Similarity Funciton.
#########################################################
from sentence_transformers import SentenceTransformer, util

class EmbeddingCosineSimilarity(ValueFunction):
    _model: ClassVar[Optional[SentenceTransformer]] = None

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if EmbeddingCosineSimilarity._model is None:
            print(f"Loading sentence transformer model {model_name}...")
            EmbeddingCosineSimilarity._model = SentenceTransformer(model_name)

    @lru_cache(maxsize=2_000)
    def __call__(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0
        assert self._model is not None
        embeddings = self._model.encode([string1, string2], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]))