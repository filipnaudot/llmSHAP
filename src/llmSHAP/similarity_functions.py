from abc import ABC, abstractmethod
from functools import lru_cache



class SimilarityFunction(ABC):
    @abstractmethod
    def __call__(self, s1: str, s2: str) -> float:
        pass


#########################################################
# Basic TFIDF-based Cosine Similarity Funciton.
#########################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFCosineSimilarity(SimilarityFunction):
    _vectorizer = TfidfVectorizer()

    @lru_cache(maxsize=2_000)
    def __call__(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0
        vectors = self._vectorizer.fit_transform([string1, string2])
        return float(cosine_similarity(vectors)[0, 1])


#########################################################
# Embedding-Based Similarity Funciton.
#########################################################
from sentence_transformers import SentenceTransformer, util
from llmSHAP.types import ClassVar

class EmbeddingCosineSimilarity(SimilarityFunction):
    _model: ClassVar[SentenceTransformer | None] = None

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if EmbeddingCosineSimilarity._model is None:
            EmbeddingCosineSimilarity._model = SentenceTransformer(model_name)

    def __call__(self, string1: str, string2: str) -> float:
        if not string1.strip() or not string2.strip(): return 0.0
        assert self._model is not None
        embeddings = self._model.encode([string1, string2], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]))