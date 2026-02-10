import pytest
import sys
import types

from llmSHAP import TFIDFCosineSimilarity, EmbeddingCosineSimilarity
from llmSHAP.generation import Generation


# @pytest.fixture(scope="module")
# def sentences():
#     source = (
#         "Apple unveiled its latest AI-powered chips, aiming to revolutionize on-device processing "
#         "in the next generation of smartphones."
#     )
#     similar = (
#         "The new smartphone features cutting-edge AI chips designed to boost performance and efficiency."
#     )
#     dissimilar = (
#         "A local art gallery opened its summer exhibition with sculptures made entirely from recycled materials."
#     )
#     return source, similar, dissimilar


# @pytest.mark.parametrize("similarity_function", [
#     TFIDFCosineSimilarity(),
#     EmbeddingCosineSimilarity()
# ])
# def test_similarity_scores_order(sentences, similarity_function):
#     source, similar, dissimilar = sentences
#     score_sim = similarity_function(Generation(output=source), Generation(output=similar))
#     score_diff = similarity_function(Generation(output=source), Generation(output=dissimilar))
#     assert score_sim > score_diff

# def test_empty_strings_return_zero():
#     tfidf = TFIDFCosineSimilarity()
#     embedding = EmbeddingCosineSimilarity()
#     assert tfidf(Generation(output=""), Generation(output="")) == 0.0
#     assert tfidf(Generation(output="Hello"), Generation(output="")) == 0.0
#     assert embedding(Generation(output=""), Generation(output="")) == 0.0
#     assert embedding(Generation(output="Hello"), Generation(output="")) == 0.0


def test_openai_embedding_endpoint(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEmbeddingsAPI:
        def create(self, *, model, input):
            captured["model"] = model
            captured["input"] = input
            data = [
                types.SimpleNamespace(embedding=[1.0, 0.0, 0.0]),
                types.SimpleNamespace(embedding=[0.5, 0.5, 0.0]),
            ]
            return types.SimpleNamespace(data=data)

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.embeddings = FakeEmbeddingsAPI()

    fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    fake_dotenv_module = types.SimpleNamespace(load_dotenv=lambda: None)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv_module)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    similarity = EmbeddingCosineSimilarity(api_url_endpoint="https://example.test/v1")
    score = similarity(Generation(output="A"), Generation(output="B"))

    assert captured["api_key"] == "fake-key"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["A", "B"]
    assert score == pytest.approx(0.7071067, rel=1e-6)
