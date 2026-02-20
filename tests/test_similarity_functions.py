import pytest
import sys
import types

from llmSHAP import TFIDFCosineSimilarity, EmbeddingCosineSimilarity
from llmSHAP.generation import Generation


@pytest.fixture(scope="module")
def sample_sentences():
    source_sentence = (
        "Apple unveiled its latest AI-powered chips, aiming to revolutionize on-device processing "
        "in the next generation of smartphones."
    )
    similar_sentence = (
        "The new smartphone features cutting-edge AI chips designed to boost performance and efficiency."
    )
    dissimilar_sentence = (
        "A local art gallery opened its summer exhibition with sculptures made entirely from recycled materials."
    )
    return source_sentence, similar_sentence, dissimilar_sentence


def test_tfidf_similarity_scores_order(sample_sentences):
    source_sentence, similar_sentence, dissimilar_sentence = sample_sentences
    tfidf_similarity = TFIDFCosineSimilarity()

    similar_score = tfidf_similarity(Generation(output=source_sentence), Generation(output=similar_sentence))
    dissimilar_score = tfidf_similarity(Generation(output=source_sentence), Generation(output=dissimilar_sentence))

    assert similar_score > dissimilar_score


def test_tfidf_empty_strings_return_zero():
    tfidf_similarity = TFIDFCosineSimilarity()
    assert tfidf_similarity(Generation(output=""), Generation(output="")) == 0.0
    assert tfidf_similarity(Generation(output="Hello"), Generation(output="")) == 0.0
    assert tfidf_similarity(Generation(output=""), Generation(output="Hello")) == 0.0


def test_tfidf_identical_text_returns_one():
    tfidf_similarity = TFIDFCosineSimilarity()
    score = tfidf_similarity(
        Generation(output="minimal performant clean tfidf"),
        Generation(output="minimal performant clean tfidf"),
    )
    assert score == pytest.approx(1.0, rel=1e-12)


def test_tfidf_similarity_is_symmetric():
    tfidf_similarity = TFIDFCosineSimilarity()
    first_text = "alpha beta gamma gamma"
    second_text = "alpha gamma delta"
    forward_score = tfidf_similarity(Generation(output=first_text), Generation(output=second_text))
    reverse_score = tfidf_similarity(Generation(output=second_text), Generation(output=first_text))
    assert forward_score == pytest.approx(reverse_score, rel=1e-12)


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
