import pytest
from llmSHAP import TFIDFCosineSimilarity, EmbeddingCosineSimilarity



@pytest.fixture(scope="module")
def sentences():
    source = (
        "Apple unveiled its latest AI-powered chips, aiming to revolutionize on-device processing "
        "in the next generation of smartphones."
    )
    similar = (
        "The new smartphone features cutting-edge AI chips designed to boost performance and efficiency."
    )
    dissimilar = (
        "A local art gallery opened its summer exhibition with sculptures made entirely from recycled materials."
    )
    return source, similar, dissimilar


@pytest.mark.parametrize("similarity_function", [
    TFIDFCosineSimilarity(),
    EmbeddingCosineSimilarity()
])
def test_similarity_scores_order(sentences, similarity_function):
    source, similar, dissimilar = sentences
    score_sim = similarity_function(source, similar)
    score_diff = similarity_function(source, dissimilar)
    assert score_sim > score_diff

def test_empty_strings_return_zero():
    tfidf = TFIDFCosineSimilarity()
    embedding = EmbeddingCosineSimilarity()
    assert tfidf("", "") == 0.0
    assert tfidf("Hello", "") == 0.0
    assert embedding("", "") == 0.0
    assert embedding("Hello", "") == 0.0