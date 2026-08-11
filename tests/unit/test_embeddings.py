from semantic_cache.embeddings.mock_embeddings import canonicalize_for_mock, hashed_embedding


def test_paraphrase_canonicalization() -> None:
    left = canonicalize_for_mock("What is Python?")
    right = canonicalize_for_mock("Explain Python to me.")
    assert left == right == "explain python"
    prefixed = canonicalize_for_mock("user: What is Python?")
    assert prefixed == "explain python"


def test_similar_prompts_have_high_cosine() -> None:
    left = hashed_embedding("What is Python?", 256)
    right = hashed_embedding("Explain Python to me.", 256)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    assert dot >= 0.95
