from functools import lru_cache
from typing import Iterable

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token]


def _fallback_embedding(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(1, dtype=np.float32)

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    values = np.asarray(list(counts.values()), dtype=np.float32)
    norm = np.linalg.norm(values)
    if norm == 0:
        return values
    return values / norm


def _fallback_similarity(text_a: str, text_b: str) -> float:
    counts_a: dict[str, int] = {}
    counts_b: dict[str, int] = {}

    for token in _tokenize(text_a):
        counts_a[token] = counts_a.get(token, 0) + 1
    for token in _tokenize(text_b):
        counts_b[token] = counts_b.get(token, 0) + 1

    vocab = sorted(set(counts_a) | set(counts_b))
    if not vocab:
        return 0.0

    vec_a = np.asarray([counts_a.get(token, 0) for token in vocab], dtype=np.float32)
    vec_b = np.asarray([counts_b.get(token, 0) for token in vocab], dtype=np.float32)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def get_embedding(text: str) -> np.ndarray:
    try:
        model = _get_model()
        embedding = model.encode([text], normalize_embeddings=True)
        return np.asarray(embedding[0], dtype=np.float32)
    except Exception:
        return _fallback_embedding(text)


def cosine_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        embedding_a = get_embedding(text_a)
        embedding_b = get_embedding(text_b)
        if embedding_a.shape == embedding_b.shape:
            similarity = float(np.dot(embedding_a, embedding_b))
            return max(-1.0, min(1.0, similarity))
    except Exception:
        pass

    return max(-1.0, min(1.0, _fallback_similarity(text_a, text_b)))


def batch_embed(texts: Iterable[str]) -> list[np.ndarray]:
    cleaned_texts = list(texts)
    if not cleaned_texts:
        return []
    try:
        model = _get_model()
        embeddings = model.encode(cleaned_texts, normalize_embeddings=True)
        return [np.asarray(vector, dtype=np.float32) for vector in embeddings]
    except Exception:
        return [_fallback_embedding(text) for text in cleaned_texts]
