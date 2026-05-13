"""
embedder.py

Local sentence-transformers model that simulates Vertex AI's
textembedding-gecko behaviour, plus a mock of
vertexai.language_models.TextEmbeddingModel for unit testing.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Local embedder — simulates textembedding-gecko
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    Wraps sentence-transformers all-MiniLM-L6-v2 to simulate the behaviour
    of Vertex AI's textembedding-gecko model.

    The model produces 384-dimensional vectors.  Vectors are L2-normalised
    so that inner-product search equals cosine similarity, matching gecko's
    default behaviour.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = SentenceTransformer(self.MODEL_NAME)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of document texts.

        Returns
        -------
        np.ndarray  shape (N, 384), float32, L2-normalised
        """
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray  shape (384,), float32, L2-normalised
        """
        vector = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Mock for vertexai.language_models.TextEmbeddingModel
# ---------------------------------------------------------------------------

def make_mock_text_embedding_model(embedder: LocalEmbedder | None = None) -> MagicMock:
    """
    Return a MagicMock that mirrors the interface of
    vertexai.language_models.TextEmbeddingModel.

    The mock delegates actual embedding work to a LocalEmbedder so tests
    exercise real vector arithmetic without needing a GCP project.

    Usage in tests
    --------------
    with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained",
               return_value=make_mock_text_embedding_model()):
        ...
    """
    _backend = embedder or LocalEmbedder()

    mock_model = MagicMock(name="TextEmbeddingModel")

    def _get_embeddings(texts: List[str]):
        vectors = _backend.embed_documents(texts)
        results = []
        for vec in vectors:
            embedding_obj = MagicMock()
            embedding_obj.values = vec.tolist()
            results.append(embedding_obj)
        return results

    mock_model.get_embeddings.side_effect = _get_embeddings
    return mock_model
