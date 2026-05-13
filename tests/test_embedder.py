"""
test_embedder.py

Tests for LocalEmbedder (sentence-transformers) and the mock for
vertexai.language_models.TextEmbeddingModel.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.embedder import LocalEmbedder, make_mock_text_embedding_model


SAMPLE_TEXTS = [
    "The system scales horizontally to handle peak traffic.",
    "Caching reduces database load using Redis and CDN.",
    "FAISS stores embeddings for fast nearest-neighbour search.",
]


class TestLocalEmbedder:

    def test_embed_documents_returns_correct_shape(self):
        embedder = LocalEmbedder()
        vectors = embedder.embed_documents(SAMPLE_TEXTS)
        assert vectors.shape == (3, 384)

    def test_embed_documents_are_unit_normalised(self):
        embedder = LocalEmbedder()
        vectors = embedder.embed_documents(SAMPLE_TEXTS)
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_documents_dtype_is_float32(self):
        embedder = LocalEmbedder()
        vectors = embedder.embed_documents(SAMPLE_TEXTS)
        assert vectors.dtype == np.float32

    def test_embed_query_returns_1d_vector(self):
        embedder = LocalEmbedder()
        vector = embedder.embed_query("load balancing under peak traffic")
        assert vector.ndim == 1
        assert vector.shape[0] == 384

    def test_embed_query_is_unit_normalised(self):
        embedder = LocalEmbedder()
        vector = embedder.embed_query("some query text")
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-5

    def test_different_texts_produce_different_vectors(self):
        embedder = LocalEmbedder()
        vectors = embedder.embed_documents(SAMPLE_TEXTS)
        # No two rows should be identical
        for i in range(len(SAMPLE_TEXTS)):
            for j in range(i + 1, len(SAMPLE_TEXTS)):
                assert not np.allclose(vectors[i], vectors[j])

    def test_same_text_produces_same_vector(self):
        embedder = LocalEmbedder()
        v1 = embedder.embed_query("peak load handling")
        v2 = embedder.embed_query("peak load handling")
        np.testing.assert_array_equal(v1, v2)


class TestMockTextEmbeddingModel:
    """
    Verifies the mock of vertexai.language_models.TextEmbeddingModel.
    """

    def test_mock_get_embeddings_returns_correct_count(self):
        mock_model = make_mock_text_embedding_model()
        results = mock_model.get_embeddings(SAMPLE_TEXTS)
        assert len(results) == len(SAMPLE_TEXTS)

    def test_mock_embedding_values_are_floats(self):
        mock_model = make_mock_text_embedding_model()
        results = mock_model.get_embeddings(SAMPLE_TEXTS)
        for result in results:
            assert isinstance(result.values, list)
            assert all(isinstance(v, float) for v in result.values)

    def test_mock_embedding_dimension_is_384(self):
        mock_model = make_mock_text_embedding_model()
        results = mock_model.get_embeddings(SAMPLE_TEXTS)
        for result in results:
            assert len(result.values) == 384

    def test_mock_is_a_magicmock(self):
        mock_model = make_mock_text_embedding_model()
        assert isinstance(mock_model, MagicMock)

    def test_patching_vertexai_text_embedding_model(self):
        """
        Shows how to patch vertexai.language_models.TextEmbeddingModel
        in tests so no GCP credentials are needed.
        """
        mock_model = make_mock_text_embedding_model()

        with patch(
            "vertexai.language_models.TextEmbeddingModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained:
            # Simulate what production code would do
            import vertexai.language_models as vtx
            model = vtx.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            embeddings = model.get_embeddings(SAMPLE_TEXTS)

            mock_from_pretrained.assert_called_once_with("textembedding-gecko@003")
            assert len(embeddings) == 3
