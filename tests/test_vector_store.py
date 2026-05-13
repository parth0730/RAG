"""
test_vector_store.py

Tests for FAISSVectorStore: ingestion, cosine search via inner product,
edge cases.
"""

import numpy as np
import pytest

from rag.vector_store import Document, FAISSVectorStore, SearchResult


def _unit(values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_docs(n):
    return [Document(doc_id=f"d{i}", text=f"document text {i}", metadata={}) for i in range(n)]


class TestFAISSVectorStore:

    def test_add_and_len(self):
        store = FAISSVectorStore()
        docs = _make_docs(4)
        vecs = np.random.randn(4, 384).astype(np.float32)
        store.add_documents(docs, vecs)
        assert len(store) == 4

    def test_search_returns_top_k_results(self):
        store = FAISSVectorStore()
        docs = _make_docs(6)
        vecs = np.random.randn(6, 384).astype(np.float32)
        store.add_documents(docs, vecs)
        results = store.search(_unit([1.0] * 384), top_k=3)
        assert len(results) == 3

    def test_search_results_are_descending_by_score(self):
        store = FAISSVectorStore()
        docs = _make_docs(5)
        vecs = np.random.randn(5, 384).astype(np.float32)
        store.add_documents(docs, vecs)
        results = store.search(_unit([1.0] * 384), top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_is_rank_1(self):
        store = FAISSVectorStore()
        target = _unit(list(range(1, 385)))
        vecs = np.random.randn(5, 384).astype(np.float32)
        vecs[3] = target
        docs = _make_docs(5)
        store.add_documents(docs, vecs)
        results = store.search(target, top_k=1)
        assert results[0].document.doc_id == "d3"

    def test_cosine_score_near_1_for_identical_vector(self):
        store = FAISSVectorStore()
        target = _unit(list(range(1, 385)))
        vecs = np.random.randn(5, 384).astype(np.float32)
        vecs[0] = target
        docs = _make_docs(5)
        store.add_documents(docs, vecs)
        results = store.search(target, top_k=1)
        assert results[0].score > 0.999

    def test_result_ranks_are_sequential(self):
        store = FAISSVectorStore()
        docs = _make_docs(5)
        vecs = np.random.randn(5, 384).astype(np.float32)
        store.add_documents(docs, vecs)
        results = store.search(_unit([1.0] * 384), top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_empty_store_returns_empty_list(self):
        store = FAISSVectorStore()
        results = store.search(_unit([1.0] * 384), top_k=3)
        assert results == []

    def test_mismatched_docs_and_vectors_raises(self):
        store = FAISSVectorStore()
        docs = _make_docs(3)
        vecs = np.random.randn(2, 384).astype(np.float32)
        with pytest.raises(ValueError):
            store.add_documents(docs, vecs)

    def test_search_result_has_document_and_score(self):
        store = FAISSVectorStore()
        docs = _make_docs(3)
        vecs = np.random.randn(3, 384).astype(np.float32)
        store.add_documents(docs, vecs)
        results = store.search(_unit([1.0] * 384), top_k=1)
        assert isinstance(results[0], SearchResult)
        assert isinstance(results[0].document, Document)
        assert isinstance(results[0].score, float)
