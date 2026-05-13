"""
test_pipeline.py

Integration tests for RAGPipeline — ingestion, Strategy A, Strategy B.
"""

from unittest.mock import MagicMock

import pytest

from rag.corpus import DOCUMENTS
from rag.pipeline import RAGPipeline
from rag.query_expander import QueryExpander, _MockResponse
from rag.vector_store import SearchResult


@pytest.fixture(scope="module")
def pipeline():
    p = RAGPipeline()
    p.ingest()
    return p


class TestIngestion:

    def test_all_documents_are_stored(self, pipeline):
        assert len(pipeline._store) == len(DOCUMENTS)

    def test_search_before_ingest_raises(self):
        p = RAGPipeline()
        with pytest.raises(RuntimeError, match="ingest"):
            p.search_strategy_a("test")

    def test_custom_documents_can_be_ingested(self):
        p = RAGPipeline()
        custom = [{"id": "x1", "text": "custom document text", "title": "Custom"}]
        p.ingest(raw_docs=custom)
        assert len(p._store) == 1


class TestStrategyA:

    def test_returns_list_of_search_results(self, pipeline):
        results = pipeline.search_strategy_a("peak load", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_returns_requested_count(self, pipeline):
        for k in (1, 2, 3):
            results = pipeline.search_strategy_a("caching", top_k=k)
            assert len(results) == k

    def test_scores_are_in_valid_cosine_range(self, pipeline):
        results = pipeline.search_strategy_a("database sharding", top_k=3)
        for r in results:
            assert -1.01 <= r.score <= 1.01

    def test_results_are_descending_by_score(self, pipeline):
        results = pipeline.search_strategy_a("vector embeddings FAISS", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_doc_ids(self, pipeline):
        results = pipeline.search_strategy_a("query expansion generative model", top_k=3)
        ids = [r.document.doc_id for r in results]
        assert len(ids) == len(set(ids))


class TestStrategyB:

    def test_returns_list_of_search_results(self, pipeline):
        results = pipeline.search_strategy_b("peak load", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_expander_is_called_for_strategy_b(self):
        mock_expander = MagicMock(spec=QueryExpander)
        mock_expander.expand.return_value = "expanded query about load and scaling"

        p = RAGPipeline(expander=mock_expander)
        p.ingest()

        # Strategy A must NOT call expander
        p.search_strategy_a("some query", top_k=1)
        mock_expander.expand.assert_not_called()

        # Strategy B MUST call expander
        p.search_strategy_b("some query", top_k=1)
        mock_expander.expand.assert_called_once_with("some query")

    def test_scores_are_in_valid_cosine_range(self, pipeline):
        results = pipeline.search_strategy_b("How does the system handle peak load?", top_k=3)
        for r in results:
            assert -1.01 <= r.score <= 1.01


class TestBenchmarkQueries:

    @pytest.mark.parametrize("query", [
        "How does the system handle peak load?",
        "Explain the caching mechanism",
        "How are embeddings stored and searched?",
    ])
    def test_both_strategies_return_three_results(self, pipeline, query):
        a = pipeline.search_strategy_a(query, top_k=3)
        b = pipeline.search_strategy_b(query, top_k=3)
        assert len(a) == 3
        assert len(b) == 3
