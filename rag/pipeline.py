"""
pipeline.py

RAGPipeline is the single class that manages:
  - ingestion of the text dataset (embed + store)
  - Strategy A: raw vector search
  - Strategy B: query-expanded vector search
"""

from __future__ import annotations

from typing import List, Optional

from rag.corpus import DOCUMENTS
from rag.embedder import LocalEmbedder
from rag.query_expander import QueryExpander
from rag.vector_store import Document, FAISSVectorStore, SearchResult


class RAGPipeline:
    """
    Manages ingestion of a text dataset and exposes two retrieval strategies.

    Parameters
    ----------
    embedder : LocalEmbedder, optional
        Embedding backend.  Defaults to LocalEmbedder (sentence-transformers).
    expander : QueryExpander, optional
        Query expansion backend.  Defaults to QueryExpander (mock GenerativeModel).
    """

    def __init__(
        self,
        embedder: Optional[LocalEmbedder] = None,
        expander: Optional[QueryExpander] = None,
    ) -> None:
        self._embedder = embedder or LocalEmbedder()
        self._expander = expander or QueryExpander()
        self._store = FAISSVectorStore(embedding_dim=384)
        self._ingested = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, raw_docs: Optional[List[dict]] = None) -> None:
        """
        Embed each document and add it to the FAISS vector store.

        Parameters
        ----------
        raw_docs : list of dicts, optional
            Each dict must contain 'id' and 'text' keys, and optionally 'title'.
            Defaults to the built-in 10-paragraph corpus.
        """
        source = raw_docs or DOCUMENTS

        documents = [
            Document(
                doc_id=d["id"],
                text=d["text"],
                metadata={"title": d.get("title", "")},
            )
            for d in source
        ]

        texts = [doc.text for doc in documents]
        embeddings = self._embedder.embed_documents(texts)
        self._store.add_documents(documents, embeddings)
        self._ingested = True

    # ------------------------------------------------------------------
    # Strategy A — raw vector search
    # ------------------------------------------------------------------

    def search_strategy_a(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Embed the query as-is and retrieve the top-k most similar documents.
        """
        self._check_ingested()
        query_vector = self._embedder.embed_query(query)
        return self._store.search(query_vector, top_k=top_k)

    # ------------------------------------------------------------------
    # Strategy B — AI-enhanced retrieval (query expansion)
    # ------------------------------------------------------------------

    def search_strategy_b(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Expand the query with the generative model, then embed and retrieve.
        """
        self._check_ingested()
        expanded_query = self._expander.expand(query)
        query_vector = self._embedder.embed_query(expanded_query)
        return self._store.search(query_vector, top_k=top_k)

    # ------------------------------------------------------------------

    def _check_ingested(self) -> None:
        if not self._ingested:
            raise RuntimeError("Call pipeline.ingest() before searching.")
