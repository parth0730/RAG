"""
vector_store.py

FAISS-backed vector store for storing and searching document embeddings.
Uses IndexFlatIP (exact inner-product / cosine on unit-normalised vectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class FAISSVectorStore:
    """
    Stores L2-normalised embeddings in a FAISS IndexFlatIP index.

    Because all vectors are unit-normalised, inner product equals
    cosine similarity.  Higher score = more similar.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self._index: Optional[faiss.IndexFlatIP] = None
        self._documents: List[Document] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their pre-computed embeddings to the store.

        Parameters
        ----------
        documents  : list of Document objects
        embeddings : np.ndarray, shape (N, D), float32, L2-normalised
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Got {len(documents)} documents but {embeddings.shape[0]} embeddings."
            )

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)          # defensive re-normalise

        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])

        self._index.add(vectors)
        self._documents.extend(documents)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Return the top-k most similar documents for a query vector.

        Parameters
        ----------
        query_vector : np.ndarray, shape (D,), float32
        top_k        : int

        Returns
        -------
        list of SearchResult ordered by descending cosine similarity
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:           # FAISS sentinel for "not found"
                continue
            results.append(
                SearchResult(
                    document=self._documents[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._documents)
