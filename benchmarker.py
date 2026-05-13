"""
benchmarker.py

Runs Strategy A vs Strategy B across the three required benchmark queries
and writes:
  - reports/benchmark_results.json
  - reports/retrieval_benchmark.md

Run directly:
    python benchmarker.py
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict, dataclass
from typing import List

from rag.pipeline import RAGPipeline
from rag.vector_store import SearchResult

BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "Explain the caching mechanism",
    "How are embeddings stored and searched?",
]

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    rank: int
    doc_id: str
    title: str
    score: float
    snippet: str


@dataclass
class QueryComparison:
    query: str
    expanded_query: str
    strategy_a: List[RetrievedChunk]
    strategy_b: List[RetrievedChunk]
    overlap_count: int
    top1_changed: bool


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------

class Benchmarker:

    def __init__(self, pipeline: RAGPipeline, top_k: int = 3) -> None:
        self._pipeline = pipeline
        self._top_k = top_k

    def run(self, queries: List[str] | None = None) -> List[QueryComparison]:
        queries = queries or BENCHMARK_QUERIES
        os.makedirs(REPORTS_DIR, exist_ok=True)

        comparisons = [self._compare(q) for q in queries]
        self._write_json(comparisons)
        self._write_markdown(comparisons)
        return comparisons

    # ------------------------------------------------------------------

    def _compare(self, query: str) -> QueryComparison:
        results_a = self._pipeline.search_strategy_a(query, top_k=self._top_k)
        results_b = self._pipeline.search_strategy_b(query, top_k=self._top_k)
        expanded = self._pipeline._expander.expand(query)

        chunks_a = [self._to_chunk(r) for r in results_a]
        chunks_b = [self._to_chunk(r) for r in results_b]

        ids_a = {c.doc_id for c in chunks_a}
        ids_b = {c.doc_id for c in chunks_b}

        return QueryComparison(
            query=query,
            expanded_query=expanded,
            strategy_a=chunks_a,
            strategy_b=chunks_b,
            overlap_count=len(ids_a & ids_b),
            top1_changed=(
                bool(chunks_a) and bool(chunks_b)
                and chunks_a[0].doc_id != chunks_b[0].doc_id
            ),
        )

    @staticmethod
    def _to_chunk(r: SearchResult) -> RetrievedChunk:
        return RetrievedChunk(
            rank=r.rank,
            doc_id=r.document.doc_id,
            title=r.document.metadata.get("title", ""),
            score=round(r.score, 4),
            snippet=r.document.text[:200].replace("\n", " "),
        )

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    def _write_json(self, comparisons: List[QueryComparison]) -> None:
        path = os.path.join(REPORTS_DIR, "benchmark_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in comparisons], f, indent=2)
        print(f"JSON report  -> {path}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def _write_markdown(self, comparisons: List[QueryComparison]) -> None:
        path = os.path.join(REPORTS_DIR, "retrieval_benchmark.md")
        lines = []

        lines.append("# Retrieval Benchmark: Strategy A vs Strategy B\n\n")
        lines.append(
            "**Strategy A** — raw vector search (query embedded as-is).  \n"
            "**Strategy B** — AI-enhanced retrieval (query expanded by generative model before embedding).\n\n"
        )

        for i, c in enumerate(comparisons, start=1):
            lines.append(f"---\n\n## Query {i}: `{c.query}`\n\n")
            lines.append(f"**Expanded query (Strategy B):**\n\n> {c.expanded_query}\n\n")

            lines.append(
                "| Rank | Strategy A — Doc ID | Title | Score |"
                " Strategy B — Doc ID | Title | Score |\n"
                "|------|---------------------|-------|-------|"
                "---------------------|-------|-------|\n"
            )
            for rank in range(self._top_k):
                a = c.strategy_a[rank] if rank < len(c.strategy_a) else None
                b = c.strategy_b[rank] if rank < len(c.strategy_b) else None
                lines.append(
                    f"| {rank+1} "
                    f"| {a.doc_id if a else '—'} | {(a.title[:28] if a else '—')} | {a.score if a else '—'} "
                    f"| {b.doc_id if b else '—'} | {(b.title[:28] if b else '—')} | {b.score if b else '—'} |\n"
                )

            lines.append(
                f"\n**Overlap:** {c.overlap_count}/{self._top_k} documents shared  \n"
                f"**Top-1 changed:** {'Yes' if c.top1_changed else 'No'}\n\n"
            )

            lines.append("### Strategy A — snippets\n\n")
            for chunk in c.strategy_a:
                lines.append(f"**{chunk.rank}. [{chunk.doc_id}] {chunk.title}** (score={chunk.score})\n\n")
                lines.append(f"> {chunk.snippet}…\n\n")

            lines.append("### Strategy B — snippets\n\n")
            for chunk in c.strategy_b:
                lines.append(f"**{chunk.rank}. [{chunk.doc_id}] {chunk.title}** (score={chunk.score})\n\n")
                lines.append(f"> {chunk.snippet}…\n\n")

        lines.append(textwrap.dedent("""\
            ---

            ## Similarity Metric: Cosine vs Euclidean

            ### Why cosine similarity?

            All embedding vectors are L2-normalised before storage. On unit vectors,
            inner product equals cosine similarity, so FAISS IndexFlatIP gives exact
            cosine search at no extra cost. Cosine is the right choice here because:

            - It is length-invariant: a long document and a short one on the same topic
              score equally against a relevant query.
            - It is standard practice for sentence-transformer embeddings. The model was
              trained with cosine objectives; using a different metric at inference time
              breaks that assumption.
            - High-dimensional embedding spaces make Euclidean distances cluster tightly,
              reducing discriminability between relevant and irrelevant documents.

            ### When Euclidean would be appropriate

            Euclidean distance makes sense when vector magnitude carries meaning — for
            example count vectors, spectrograms, or pixel grids. For dense semantic
            embeddings it is generally inferior to cosine.

            ---

            ## Production Migration to Vertex AI Vector Search (Matching Engine)

            | Step | Action |
            |------|--------|
            | 1 | Export embeddings as JSONL to GCS: `{"id": "...", "embedding": [...]}` |
            | 2 | Create index via `aiplatform.MatchingEngineIndex.create_tree_ah_index()` with `dimensions=384`, `distance_measure_type=DOT_PRODUCT_DISTANCE` |
            | 3 | Deploy to an `IndexEndpoint` |
            | 4 | Replace `FAISSVectorStore.search()` with `endpoint.match(deployed_index_id=..., queries=[q_vec], num_neighbors=top_k)` |
            | 5 | Replace `LocalEmbedder` with real `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` |
            | 6 | Replace `MockGenerativeModel` with `vertexai.generative_models.GenerativeModel("gemini-1.5-flash")` |
        """))

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Markdown report -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.ingest()

    benchmarker = Benchmarker(pipeline)
    comparisons = benchmarker.run()

    print("\n=== SUMMARY ===\n")
    for c in comparisons:
        print(f"Query : {c.query}")
        print(f"  A top-3 : {[x.doc_id for x in c.strategy_a]}")
        print(f"  B top-3 : {[x.doc_id for x in c.strategy_b]}")
        print(f"  Overlap : {c.overlap_count}/3 | Top-1 changed: {c.top1_changed}\n")
