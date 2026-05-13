# Semantic RAG & Vector Search — Senior Gen AI Assessment

## Project Structure

```
rag_engine/
├── rag/
│   ├── __init__.py
│   ├── corpus.py          # 10 technical paragraphs (the text dataset)
│   ├── embedder.py        # LocalEmbedder (sentence-transformers) + mock TextEmbeddingModel
│   ├── vector_store.py    # FAISSVectorStore (IndexFlatIP, cosine similarity)
│   ├── query_expander.py  # MockGenerativeModel + QueryExpander
│   └── pipeline.py        # RAGPipeline class — ingestion, Strategy A, Strategy B
├── tests/
│   ├── _vertexai_stub/    # Lightweight stubs so vertexai SDK can be patched without install
│   │   ├── __init__.py
│   │   ├── language_models.py
│   │   └── generative_models.py
│   ├── conftest.py        # Path setup + vertexai stub registration
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_query_expander.py
│   └── test_pipeline.py
├── benchmarker.py         # Runs A vs B comparison, writes reports/
├── reports/               # Generated output (git-committed sample included)
│   ├── benchmark_results.json
│   └── retrieval_benchmark.md
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the benchmark

```bash
cd rag_engine
python benchmarker.py
```

Output files are written to `reports/`.

## Run tests

```bash
cd rag_engine
pytest tests/ -v
```

## Design decisions

### Embedding model
`sentence-transformers/all-MiniLM-L6-v2` is used locally to simulate
`textembedding-gecko`. It produces 384-dimensional L2-normalised vectors with
the same semantic quality characteristics. Swap `LocalEmbedder` with the real
`TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` for production.

### Vector database
FAISS `IndexFlatIP` performs exact inner-product search. On unit-normalised
vectors this is identical to cosine similarity. No approximation errors,
no configuration overhead — correct for a dataset of 10 documents and easily
swapped for `IndexIVFFlat` at scale.

### Mocking
- `make_mock_text_embedding_model()` returns a `MagicMock` that mirrors
  `vertexai.language_models.TextEmbeddingModel` — same `.get_embeddings(texts)`
  method, same `.values` attribute on each result object.
- `MockGenerativeModel` mirrors `vertexai.generative_models.GenerativeModel` —
  same `.generate_content(prompt)` method returning an object with `.text`.
  Both mocks delegate real work to local models so tests exercise actual vector
  arithmetic without any GCP credentials.

### Similarity metric
See `reports/retrieval_benchmark.md` for the full discussion.
