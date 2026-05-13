# Retrieval Benchmark: Strategy A vs Strategy B

**Strategy A** — raw vector search (query embedded as-is).  
**Strategy B** — AI-enhanced retrieval (query expanded by generative model before embedding).

---

## Query 1: `How does the system handle peak load?`

**Expanded query (Strategy B):**

> The system manages peak load by combining horizontal auto-scaling, circuit breakers, and backpressure mechanisms. When traffic spikes, the load balancer detects rising CPU utilisation and queue depth, triggering the orchestrator to spin up additional compute replicas within seconds. Circuit breakers prevent cascade failures by returning cached fallback responses when downstream error rates are high. Rate limiting via token-bucket quotas at the API gateway further shields backend services from being overwhelmed.

| Rank | Strategy A — Doc ID | Title | Score | Strategy B — Doc ID | Title | Score |
|------|---------------------|-------|-------|---------------------|-------|-------|
| 1 | doc_001 | Horizontal Scaling Under Pea | 0.5706 | doc_001 | Horizontal Scaling Under Pea | 0.8125 |
| 2 | doc_002 | Circuit Breaker and Backpres | 0.4131 | doc_002 | Circuit Breaker and Backpres | 0.6495 |
| 3 | doc_009 | Asynchronous Job Queue Archi | 0.3357 | doc_010 | Rate Limiting and Quota Enfo | 0.6012 |

**Overlap:** 2/3 documents shared  
**Top-1 changed:** No

### Strategy A — snippets

**1. [doc_001] Horizontal Scaling Under Peak Load** (score=0.5706)

> When traffic spikes, the system relies on horizontal auto-scaling to distribute incoming requests across newly provisioned compute nodes. A load balancer continuously monitors CPU utilisation and requ…

**2. [doc_002] Circuit Breaker and Backpressure** (score=0.4131)

> To prevent cascade failures, every inter-service call is wrapped in a circuit breaker. When downstream error rates exceed 50% over a rolling 10-second window the circuit opens and callers receive a pr…

**3. [doc_009] Asynchronous Job Queue Architecture** (score=0.3357)

> Long-running tasks such as report generation, email dispatch, and ML inference are offloaded from the synchronous request path to a durable job queue backed by Apache Kafka. Workers pull tasks in conf…

### Strategy B — snippets

**1. [doc_001] Horizontal Scaling Under Peak Load** (score=0.8125)

> When traffic spikes, the system relies on horizontal auto-scaling to distribute incoming requests across newly provisioned compute nodes. A load balancer continuously monitors CPU utilisation and requ…

**2. [doc_002] Circuit Breaker and Backpressure** (score=0.6495)

> To prevent cascade failures, every inter-service call is wrapped in a circuit breaker. When downstream error rates exceed 50% over a rolling 10-second window the circuit opens and callers receive a pr…

**3. [doc_010] Rate Limiting and Quota Enforcement** (score=0.6012)

> Each API consumer is assigned a token-bucket quota enforced at the edge gateway. Buckets refill at a steady per-second rate and allow short bursts up to twice the baseline rate, accommodating legitima…

---

## Query 2: `Explain the caching mechanism`

**Expanded query (Strategy B):**

> The caching architecture uses multiple layers to reduce database load and lower latency. An in-process LRU cache handles the hottest data, a distributed Redis cluster serves shared application state, and a CDN edge cache absorbs static content at the network perimeter. Cache invalidation combines TTL-based expiry for stable reference data with event-driven pub/sub invalidation for frequently changing user records.

| Rank | Strategy A — Doc ID | Title | Score | Strategy B — Doc ID | Title | Score |
|------|---------------------|-------|-------|---------------------|-------|-------|
| 1 | doc_003 | Caching Strategy and Cache I | 0.479 | doc_003 | Caching Strategy and Cache I | 0.8901 |
| 2 | doc_002 | Circuit Breaker and Backpres | 0.4173 | doc_007 | Database Sharding and Read R | 0.509 |
| 3 | doc_001 | Horizontal Scaling Under Pea | 0.3699 | doc_002 | Circuit Breaker and Backpres | 0.4769 |

**Overlap:** 2/3 documents shared  
**Top-1 changed:** No

### Strategy A — snippets

**1. [doc_003] Caching Strategy and Cache Invalidation** (score=0.479)

> A three-layer caching hierarchy — in-process LRU, distributed Redis cluster, and a CDN edge cache — absorbs the majority of read traffic before it reaches the primary database. TTL-based invalidation …

**2. [doc_002] Circuit Breaker and Backpressure** (score=0.4173)

> To prevent cascade failures, every inter-service call is wrapped in a circuit breaker. When downstream error rates exceed 50% over a rolling 10-second window the circuit opens and callers receive a pr…

**3. [doc_001] Horizontal Scaling Under Peak Load** (score=0.3699)

> When traffic spikes, the system relies on horizontal auto-scaling to distribute incoming requests across newly provisioned compute nodes. A load balancer continuously monitors CPU utilisation and requ…

### Strategy B — snippets

**1. [doc_003] Caching Strategy and Cache Invalidation** (score=0.8901)

> A three-layer caching hierarchy — in-process LRU, distributed Redis cluster, and a CDN edge cache — absorbs the majority of read traffic before it reaches the primary database. TTL-based invalidation …

**2. [doc_007] Database Sharding and Read Replicas** (score=0.509)

> The primary datastore is sharded horizontally by tenant ID using consistent hashing, which evenly distributes write load and keeps cross-shard joins rare. Each shard maintains two synchronous read rep…

**3. [doc_002] Circuit Breaker and Backpressure** (score=0.4769)

> To prevent cascade failures, every inter-service call is wrapped in a circuit breaker. When downstream error rates exceed 50% over a rolling 10-second window the circuit opens and callers receive a pr…

---

## Query 3: `How are embeddings stored and searched?`

**Expanded query (Strategy B):**

> Text documents are chunked, encoded into dense embedding vectors using a sentence-transformer model, and L2-normalised before being stored in a FAISS index. The IndexFlatIP structure performs exact inner-product search, which equals cosine similarity on unit-normalised vectors. At query time the query text is embedded with the same model and the top-k nearest document chunks are retrieved by similarity score.

| Rank | Strategy A — Doc ID | Title | Score | Strategy B — Doc ID | Title | Score |
|------|---------------------|-------|-------|---------------------|-------|-------|
| 1 | doc_005 | FAISS Index Construction and | 0.6046 | doc_004 | Vector Embedding Generation  | 0.8467 |
| 2 | doc_004 | Vector Embedding Generation  | 0.5096 | doc_005 | FAISS Index Construction and | 0.6756 |
| 3 | doc_006 | Query Expansion via Generati | 0.4665 | doc_006 | Query Expansion via Generati | 0.5565 |

**Overlap:** 3/3 documents shared  
**Top-1 changed:** Yes

### Strategy A — snippets

**1. [doc_005] FAISS Index Construction and Search** (score=0.6046)

> Embeddings are stored in a FAISS index using the IndexFlatIP structure for exact inner-product search on unit-normalised vectors, which is equivalent to cosine similarity. For datasets larger than 10 …

**2. [doc_004] Vector Embedding Generation Pipeline** (score=0.5096)

> Raw text documents are first cleaned and chunked into 512-token segments with a 64-token overlap to preserve sentence context across boundaries. Each chunk is then passed through a sentence-transforme…

**3. [doc_006] Query Expansion via Generative Rewriting** (score=0.4665)

> Short or ambiguous user queries often produce poor retrieval recall because their embeddings lie far from the relevant document clusters. A lightweight generative model rewrites each incoming query in…

### Strategy B — snippets

**1. [doc_004] Vector Embedding Generation Pipeline** (score=0.8467)

> Raw text documents are first cleaned and chunked into 512-token segments with a 64-token overlap to preserve sentence context across boundaries. Each chunk is then passed through a sentence-transforme…

**2. [doc_005] FAISS Index Construction and Search** (score=0.6756)

> Embeddings are stored in a FAISS index using the IndexFlatIP structure for exact inner-product search on unit-normalised vectors, which is equivalent to cosine similarity. For datasets larger than 10 …

**3. [doc_006] Query Expansion via Generative Rewriting** (score=0.5565)

> Short or ambiguous user queries often produce poor retrieval recall because their embeddings lie far from the relevant document clusters. A lightweight generative model rewrites each incoming query in…

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
