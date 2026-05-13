DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Horizontal Scaling Under Peak Load",
        "text": (
            "When traffic spikes, the system relies on horizontal auto-scaling to distribute "
            "incoming requests across newly provisioned compute nodes. A load balancer continuously "
            "monitors CPU utilisation and request queue depth; once either metric crosses a "
            "configurable threshold the orchestrator spins up additional replicas within seconds. "
            "This elastic behaviour ensures that no single node becomes a bottleneck during "
            "peak-load events such as product launches or end-of-month batch runs."
        ),
    },
    {
        "id": "doc_002",
        "title": "Circuit Breaker and Backpressure",
        "text": (
            "To prevent cascade failures, every inter-service call is wrapped in a circuit breaker. "
            "When downstream error rates exceed 50% over a rolling 10-second window the circuit "
            "opens and callers receive a pre-cached fallback response instead of waiting for a "
            "timeout. Simultaneously, the message queue applies backpressure by signalling producers "
            "to slow emission rates, which stops memory exhaustion in the consumer tier during "
            "sustained high-throughput periods."
        ),
    },
    {
        "id": "doc_003",
        "title": "Caching Strategy and Cache Invalidation",
        "text": (
            "A three-layer caching hierarchy — in-process LRU, distributed Redis cluster, and a "
            "CDN edge cache — absorbs the majority of read traffic before it reaches the primary "
            "database. TTL-based invalidation is used for relatively static reference data, while "
            "event-driven invalidation via a pub/sub channel is employed for high-churn user "
            "records. This hybrid approach reduces average database query load by roughly 80% "
            "during normal operating hours and by even more during anticipated traffic surges."
        ),
    },
    {
        "id": "doc_004",
        "title": "Vector Embedding Generation Pipeline",
        "text": (
            "Raw text documents are first cleaned and chunked into 512-token segments with a "
            "64-token overlap to preserve sentence context across boundaries. Each chunk is then "
            "passed through a sentence-transformer model to produce a 384-dimensional dense "
            "embedding vector. These vectors are L2-normalised before storage so that dot-product "
            "similarity is equivalent to cosine similarity, simplifying both indexing and "
            "query-time arithmetic."
        ),
    },
    {
        "id": "doc_005",
        "title": "FAISS Index Construction and Search",
        "text": (
            "Embeddings are stored in a FAISS index using the IndexFlatIP structure for exact "
            "inner-product search on unit-normalised vectors, which is equivalent to cosine "
            "similarity. For datasets larger than 10 million vectors, an IVF index partitioned "
            "into Voronoi cells is preferred to reduce search latency. Periodic index rebuilds "
            "are triggered whenever the ratio of new vectors to total indexed vectors exceeds "
            "15%, keeping recall metrics stable over time."
        ),
    },
    {
        "id": "doc_006",
        "title": "Query Expansion via Generative Rewriting",
        "text": (
            "Short or ambiguous user queries often produce poor retrieval recall because their "
            "embeddings lie far from the relevant document clusters. A lightweight generative "
            "model rewrites each incoming query into an expanded, context-rich paragraph that "
            "makes the semantic intent explicit. The expanded text is then embedded and used as "
            "the search vector, consistently improving recall in offline evaluations."
        ),
    },
    {
        "id": "doc_007",
        "title": "Database Sharding and Read Replicas",
        "text": (
            "The primary datastore is sharded horizontally by tenant ID using consistent hashing, "
            "which evenly distributes write load and keeps cross-shard joins rare. Each shard "
            "maintains two synchronous read replicas in separate availability zones; all "
            "analytical and reporting queries are routed to these replicas to avoid contending "
            "with transactional writes. A rolling-replica promotion strategy allows zero-downtime "
            "schema migrations without locking production traffic."
        ),
    },
    {
        "id": "doc_008",
        "title": "Observability Stack and Alerting",
        "text": (
            "A centralised observability platform aggregates traces, metrics, and structured logs "
            "from every service via OpenTelemetry collectors. Latency histograms at p50, p95, and "
            "p99 percentiles are exported to a time-series database every 15 seconds. Anomaly "
            "detection models scan these histograms in real time and fire alerts when p99 latency "
            "exceeds the 7-day rolling baseline by more than two standard deviations."
        ),
    },
    {
        "id": "doc_009",
        "title": "Asynchronous Job Queue Architecture",
        "text": (
            "Long-running tasks such as report generation, email dispatch, and ML inference are "
            "offloaded from the synchronous request path to a durable job queue backed by Apache "
            "Kafka. Workers pull tasks in configurable batch sizes and process them idempotently "
            "using a deduplication key stored in Redis. Failed jobs are automatically retried "
            "with exponential back-off up to a configurable maximum attempt count."
        ),
    },
    {
        "id": "doc_010",
        "title": "Rate Limiting and Quota Enforcement",
        "text": (
            "Each API consumer is assigned a token-bucket quota enforced at the edge gateway. "
            "Buckets refill at a steady per-second rate and allow short bursts up to twice the "
            "baseline rate, accommodating legitimate traffic spikes without penalising well-behaved "
            "clients. When a bucket empties the gateway responds with HTTP 429 and a Retry-After "
            "header so clients can back off gracefully."
        ),
    },
]
