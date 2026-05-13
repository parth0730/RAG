"""
query_expander.py

Mocks vertexai.generative_models.GenerativeModel for the query
expansion phase (Strategy B).

In production you would replace MockGenerativeModel with:

    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project="your-project", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Pre-written expansions for the three benchmark queries
# ---------------------------------------------------------------------------

_EXPANSIONS = {
    "how does the system handle peak load": (
        "The system manages peak load by combining horizontal auto-scaling, circuit breakers, "
        "and backpressure mechanisms. When traffic spikes, the load balancer detects rising CPU "
        "utilisation and queue depth, triggering the orchestrator to spin up additional compute "
        "replicas within seconds. Circuit breakers prevent cascade failures by returning cached "
        "fallback responses when downstream error rates are high. Rate limiting via token-bucket "
        "quotas at the API gateway further shields backend services from being overwhelmed."
    ),
    "explain the caching mechanism": (
        "The caching architecture uses multiple layers to reduce database load and lower latency. "
        "An in-process LRU cache handles the hottest data, a distributed Redis cluster serves "
        "shared application state, and a CDN edge cache absorbs static content at the network "
        "perimeter. Cache invalidation combines TTL-based expiry for stable reference data with "
        "event-driven pub/sub invalidation for frequently changing user records."
    ),
    "how are embeddings stored and searched": (
        "Text documents are chunked, encoded into dense embedding vectors using a "
        "sentence-transformer model, and L2-normalised before being stored in a FAISS index. "
        "The IndexFlatIP structure performs exact inner-product search, which equals cosine "
        "similarity on unit-normalised vectors. At query time the query text is embedded with "
        "the same model and the top-k nearest document chunks are retrieved by similarity score."
    ),
}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# ---------------------------------------------------------------------------
# Mock GenerativeModel
# ---------------------------------------------------------------------------

class MockGenerativeModel:
    """
    Mirrors the interface of vertexai.generative_models.GenerativeModel.

    generate_content(prompt) returns an object with a .text attribute,
    exactly as the real SDK does.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self.model_name = model_name

    def generate_content(self, prompt: str) -> "_MockResponse":
        match = re.search(r'User query:\s*"(.+?)"', prompt, re.DOTALL)
        raw = match.group(1).strip() if match else prompt
        key = _normalise(raw)

        for stored_key, expansion in _EXPANSIONS.items():
            if stored_key in key or key in stored_key:
                return _MockResponse(expansion)

        # Generic fallback for any query outside the three benchmark ones
        fallback = (
            f"This question is about: {raw}. "
            "A relevant document would discuss the underlying architecture, "
            "operational strategies, and trade-offs around performance and reliability."
        )
        return _MockResponse(fallback)


class _MockResponse:
    """Mirrors the response object returned by the real GenerativeModel."""

    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# Query expander — wraps the mock (or a real model in production)
# ---------------------------------------------------------------------------

_EXPANSION_PROMPT = (
    'You are a search query optimiser.\n'
    'Rewrite the following short query as a detailed paragraph (3-5 sentences) '
    'that makes the user intent explicit and uses vocabulary a relevant document would contain.\n\n'
    'User query: "{query}"\n\n'
    'Expanded query:'
)


class QueryExpander:
    """
    Expands a short user query using a generative model.

    Parameters
    ----------
    model : optional
        Pass a real GenerativeModel for production use.
        Defaults to MockGenerativeModel.
    """

    def __init__(self, model=None) -> None:
        self._model = model or MockGenerativeModel()

    def expand(self, query: str) -> str:
        prompt = _EXPANSION_PROMPT.format(query=query)
        response = self._model.generate_content(prompt)
        return response.text.strip()
