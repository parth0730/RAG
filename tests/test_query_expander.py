"""
test_query_expander.py

Tests for QueryExpander and the mock of
vertexai.generative_models.GenerativeModel.
"""

from unittest.mock import MagicMock, patch

import pytest

from rag.query_expander import (
    MockGenerativeModel,
    QueryExpander,
    _MockResponse,
    _EXPANSION_PROMPT,
)


class TestMockGenerativeModel:

    def test_known_peak_load_query_returns_relevant_expansion(self):
        model = MockGenerativeModel()
        prompt = _EXPANSION_PROMPT.format(query="How does the system handle peak load")
        response = model.generate_content(prompt)
        text = response.text.lower()
        assert any(kw in text for kw in ("scal", "load", "circuit", "replicas", "traffic"))

    def test_known_caching_query_returns_relevant_expansion(self):
        model = MockGenerativeModel()
        prompt = _EXPANSION_PROMPT.format(query="Explain the caching mechanism")
        response = model.generate_content(prompt)
        text = response.text.lower()
        assert any(kw in text for kw in ("cache", "redis", "ttl", "invalidat", "lru"))

    def test_unknown_query_returns_fallback(self):
        model = MockGenerativeModel()
        prompt = _EXPANSION_PROMPT.format(query="What is the boiling point of water")
        response = model.generate_content(prompt)
        assert len(response.text) > 20

    def test_response_has_text_attribute(self):
        model = MockGenerativeModel()
        response = model.generate_content(_EXPANSION_PROMPT.format(query="test"))
        assert hasattr(response, "text")
        assert isinstance(response.text, str)

    def test_model_name_is_stored(self):
        model = MockGenerativeModel(model_name="gemini-1.5-pro")
        assert model.model_name == "gemini-1.5-pro"

    def test_mock_response_mirrors_real_sdk_interface(self):
        """
        The real GenerativeModel response also exposes .text.
        This documents the contract for when we swap in the real model.
        """
        response = _MockResponse("some expanded text")
        assert response.text == "some expanded text"


class TestQueryExpander:

    def test_expand_returns_longer_string(self):
        expander = QueryExpander()
        original = "How does the system handle peak load?"
        expanded = expander.expand(original)
        assert len(expanded) > len(original)

    def test_expand_calls_model_generate_content(self):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = _MockResponse("expanded result")
        expander = QueryExpander(model=mock_model)
        result = expander.expand("some query")
        mock_model.generate_content.assert_called_once()
        assert result == "expanded result"

    def test_original_query_appears_in_prompt(self):
        captured = []

        class CapturingModel:
            def generate_content(self, prompt):
                captured.append(prompt)
                return _MockResponse("ok")

        QueryExpander(model=CapturingModel()).expand("my specific test query")
        assert "my specific test query" in captured[0]

    def test_expand_strips_whitespace(self):
        expander = QueryExpander()
        result = expander.expand("  caching mechanism  ")
        assert result == result.strip()


class TestPatchingVertexAIGenerativeModel:
    """
    Demonstrates how to patch vertexai.generative_models.GenerativeModel
    in CI without a real GCP project.
    """

    def test_patch_generative_model_generate_content(self):
        fake_response = _MockResponse("mocked expansion from vertex ai")

        with patch(
            "rag.query_expander.MockGenerativeModel.generate_content",
            return_value=fake_response,
        ) as mocked:
            expander = QueryExpander()
            result = expander.expand("peak load query")
            mocked.assert_called_once()
            assert result == "mocked expansion from vertex ai"

    def test_patching_real_vertexai_path(self):
        """
        Shows the exact patch path for production code that imports
        from vertexai.generative_models directly.
        """
        fake_model = MagicMock()
        fake_model.generate_content.return_value = _MockResponse("vertex expanded text")

        with patch(
            "vertexai.generative_models.GenerativeModel",
            return_value=fake_model,
        ) as MockClass:
            import vertexai.generative_models as vg
            model = vg.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content("some prompt")

            MockClass.assert_called_once_with("gemini-1.5-flash")
            assert response.text == "vertex expanded text"
