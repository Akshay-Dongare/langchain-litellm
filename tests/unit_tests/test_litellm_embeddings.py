"""Unit tests for LiteLLMEmbeddings."""

from typing import Type
from unittest.mock import MagicMock, patch

from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_litellm.embeddings import LiteLLMEmbeddings


class TestLiteLLMEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[LiteLLMEmbeddings]:
        return LiteLLMEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model": "openai/text-embedding-3-small",
            "api_key": "fake-key",
        }


def _mock_embedding_response(texts):
    """Create a mock litellm embedding response."""
    mock_response = MagicMock()
    mock_response.data = [
        {"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(texts))
    ]
    return mock_response


class TestLiteLLMEmbeddingsParams:
    def test_default_params(self):
        """Test default parameter values."""
        embeddings = LiteLLMEmbeddings(api_key="fake")
        assert embeddings.model == "openai/text-embedding-3-small"
        assert embeddings.max_retries == 1
        assert embeddings.api_base is None

    def test_custom_params(self):
        """Test custom parameter passthrough."""
        embeddings = LiteLLMEmbeddings(
            model="cohere/embed-english-v3.0",
            api_key="fake-key",
            api_base="https://custom.endpoint.com",
            dimensions=256,
            request_timeout=30.0,
        )
        params = embeddings._get_litellm_params()
        assert params["model"] == "cohere/embed-english-v3.0"
        assert params["api_key"] == "fake-key"
        assert params["api_base"] == "https://custom.endpoint.com"
        assert params["dimensions"] == 256
        assert params["timeout"] == 30.0

    def test_none_params_excluded(self):
        """Test that None-valued params are excluded from the litellm call."""
        embeddings = LiteLLMEmbeddings(model="openai/text-embedding-3-small", api_key="fake")
        params = embeddings._get_litellm_params()
        assert "api_base" not in params
        assert "api_version" not in params
        assert "dimensions" not in params

    def test_model_kwargs_merged(self):
        """Test that model_kwargs are merged into params."""
        embeddings = LiteLLMEmbeddings(
            api_key="fake",
            model_kwargs={"user": "test-user"},
        )
        params = embeddings._get_litellm_params()
        assert params["user"] == "test-user"

    @patch("litellm.embedding")
    def test_embed_documents(self, mock_embedding):
        """Test embed_documents calls litellm.embedding correctly."""
        mock_embedding.return_value = _mock_embedding_response(["hello", "world"])

        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
        )
        result = embeddings.embed_documents(["hello", "world"])

        mock_embedding.assert_called_once()
        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["input"] == ["hello", "world"]
        assert call_kwargs["model"] == "openai/text-embedding-3-small"
        assert call_kwargs["api_key"] == "fake-key"
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("litellm.embedding")
    def test_embed_query(self, mock_embedding):
        """Test embed_query calls litellm.embedding with a single-item list."""
        mock_embedding.return_value = _mock_embedding_response(["hello"])

        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
        )
        result = embeddings.embed_query("hello")

        mock_embedding.assert_called_once()
        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["input"] == ["hello"]
        assert result == [0.1, 0.2, 0.3]
