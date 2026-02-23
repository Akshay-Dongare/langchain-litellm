"""Wrapper around LiteLLM's embedding API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LiteLLMEmbeddings(BaseModel, Embeddings):
    """LiteLLM embedding model.

    Uses `litellm.embedding()` to support 100+ providers through a unified
    interface. All provider configuration (api_key, api_base, etc.) can be
    passed explicitly—no environment variables required.

    Example:
        .. code-block:: python

            from langchain_litellm import LiteLLMEmbeddings

            embeddings = LiteLLMEmbeddings(
                model="openai/text-embedding-3-small",
                api_key="sk-...",
            )
            vectors = embeddings.embed_documents(["hello", "world"])
    """

    model: str = "openai/text-embedding-3-small"
    """Model name in litellm format (e.g. 'openai/text-embedding-3-small',
    'cohere/embed-english-v3.0', 'bedrock/amazon.titan-embed-text-v1')."""

    api_key: Optional[str] = None
    """API key for the provider."""

    api_base: Optional[str] = None
    """Base URL for the API endpoint."""

    api_version: Optional[str] = None
    """API version (e.g. for Azure)."""

    custom_llm_provider: Optional[str] = None
    """Override the litellm provider routing."""

    organization: Optional[str] = None
    """Organization ID (e.g. for OpenAI)."""

    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for API requests."""

    max_retries: int = 1
    """Maximum number of retries on failure."""

    extra_headers: Optional[Dict[str, str]] = None
    """Extra headers to include in the request."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional model parameters passed to litellm.embedding()."""

    dimensions: Optional[int] = None
    """Output embedding dimensions (if supported by the model)."""

    encoding_format: Optional[str] = None
    """Encoding format for the embeddings (e.g. 'float', 'base64')."""

    def _get_litellm_params(self) -> Dict[str, Any]:
        """Build parameter dict for litellm.embedding(), excluding None values."""
        params: Dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "custom_llm_provider": self.custom_llm_provider,
            "organization": self.organization,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "extra_headers": self.extra_headers,
            "dimensions": self.dimensions,
            "encoding_format": self.encoding_format,
            **self.model_kwargs,
        }
        return {k: v for k, v in params.items() if v is not None}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import litellm

        params = self._get_litellm_params()
        response = litellm.embedding(input=texts, **params)
        return [item["embedding"] for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import litellm

        params = self._get_litellm_params()
        response = await litellm.aembedding(input=texts, **params)
        return [item["embedding"] for item in response.data]

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return (await self.aembed_documents([text]))[0]
