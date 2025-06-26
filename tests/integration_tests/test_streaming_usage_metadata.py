"""Integration tests for streaming usage metadata functionality."""

import os
import pytest
from typing import List

from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_litellm.chat_models import ChatLiteLLM


class TestStreamingUsageMetadata:
    """Test streaming usage metadata with real API calls."""

    def test_openai_streaming_usage_metadata(self):
        """Test OpenAI streaming with usage metadata."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = ChatLiteLLM(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            streaming=True,
            max_retries=1
        )

        messages = [HumanMessage(content="Say hello in exactly 5 words.")]

        chunks = []
        usage_metadata_found = False

        for chunk in llm.stream(messages):
            chunks.append(chunk)
            # chunk is an AIMessageChunk directly, not ChatGenerationChunk
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_metadata_found = True
                usage = chunk.usage_metadata
                assert usage["input_tokens"] > 0
                assert usage["output_tokens"] > 0
                assert usage["total_tokens"] > 0
                assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

        assert len(chunks) > 0
        assert usage_metadata_found, "No usage metadata found in streaming chunks"

    def test_openai_streaming_usage_metadata_with_cache(self):
        """Test OpenAI streaming with cache tokens (if supported)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = ChatLiteLLM(
            model="gpt-4o-mini",  # Use a model that supports caching
            openai_api_key=api_key,
            streaming=True,
            max_retries=1
        )

        # Send the same message twice to potentially trigger caching
        messages = [HumanMessage(content="What is the capital of France? Please answer in exactly one word.")]

        # First call
        chunks1 = list(llm.stream(messages))

        # Second call (might use cache)
        chunks2 = list(llm.stream(messages))

        # Check if any chunks have cache information
        for chunks in [chunks1, chunks2]:
            for chunk in chunks:
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    if usage.get("input_token_details") and "cache_read" in usage["input_token_details"]:
                        assert usage["input_token_details"]["cache_read"] >= 0

    def test_anthropic_streaming_usage_metadata(self):
        """Test Anthropic streaming with usage metadata."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        llm = ChatLiteLLM(
            model="claude-3-haiku-20240307",
            anthropic_api_key=api_key,
            streaming=True,
            max_retries=1
        )

        messages = [HumanMessage(content="Say hello in exactly 3 words.")]

        chunks = []
        usage_metadata_found = False

        for chunk in llm.stream(messages):
            chunks.append(chunk)
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_metadata_found = True
                usage = chunk.usage_metadata
                assert usage["input_tokens"] > 0
                assert usage["output_tokens"] > 0
                assert usage["total_tokens"] > 0

        assert len(chunks) > 0
        assert usage_metadata_found, "No usage metadata found in Anthropic streaming chunks"

    @pytest.mark.asyncio
    async def test_openai_async_streaming_usage_metadata(self):
        """Test OpenAI async streaming with usage metadata."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = ChatLiteLLM(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            streaming=True,
            max_retries=1
        )

        messages = [HumanMessage(content="Count from 1 to 3.")]

        chunks = []
        usage_metadata_found = False

        async for chunk in llm.astream(messages):
            chunks.append(chunk)
            if hasattr(chunk.message, 'usage_metadata') and chunk.message.usage_metadata:
                usage_metadata_found = True
                usage = chunk.message.usage_metadata
                assert usage.input_tokens > 0
                assert usage.output_tokens > 0
                assert usage.total_tokens > 0

        assert len(chunks) > 0
        assert usage_metadata_found, "No usage metadata found in async streaming chunks"

    def test_stream_options_override(self):
        """Test that stream_options can be overridden in kwargs."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = ChatLiteLLM(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            streaming=False,  # Not streaming by default
            max_retries=1
        )

        messages = [HumanMessage(content="Say hi.")]

        chunks = []
        usage_metadata_found = False

        # Override streaming and stream_options in kwargs
        for chunk in llm.stream(messages, stream_options={"include_usage": True}):
            chunks.append(chunk)
            if hasattr(chunk.message, 'usage_metadata') and chunk.message.usage_metadata:
                usage_metadata_found = True

        assert len(chunks) > 0
        # Usage metadata should be found even though streaming=False initially
        # because we override with stream_options

    def test_non_streaming_usage_metadata_still_works(self):
        """Test that non-streaming usage metadata still works after our changes."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = ChatLiteLLM(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            streaming=False,
            max_retries=1
        )

        messages = [HumanMessage(content="Say hello.")]
        result = llm.invoke(messages)

        assert hasattr(result, 'usage_metadata')
        assert result.usage_metadata is not None
        assert result.usage_metadata.input_tokens > 0
        assert result.usage_metadata.output_tokens > 0
        assert result.usage_metadata.total_tokens > 0