"""Test usage metadata functionality."""

from langchain_litellm.chat_models.litellm import _create_usage_metadata


def test_create_usage_metadata_basic():
    """Test _create_usage_metadata with basic token usage."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    }

    metadata = _create_usage_metadata(token_usage)

    assert metadata.input_tokens == 100
    assert metadata.output_tokens == 50
    assert metadata.total_tokens == 150
    assert metadata.input_token_details == {}
    assert metadata.output_token_details == {}


def test_create_usage_metadata_with_cache_tokens():
    """Test _create_usage_metadata with cache tokens."""
    token_usage = {
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
        "cache_read_input_tokens": 150,
        "cache_creation_input_tokens": 50
    }

    metadata = _create_usage_metadata(token_usage)

    assert metadata.input_tokens == 200
    assert metadata.output_tokens == 100
    assert metadata.total_tokens == 300
    assert metadata.input_token_details["cache_read"] == 150
    assert metadata.input_token_details["cache_creation"] == 50
    assert metadata.output_token_details == {}


def test_create_usage_metadata_with_audio_tokens():
    """Test _create_usage_metadata with audio tokens for multimodal models."""
    token_usage = {
        "prompt_tokens": 300,
        "completion_tokens": 150,
        "total_tokens": 450,
        "audio_input_tokens": 25,
        "audio_output_tokens": 35
    }

    metadata = _create_usage_metadata(token_usage)

    assert metadata.input_tokens == 300
    assert metadata.output_tokens == 150
    assert metadata.total_tokens == 450
    assert metadata.input_token_details["audio"] == 25
    assert metadata.output_token_details["audio"] == 35


def test_create_usage_metadata_with_reasoning_tokens():
    """Test _create_usage_metadata with reasoning tokens for thinking models."""
    token_usage = {
        "prompt_tokens": 400,
        "completion_tokens": 200,
        "total_tokens": 600,
        "completion_tokens_details": {
            "reasoning_tokens": 180
        }
    }

    metadata = _create_usage_metadata(token_usage)

    assert metadata.input_tokens == 400
    assert metadata.output_tokens == 200
    assert metadata.total_tokens == 600
    assert metadata.input_token_details == {}
    assert metadata.output_token_details["reasoning"] == 180


def test_create_usage_metadata_complete_schema():
    """Test _create_usage_metadata with complete schema including all token types."""
    token_usage = {
        "prompt_tokens": 350,
        "completion_tokens": 240,
        "total_tokens": 590,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 200,
        "audio_input_tokens": 10,
        "audio_output_tokens": 10,
        "completion_tokens_details": {
            "reasoning_tokens": 200
        }
    }

    metadata = _create_usage_metadata(token_usage)

    # Basic tokens
    assert metadata.input_tokens == 350
    assert metadata.output_tokens == 240
    assert metadata.total_tokens == 590

    # Input token details
    assert metadata.input_token_details["cache_read"] == 100
    assert metadata.input_token_details["cache_creation"] == 200
    assert metadata.input_token_details["audio"] == 10

    # Output token details
    assert metadata.output_token_details["audio"] == 10
    assert metadata.output_token_details["reasoning"] == 200


def test_create_usage_metadata_edge_cases():
    """Test _create_usage_metadata with edge cases and missing fields."""
    # Test with empty completion_tokens_details
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "completion_tokens_details": {}
    }

    metadata = _create_usage_metadata(token_usage)
    assert metadata.output_token_details == {}

    # Test with missing completion_tokens_details
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50
    }

    metadata = _create_usage_metadata(token_usage)
    assert metadata.output_token_details == {}

    # Test with zero tokens
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "audio_input_tokens": 0,
        "audio_output_tokens": 0
    }

    metadata = _create_usage_metadata(token_usage)
    assert metadata.input_tokens == 0
    assert metadata.output_tokens == 0
    assert metadata.total_tokens == 0
    assert metadata.input_token_details["cache_read"] == 0
    assert metadata.input_token_details["cache_creation"] == 0
    assert metadata.input_token_details["audio"] == 0
    assert metadata.output_token_details["audio"] == 0


def test_create_usage_metadata_missing_optional_fields():
    """Test _create_usage_metadata with missing optional fields."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
        # No cache, audio, or reasoning tokens
    }

    metadata = _create_usage_metadata(token_usage)

    assert metadata.input_tokens == 100
    assert metadata.output_tokens == 50
    assert metadata.total_tokens == 150
    assert metadata.input_token_details == {}
    assert metadata.output_token_details == {}