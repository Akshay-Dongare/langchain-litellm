"""Test chat model integration."""

from typing import Type

from langchain_core.messages import AIMessageChunk
from langchain_tests.unit_tests import ChatModelUnitTests
from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function

from langchain_litellm.chat_models import ChatLiteLLM
from langchain_litellm.chat_models.litellm import _convert_delta_to_message_chunk


class TestChatLiteLLMUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatLiteLLM]:
        return ChatLiteLLM

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "custom_llm_provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "<your_api_key>",
            "max_retries": 1,
        }

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False

    def test_litellm_delta_to_langchain_message_chunk(self):
        """Test the litellm._convert_delta_to_message_chunk method, to ensure compatibility when converting a LiteLLM delta to a LangChain message chunk."""
        mock_content = "This is a test content"
        mock_tool_call_id = "call_test"
        mock_tool_call_name = "test_tool_call"
        mock_tool_call_arguments = ""
        mock_tool_call_index = 3
        mock_delta = Delta(
            content=mock_content,
            role="assistant",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id=mock_tool_call_id,
                    function=Function(
                        arguments=mock_tool_call_arguments, name=mock_tool_call_name
                    ),
                    type="function",
                    index=mock_tool_call_index,
                )
            ],
        )
        message_chunk = _convert_delta_to_message_chunk(mock_delta, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert message_chunk.content == mock_content
        tool_call_chunk = message_chunk.tool_call_chunks[0]
        assert tool_call_chunk["id"] == mock_tool_call_id
        assert tool_call_chunk["name"] == mock_tool_call_name
        assert tool_call_chunk["args"] == mock_tool_call_arguments
        assert tool_call_chunk["index"] == mock_tool_call_index

    def test_convert_dict_to_tool_message(self):
        """Ensure tool role dicts convert to ToolMessage."""
        from langchain_litellm.chat_models.litellm import _convert_dict_to_message

        mock_dict = {"role": "tool", "content": "result", "tool_call_id": "123"}
        message = _convert_dict_to_message(mock_dict)
        from langchain_core.messages import ToolMessage

        assert isinstance(message, ToolMessage)
        assert message.content == "result"
        assert message.tool_call_id == "123"

    def test_default_params_includes_stream_options_when_streaming(self):
        """Test that _default_params includes stream_options when streaming is enabled."""
        from langchain_litellm.chat_models.litellm import ChatLiteLLM

        # Test with streaming=True
        llm = ChatLiteLLM(model="gpt-3.5-turbo", streaming=True)
        params = llm._default_params
        assert "stream_options" in params
        assert params["stream_options"] == {"include_usage": True}

        # Test with streaming=False
        llm_no_stream = ChatLiteLLM(model="gpt-3.5-turbo", streaming=False)
        params_no_stream = llm_no_stream._default_params
        assert "stream_options" not in params_no_stream

    def test_create_usage_metadata_basic(self):
        """Test _create_usage_metadata with basic token usage."""
        from langchain_litellm.chat_models.litellm import _create_usage_metadata

        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }

        usage_metadata = _create_usage_metadata(token_usage)
        assert usage_metadata["input_tokens"] == 10
        assert usage_metadata["output_tokens"] == 20
        assert usage_metadata["total_tokens"] == 30
        assert usage_metadata["input_token_details"] == {}
        assert usage_metadata["output_token_details"] == {}

    def test_create_usage_metadata_with_cache_tokens(self):
        """Test _create_usage_metadata with cache tokens."""
        from langchain_litellm.chat_models.litellm import _create_usage_metadata

        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cache_read_input_tokens": 5
        }

        usage_metadata = _create_usage_metadata(token_usage)
        assert usage_metadata["input_tokens"] == 10
        assert usage_metadata["output_tokens"] == 20
        assert usage_metadata["total_tokens"] == 30
        assert usage_metadata["input_token_details"] == {"cache_read": 5}
        assert usage_metadata["output_token_details"] == {}

    def test_create_usage_metadata_with_reasoning_tokens(self):
        """Test _create_usage_metadata with reasoning tokens."""
        from langchain_litellm.chat_models.litellm import _create_usage_metadata

        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "completion_tokens_details": {
                "reasoning_tokens": 15
            }
        }

        usage_metadata = _create_usage_metadata(token_usage)
        assert usage_metadata["input_tokens"] == 10
        assert usage_metadata["output_tokens"] == 20
        assert usage_metadata["total_tokens"] == 30
        assert usage_metadata["input_token_details"] == {}
        assert usage_metadata["output_token_details"] == {"reasoning": 15}

    def test_create_usage_metadata_with_all_advanced_fields(self):
        """Test _create_usage_metadata with all advanced fields."""
        from langchain_litellm.chat_models.litellm import _create_usage_metadata

        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cache_read_input_tokens": 5,
            "completion_tokens_details": {
                "reasoning_tokens": 15
            }
        }

        usage_metadata = _create_usage_metadata(token_usage)
        assert usage_metadata["input_tokens"] == 10
        assert usage_metadata["output_tokens"] == 20
        assert usage_metadata["total_tokens"] == 30
        assert usage_metadata["input_token_details"] == {"cache_read": 5}
        assert usage_metadata["output_token_details"] == {"reasoning": 15}
