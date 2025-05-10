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
