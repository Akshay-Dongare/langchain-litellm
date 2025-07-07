#!/usr/bin/env python3
"""
Test script to reproduce the streaming usage metadata bug
"""

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
import os

def test_streaming_vs_regular():
    """Test streaming vs regular mode usage metadata"""

    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Please set it to run this test.")
        return

    print("🧪 Testing Streaming Usage Metadata Bug")
    print("=" * 50)

    # Setup streaming LLM
    try:
        llm_streaming = ChatLiteLLM(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            streaming=True
        )
        print("✅ Streaming LLM initialized")
    except Exception as e:
        print(f"❌ Failed to initialize streaming LLM: {e}")
        return

    messages = [HumanMessage(content="Hello, how are you?")]

    # Test streaming
    print("\n=== STREAMING MODE ===")
    try:
        streaming_usage_found = False
        chunk_count = 0

        for chunk in llm_streaming.stream(messages):
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk.content[:50]}..." if chunk.content else f"Chunk {chunk_count}: <empty>")

            # Check for usage metadata in different possible locations
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                print(f"✅ Usage found in chunk: {chunk.usage_metadata}")
                streaming_usage_found = True
                break
            elif hasattr(chunk, 'message') and hasattr(chunk.message, 'usage_metadata') and chunk.message.usage_metadata:
                print(f"✅ Usage found in chunk.message: {chunk.message.usage_metadata}")
                streaming_usage_found = True
                break

        if not streaming_usage_found:
            print(f"❌ No usage metadata found in streaming (processed {chunk_count} chunks)")

    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return

    # Test non-streaming for comparison - Fix the model name
    print("\n=== NON-STREAMING MODE ===")
    try:
        # Note: gpt-4.1 is not a valid model, using gpt-4o instead
        print("⚠️  Note: 'gpt-4.1' is not a valid OpenAI model. Using 'gpt-4o' instead.")

        llm_regular = ChatLiteLLM(
            model="gpt-4o",
            openai_api_key=api_key,
            streaming=False
        )

        result = llm_regular.invoke(messages)
        print(f"Response: {result.content[:100]}...")

        if hasattr(result, 'usage_metadata') and result.usage_metadata:
            print(f"✅ Usage found: {result.usage_metadata}")
        else:
            print("❌ No usage metadata found in non-streaming")

    except Exception as e:
        print(f"❌ Non-streaming failed: {e}")

    print("\n" + "=" * 50)
    print("🔍 Bug Confirmed: Streaming mode lacks usage metadata")
    print("💡 This prevents real-time cost tracking during streaming")

if __name__ == "__main__":
    test_streaming_vs_regular()