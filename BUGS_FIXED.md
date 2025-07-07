# Critical Bugs Fixed in langchain-litellm

This document outlines the critical bugs and missing features that have been identified and fixed in the `feat/streaming-usage-metadata` branch.

## 🐛 Bug #1: Missing Usage Metadata in Streaming Responses

### **Issue Description**
Streaming responses from LangChain LiteLLM models do not include token usage metadata, making it impossible to track token consumption during streaming operations.

### **Impact**
- **High Priority**: Breaks token usage tracking for streaming applications
- **Affects**: All streaming operations across all providers (OpenAI, Anthropic, etc.)
- **User Experience**: No way to monitor costs or token usage in real-time streaming

### **Root Cause**
The `_default_params` method doesn't include `stream_options={"include_usage": True}` when streaming is enabled, and streaming methods don't ensure usage metadata is captured.

### **Files Affected**
- `langchain_litellm/chat_models/litellm.py` - `_default_params`, `_stream`, `_astream` methods

### **Fix Details**
```python
# Added automatic stream_options when streaming=True
if self.streaming:
    params["stream_options"] = {"include_usage": True}

# Enhanced streaming methods to capture and attach usage metadata
if "usage" in chunk and chunk["usage"]:
    usage_metadata = _create_usage_metadata(chunk["usage"])
    message_chunk.usage_metadata = usage_metadata
```

---

## 🐛 Bug #2: Missing reasoning_content Support for Thinking-Enabled Models

### **Issue Description**
Non-streaming responses from thinking-enabled models (like o1, Claude with thinking, Gemini with reasoning) lose the `reasoning_content` field, which contains the model's internal reasoning process.

### **Impact**
- **High Priority**: Critical feature missing for advanced AI models
- **Affects**: OpenAI o1 models, Claude thinking mode, Gemini reasoning mode
- **User Experience**: Loss of valuable reasoning insights that users pay premium for

### **Root Cause**
The `_convert_dict_to_message` function only handles `function_call` and `tool_calls` but ignores `reasoning_content` from the raw response.

### **Files Affected**
- `langchain_litellm/chat_models/litellm.py` - `_convert_dict_to_message` function

### **Fix Details**
```python
# Added reasoning_content support for thinking-enabled models
if _dict.get("reasoning_content"):
    additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
```

---

## 🐛 Bug #3: Inconsistent Delta Handling in Streaming

### **Issue Description**
The `_convert_delta_to_message_chunk` function assumes delta is always a Delta object, but LiteLLM sometimes returns dictionaries, causing AttributeError crashes.

### **Impact**
- **Medium Priority**: Streaming failures with certain providers
- **Affects**: Streaming responses that return dict deltas instead of Delta objects
- **User Experience**: Unexpected crashes during streaming

### **Root Cause**
Type assumption without proper handling of both Delta objects and dictionary formats.

### **Files Affected**
- `langchain_litellm/chat_models/litellm.py` - `_convert_delta_to_message_chunk` function

### **Fix Details**
```python
# Handle both Delta objects and dicts
if isinstance(delta, dict):
    role = delta.get("role")
    content = delta.get("content") or ""
    function_call = delta.get("function_call")
    raw_tool_calls = delta.get("tool_calls")
    reasoning_content = delta.get("reasoning_content")
else:
    role = delta.role
    content = delta.content or ""
    function_call = delta.function_call
    raw_tool_calls = delta.tool_calls
    reasoning_content = getattr(delta, "reasoning_content", None)
```

---

## 🐛 Bug #4: Fragile Tool Call Chunk Processing

### **Issue Description**
Tool call chunk processing assumes specific object structure and fails with KeyError when providers return different formats.

### **Impact**
- **Medium Priority**: Tool calling failures with certain providers
- **Affects**: Function calling and tool usage across different LLM providers
- **User Experience**: Crashes when using tools with certain providers

### **Root Cause**
Hardcoded attribute access without defensive programming for different provider response formats.

### **Files Affected**
- `langchain_litellm/chat_models/litellm.py` - `_convert_delta_to_message_chunk` function

### **Fix Details**
```python
# Robust tool call chunk processing with fallbacks
tool_call_chunks = [
    ToolCallChunk(
        name=rtc.function.name if hasattr(rtc, 'function') else rtc.get('function', {}).get('name'),
        args=rtc.function.arguments if hasattr(rtc, 'function') else rtc.get('function', {}).get('arguments'),
        id=rtc.id if hasattr(rtc, 'id') else rtc.get('id'),
        index=rtc.index if hasattr(rtc, 'index') else rtc.get('index'),
    )
    for rtc in raw_tool_calls
]
```

---

## 🐛 Bug #5: Incomplete Usage Metadata Creation

### **Issue Description**
The `_create_usage_metadata` function only captures basic token counts but ignores advanced usage details like cache tokens and reasoning tokens that are crucial for cost tracking and debugging.

### **Impact**
- **Medium Priority**: Incomplete usage tracking
- **Affects**: Cost optimization, debugging, advanced model features
- **User Experience**: Missing critical usage insights

### **Root Cause**
Minimal implementation that doesn't extract detailed token usage information.

### **Files Affected**
- `langchain_litellm/chat_models/litellm.py` - `_create_usage_metadata` function

### **Fix Details**
```python
# Extract advanced usage details
input_token_details = {}
output_token_details = {}

# Cache tokens (for providers that support it like OpenAI, Anthropic)
if "cache_read_input_tokens" in token_usage:
    input_token_details["cache_read"] = token_usage["cache_read_input_tokens"]

# Reasoning tokens (for o1 models, Claude thinking, etc.)
completion_tokens_details = token_usage.get("completion_tokens_details", {})
if completion_tokens_details and "reasoning_tokens" in completion_tokens_details:
    output_token_details["reasoning"] = completion_tokens_details["reasoning_tokens"]
```

---

## 🐛 Bug #6: Async Streaming Method Inconsistency

### **Issue Description**
The `_astream` method uses incorrect async completion call pattern, potentially causing async streaming failures.

### **Impact**
- **Low Priority**: Async streaming reliability issues
- **Affects**: Applications using async streaming
- **User Experience**: Potential async streaming failures

### **Root Cause**
Incorrect usage of async completion method in streaming context.

### **Files Affected**
- `langchain_litellm/chat_models/litellm.py` - `_astream` method

### **Fix Details**
```python
# Fixed async streaming method call
async for chunk in self.acompletion_with_retry(
    messages=message_dicts, run_manager=run_manager, **params
):
```

---

## 📊 Summary

| Bug | Priority | Impact | Status |
|-----|----------|--------|--------|
| Missing Streaming Usage Metadata | High | Breaks token tracking | ✅ Fixed |
| Missing reasoning_content | High | Loses premium model features | ✅ Fixed |
| Inconsistent Delta Handling | Medium | Streaming crashes | ✅ Fixed |
| Fragile Tool Call Processing | Medium | Tool calling failures | ✅ Fixed |
| Incomplete Usage Metadata | Medium | Missing usage insights | ✅ Fixed |
| Async Streaming Issues | Low | Async reliability | ✅ Fixed |

## 🧪 Testing

Comprehensive test suite added in:
- `tests/integration_tests/test_streaming_usage_metadata.py`
- `tests/unit_tests/test_litellm.py` (enhanced)

## 🔗 Related Issues

These fixes address fundamental issues that affect:
- Token usage tracking and cost monitoring
- Advanced AI model features (thinking, reasoning)
- Streaming reliability across providers
- Tool calling robustness
- Async operation consistency

All fixes maintain backward compatibility while adding essential missing functionality.