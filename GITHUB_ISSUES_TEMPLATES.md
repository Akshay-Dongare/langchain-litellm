# GitHub Issue Templates

Copy-paste these templates directly into GitHub issues.

---

## Issue #1: Missing Usage Metadata in Streaming Responses

**Title**: `[BUG] Streaming responses missing token usage metadata`

**Labels**: `bug`, `high-priority`, `streaming`, `usage-tracking`

**Description**:

### 🐛 Bug Description
Streaming responses from LangChain LiteLLM models do not include token usage metadata, making it impossible to track token consumption during streaming operations.

### 🔥 Impact
- **Priority**: High
- **Affects**: All streaming operations across all providers (OpenAI, Anthropic, etc.)
- **User Impact**: No way to monitor costs or token usage in real-time streaming applications

### 🔍 Root Cause
The `_default_params` method doesn't include `stream_options={"include_usage": True}` when streaming is enabled, and streaming methods don't capture usage metadata from chunks.

### 📋 Steps to Reproduce
```python
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage

llm = ChatLiteLLM(model="gpt-3.5-turbo", streaming=True)
for chunk in llm.stream([HumanMessage(content="Hello")]):
    print(chunk.usage_metadata)  # Always None
```

### ✅ Expected Behavior
Each streaming chunk should include usage metadata with token counts.

### 💡 Proposed Solution
1. Add `stream_options={"include_usage": True}` to `_default_params` when streaming
2. Capture usage metadata from chunks in `_stream` and `_astream` methods
3. Attach usage metadata to message chunks

### 📁 Files Affected
- `langchain_litellm/chat_models/litellm.py`

---

## Issue #2: Missing reasoning_content Support for Thinking Models

**Title**: `[BUG] reasoning_content lost for thinking-enabled models (o1, Claude thinking, Gemini reasoning)`

**Labels**: `bug`, `high-priority`, `reasoning`, `thinking-models`, `feature-loss`

**Description**:

### 🐛 Bug Description
Non-streaming responses from thinking-enabled models (OpenAI o1, Claude with thinking, Gemini with reasoning) lose the `reasoning_content` field, which contains the model's valuable internal reasoning process.

### 🔥 Impact
- **Priority**: High
- **Affects**: OpenAI o1 models, Claude thinking mode, Gemini reasoning mode
- **User Impact**: Loss of valuable reasoning insights that users pay premium prices for

### 🔍 Root Cause
The `_convert_dict_to_message` function only handles `function_call` and `tool_calls` but completely ignores `reasoning_content` from the raw LiteLLM response.

### 📋 Steps to Reproduce
```python
import litellm
from langchain_litellm import ChatLiteLLM

# Direct LiteLLM call - reasoning_content available
response = litellm.completion(
    model="vertex_ai/gemini-2.5-flash",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    thinking={"type": "enabled", "budget_tokens": 1024}
)
print(response.choices[0].message.reasoning_content)  # Works

# LangChain wrapper - reasoning_content lost
llm = ChatLiteLLM(model="vertex_ai/gemini-2.5-flash")
result = llm.invoke("What is 2+2?")
print(result.additional_kwargs.get("reasoning_content"))  # None
```

### ✅ Expected Behavior
The reasoning content should be available in `response.additional_kwargs["reasoning_content"]`.

### 💡 Proposed Solution
Add reasoning_content handling to `_convert_dict_to_message`:
```python
if _dict.get("reasoning_content"):
    additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
```

### 📁 Files Affected
- `langchain_litellm/chat_models/litellm.py` - `_convert_dict_to_message` function

---

## Issue #3: Streaming Crashes with Dictionary Deltas

**Title**: `[BUG] AttributeError in streaming when delta is dict instead of Delta object`

**Labels**: `bug`, `medium-priority`, `streaming`, `type-error`

**Description**:

### 🐛 Bug Description
The `_convert_delta_to_message_chunk` function assumes delta is always a Delta object, but LiteLLM sometimes returns dictionaries, causing AttributeError crashes during streaming.

### 🔥 Impact
- **Priority**: Medium
- **Affects**: Streaming responses that return dict deltas instead of Delta objects
- **User Impact**: Unexpected crashes during streaming operations

### 🔍 Root Cause
Type assumption without proper handling of both Delta objects and dictionary formats.

### 📋 Steps to Reproduce
Occurs intermittently with certain providers when they return dict deltas instead of Delta objects.

### ✅ Expected Behavior
Streaming should handle both Delta objects and dictionary formats gracefully.

### 💡 Proposed Solution
Add type checking and handle both formats:
```python
if isinstance(delta, dict):
    role = delta.get("role")
    content = delta.get("content") or ""
    # ... handle dict format
else:
    role = delta.role
    content = delta.content or ""
    # ... handle Delta object format
```

### 📁 Files Affected
- `langchain_litellm/chat_models/litellm.py` - `_convert_delta_to_message_chunk` function

---

## Issue #4: Tool Call Processing Failures

**Title**: `[BUG] KeyError in tool call chunk processing with certain providers`

**Labels**: `bug`, `medium-priority`, `tool-calling`, `provider-compatibility`

**Description**:

### 🐛 Bug Description
Tool call chunk processing assumes specific object structure and fails with KeyError when providers return different response formats.

### 🔥 Impact
- **Priority**: Medium
- **Affects**: Function calling and tool usage across different LLM providers
- **User Impact**: Crashes when using tools with certain providers

### 🔍 Root Cause
Hardcoded attribute access without defensive programming for different provider response formats.

### 📋 Steps to Reproduce
Use tool calling with providers that return different tool call formats.

### ✅ Expected Behavior
Tool call processing should work reliably across all providers.

### 💡 Proposed Solution
Add robust attribute access with fallbacks:
```python
name=rtc.function.name if hasattr(rtc, 'function') else rtc.get('function', {}).get('name')
```

### 📁 Files Affected
- `langchain_litellm/chat_models/litellm.py` - `_convert_delta_to_message_chunk` function

---

## Issue #5: Incomplete Usage Metadata

**Title**: `[FEATURE] Missing advanced usage details (cache tokens, reasoning tokens)`

**Labels**: `enhancement`, `medium-priority`, `usage-tracking`, `cost-optimization`

**Description**:

### 🔧 Feature Description
The `_create_usage_metadata` function only captures basic token counts but ignores advanced usage details like cache tokens and reasoning tokens that are crucial for cost tracking and debugging.

### 🔥 Impact
- **Priority**: Medium
- **Affects**: Cost optimization, debugging, advanced model features
- **User Impact**: Missing critical usage insights for advanced models

### 📋 Current Behavior
Only basic token counts are captured: `input_tokens`, `output_tokens`, `total_tokens`.

### ✅ Desired Behavior
Should capture advanced details:
- Cache tokens (`cache_read_input_tokens`)
- Reasoning tokens (`completion_tokens_details.reasoning_tokens`)
- Other provider-specific usage details

### 💡 Proposed Solution
Enhance `_create_usage_metadata` to extract detailed usage information:
```python
# Cache tokens
if "cache_read_input_tokens" in token_usage:
    input_token_details["cache_read"] = token_usage["cache_read_input_tokens"]

# Reasoning tokens
completion_tokens_details = token_usage.get("completion_tokens_details", {})
if completion_tokens_details and "reasoning_tokens" in completion_tokens_details:
    output_token_details["reasoning"] = completion_tokens_details["reasoning_tokens"]
```

### 📁 Files Affected
- `langchain_litellm/chat_models/litellm.py` - `_create_usage_metadata` function

---

## Issue #6: Async Streaming Reliability

**Title**: `[BUG] Incorrect async completion call in _astream method`

**Labels**: `bug`, `low-priority`, `async`, `streaming`

**Description**:

### 🐛 Bug Description
The `_astream` method uses incorrect async completion call pattern, potentially causing async streaming failures.

### 🔥 Impact
- **Priority**: Low
- **Affects**: Applications using async streaming
- **User Impact**: Potential async streaming reliability issues

### 🔍 Root Cause
Incorrect usage of async completion method in streaming context.

### 💡 Proposed Solution
Fix the async streaming method call:
```python
async for chunk in self.acompletion_with_retry(
    messages=message_dicts, run_manager=run_manager, **params
):
```

### 📁 Files Affected
- `langchain_litellm/chat_models/litellm.py` - `_astream` method