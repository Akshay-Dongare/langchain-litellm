<!-- START BADGE TABLE -->
<table>
<thead>
<tr>
<th align="center">📦 Distribution</th>
<th align="center">🔧 Project</th>
<th align="center">🚀 Activity</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">
<a href="https://pypi.org/project/langchain-litellm/">
<img src="https://img.shields.io/pypi/v/langchain-litellm?label=PyPI%20package&style=flat" alt="PyPI Package Version">
</a><br/>
<a href="https://pepy.tech/projects/langchain-litellm">
<img src="https://static.pepy.tech/personalized-badge/langchain-litellm?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads" alt="PyPI Downloads">
</a><br/>
<a href="https://github.com/Akshay-Dongare/langchain-litellm/actions/workflows/pypi-release.yml">
</a>
<a href="https://opensource.org/licenses/MIT">
<img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="License: MIT">
</a>
</td>

<td align="center">
<img src="https://img.shields.io/badge/Platform-Linux%2C%20Windows%2C%20macOS-blue" alt="Platform">
<br>
<a href="https://www.python.org">
<img src="https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python">
</a><br/>

<a href="https://python-poetry.org/">
<img src="https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json" alt="Poetry">
</a>
</td>

<td align="center">
<img src="https://img.shields.io/github/issues-closed/Akshay-Dongare/langchain-litellm" alt="GitHub Issues Closed"><br/>
<img src="https://img.shields.io/github/issues/Akshay-Dongare/langchain-litellm" alt="GitHub Issues Open"><br/>
<img src="https://img.shields.io/github/issues-pr/Akshay-Dongare/langchain-litellm" alt="GitHub PRs Open"><br/>
<img src="https://img.shields.io/github/issues-pr-closed/Akshay-Dongare/langchain-litellm" alt="GitHub PRs Closed">
</td>
</tr>
</tbody>
</table>
<!-- END BADGE TABLE -->

# [langchain-litellm](https://pypi.org/project/langchain-litellm/)

This package contains the [LangChain](https://github.com/langchain-ai/langchain) integration with LiteLLM. [LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Anthropic, Azure, Huggingface, Replicate, etc.

## Installation and setup

```bash
pip install -U langchain-litellm
```

## Chat Models
```python
from langchain_litellm import ChatLiteLLM
```

```python
from langchain_litellm import ChatLiteLLMRouter
```
See a [usage example](https://github.com/Akshay-Dongare/langchain-litellm/blob/main/docs/litellm.ipynb)

## Advanced Features

<details>
<summary><strong>Vertex AI Grounding (Google Search)</strong></summary>

_Supported in v0.3.5+_

You can use Google Search grounding with Vertex AI models (e.g., `gemini-2.5-flash`). Citations and metadata are returned in `response_metadata` (Batch) or `additional_kwargs` (Streaming).

**Setup**

```python
import os
from langchain_litellm import ChatLiteLLM

os.environ["VERTEX_PROJECT"] = "your-project-id"
os.environ["VERTEX_LOCATION"] = "us-central1"

llm = ChatLiteLLM(model="vertex_ai/gemini-2.5-flash", temperature=0)
```

**Batch Usage**

```python
# Invoke with Google Search tool enabled
response = llm.invoke(
    "What is the current stock price of Google?",
    tools=[{"googleSearch": {}}]
)

# Access Citations & Metadata
provider_fields = response.response_metadata.get("provider_specific_fields")
if provider_fields:
    # Vertex returns a list; the first item contains the grounding info
    print(provider_fields[0])
```

**Streaming Usage**

```python
stream = llm.stream(
    "What is the current stock price of Google?",
    tools=[{"googleSearch": {}}]
)

for chunk in stream:
    print(chunk.content, end="", flush=True)
    # Metadata is injected into the chunk where it arrives
    if "provider_specific_fields" in chunk.additional_kwargs:
        print("\n[Metadata Found]:", chunk.additional_kwargs["provider_specific_fields"])
```
</details>
