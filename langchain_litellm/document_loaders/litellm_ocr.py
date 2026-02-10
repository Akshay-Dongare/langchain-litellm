"""LiteLLM OCR Document Loader using LiteLLM proxy."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class LiteLLMOCRLoader(BaseLoader):
    """Load documents using LiteLLM proxy's OCR endpoint.

    This loader makes HTTP requests to a LiteLLM proxy server configured
    with Azure Document Intelligence (or other OCR providers). The proxy
    handles all provider-specific authentication and configuration.

    Args:
        proxy_base_url: Base URL of the LiteLLM proxy server.
            Defaults to "http://localhost:4000".
        api_key: Optional bearer token for proxy authentication.
        model: Model name configured in the proxy (e.g., "azure-document").
            Defaults to "azure-document".
        file_path: Path to a local file to process.
        url_path: URL to a remote document to process.
        base64_content: Base64-encoded document content.
        bytes_content: Raw bytes of a document.
        mode: Output mode - "single" returns one document with all content,
            "page" returns one document per page. Defaults to "single".
        timeout: Timeout in seconds for HTTP requests. Defaults to 300.
        max_retries: Maximum number of retry attempts for failed requests.
            Uses exponential backoff between retries. Defaults to 3.

    Note:
        Exactly one of file_path, url_path, base64_content, or bytes_content
        must be provided.

    Example:
        Basic usage with default proxy:

        ```python
        from langchain_litellm import LiteLLMOCRLoader

        loader = LiteLLMOCRLoader(
            url_path="https://example.com/document.pdf",
            model="azure-document",
            mode="page"
        )
        documents = loader.load()
        ```

        With custom proxy and authentication:

        ```python
        loader = LiteLLMOCRLoader(
            proxy_base_url="https://my-proxy.com",
            api_key="my-bearer-token",
            file_path="/path/to/document.pdf",
            model="azure-document",
            mode="single"
        )
        documents = await loader.aload()
        ```
    """

    def __init__(
        self,
        *,
        proxy_base_url: str = "http://localhost:4000",
        api_key: Optional[str] = None,
        model: str = "azure-document",
        file_path: Optional[str] = None,
        url_path: Optional[str] = None,
        base64_content: Optional[str] = None,
        bytes_content: Optional[bytes] = None,
        mode: Literal["single", "page"] = "single",
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the LiteLLM OCR loader."""
        # Validate input sources
        input_sources = [file_path, url_path, base64_content, bytes_content]
        provided_sources = [s for s in input_sources if s is not None]

        if len(provided_sources) == 0:
            raise ValueError(
                "Must provide exactly one of: file_path, url_path, "
                "base64_content, or bytes_content"
            )
        if len(provided_sources) > 1:
            raise ValueError(
                "Must provide exactly one of: file_path, url_path, "
                "base64_content, or bytes_content. "
                f"Provided {len(provided_sources)} sources."
            )

        # Validate mode
        if mode not in ("single", "page"):
            raise ValueError(f"mode must be 'single' or 'page', got: {mode}")

        # Validate proxy URL format
        if not proxy_base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"proxy_base_url must start with http:// or https://, "
                f"got: {proxy_base_url}"
            )

        self.proxy_base_url = proxy_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.file_path = file_path
        self.url_path = url_path
        self.base64_content = base64_content
        self.bytes_content = bytes_content
        self.mode = mode
        self.timeout = timeout
        self.max_retries = max_retries

    def _prepare_document_payload(self) -> Dict[str, Any]:
        """Prepare the document payload for the OCR request.

        Returns:
            Dict with 'type' and 'document_url' keys in LiteLLM format.
        """
        if self.url_path:
            # Direct URL
            return {
                "type": "document_url",
                "document_url": self.url_path
            }

        elif self.file_path:
            # Read file and convert to base64 data URI
            file_path = Path(self.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")

            # Read file bytes
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                # Default to PDF if unknown
                mime_type = "application/pdf"

            # Create base64 data URI
            b64_data = base64.b64encode(file_bytes).decode("utf-8")
            data_uri = f"data:{mime_type};base64,{b64_data}"

            return {
                "type": "document_url",
                "document_url": data_uri
            }

        elif self.base64_content:
            # User provided base64, wrap in data URI
            # Assume PDF if no MIME type prefix exists
            if self.base64_content.startswith("data:"):
                data_uri = self.base64_content
            else:
                data_uri = f"data:application/pdf;base64,{self.base64_content}"

            return {
                "type": "document_url",
                "document_url": data_uri
            }

        elif self.bytes_content:
            # Convert bytes to base64 data URI
            b64_data = base64.b64encode(self.bytes_content).decode("utf-8")
            data_uri = f"data:application/pdf;base64,{b64_data}"

            return {
                "type": "document_url",
                "document_url": data_uri
            }

        else:
            raise ValueError("No input source provided")

    def _make_ocr_request(
        self,
        document_payload: Dict[str, Any],
        sync: bool = True
    ) -> Dict[str, Any]:
        """Make synchronous or asynchronous OCR request to LiteLLM proxy.

        Args:
            document_payload: Document payload dict.
            sync: Whether to make a synchronous request (True) or return
                an awaitable (False).

        Returns:
            Response JSON as a dict.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for LiteLLMOCRLoader. "
                "Install it with: pip install httpx"
            )

        url = f"{self.proxy_base_url}/ocr"
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "document": document_payload
        }

        if sync:
            last_exception: Optional[Exception] = None
            for attempt in range(self.max_retries):
                with httpx.Client(timeout=self.timeout) as client:
                    try:
                        response = client.post(
                            url, json=payload, headers=headers
                        )
                        response.raise_for_status()
                        return response.json()
                    except httpx.HTTPStatusError as e:
                        last_exception = RuntimeError(
                            f"HTTP error from LiteLLM proxy: "
                            f"{e.response.status_code} "
                            f"{e.response.text}"
                        )
                        last_exception.__cause__ = e
                    except httpx.RequestError as e:
                        last_exception = RuntimeError(
                            f"Failed to connect to LiteLLM proxy at {url}. "
                            f"Is the proxy running? Error: {e}"
                        )
                        last_exception.__cause__ = e

                if attempt < self.max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Request attempt %d/%d failed, retrying in %ds: %s",
                        attempt + 1,
                        self.max_retries,
                        wait_time,
                        last_exception,
                    )
                    time.sleep(wait_time)

            raise last_exception  # type: ignore[misc]
        else:
            # Return a coroutine for async
            async def _async_request() -> Dict[str, Any]:
                last_exception: Optional[Exception] = None
                for attempt in range(self.max_retries):
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        try:
                            response = await client.post(
                                url, json=payload, headers=headers
                            )
                            response.raise_for_status()
                            return response.json()
                        except httpx.HTTPStatusError as e:
                            last_exception = RuntimeError(
                                f"HTTP error from LiteLLM proxy: "
                                f"{e.response.status_code} "
                                f"{e.response.text}"
                            )
                            last_exception.__cause__ = e
                        except httpx.RequestError as e:
                            last_exception = RuntimeError(
                                f"Failed to connect to LiteLLM proxy at "
                                f"{url}. "
                                f"Is the proxy running? Error: {e}"
                            )
                            last_exception.__cause__ = e

                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(
                            "Request attempt %d/%d failed, retrying in "
                            "%ds: %s",
                            attempt + 1,
                            self.max_retries,
                            wait_time,
                            last_exception,
                        )
                        await asyncio.sleep(wait_time)

                raise last_exception  # type: ignore[misc]

            return _async_request()

    def _process_response(self, response: Dict[str, Any]) -> List[Document]:
        """Process OCR response and return LangChain Documents.

        Args:
            response: Response JSON from LiteLLM proxy.

        Returns:
            List of Document objects.
        """
        if "pages" not in response:
            raise ValueError(
                f"Invalid response from LiteLLM proxy: missing 'pages' field. "
                f"Response: {response}"
            )

        pages = response["pages"]

        if self.mode == "page":
            # Return one Document per page
            documents = []
            for page in pages:
                page_content = page.get("markdown", "")

                metadata: Dict[str, Any] = {
                    "page": page.get("index", 0),
                }

                # Add dimensions if available
                if "dimensions" in page:
                    dimensions = page["dimensions"]
                    metadata["width"] = dimensions.get("width")
                    metadata["height"] = dimensions.get("height")

                # Add source info
                if self.file_path:
                    metadata["source"] = self.file_path
                elif self.url_path:
                    metadata["source"] = self.url_path

                # Add model info
                if "model" in response:
                    metadata["model"] = response["model"]

                documents.append(Document(page_content=page_content, metadata=metadata))

            return documents

        else:  # mode == "single"
            # Concatenate all pages
            all_content = "\n\n".join(
                page.get("markdown", "") for page in pages
            )

            metadata: Dict[str, Any] = {
                "total_pages": len(pages),
            }

            # Add source info
            if self.file_path:
                metadata["source"] = self.file_path
            elif self.url_path:
                metadata["source"] = self.url_path

            # Add model info
            if "model" in response:
                metadata["model"] = response["model"]

            return [Document(page_content=all_content, metadata=metadata)]

    def load(self) -> List[Document]:
        """Load documents synchronously.

        Returns:
            List of Document objects.
        """
        document_payload = self._prepare_document_payload()
        response = self._make_ocr_request(document_payload, sync=True)
        return self._process_response(response)

    async def aload(self) -> List[Document]:
        """Load documents asynchronously.

        Returns:
            List of Document objects.
        """
        document_payload = self._prepare_document_payload()
        response_coro = self._make_ocr_request(document_payload, sync=False)
        response = await response_coro
        return self._process_response(response)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents (loads all, then yields one at a time).

        Yields:
            Document objects.
        """
        documents = self.load()
        for doc in documents:
            yield doc
