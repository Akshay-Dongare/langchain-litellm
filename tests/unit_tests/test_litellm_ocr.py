"""Unit tests for LiteLLMOCRLoader."""

import base64
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_litellm.document_loaders import LiteLLMOCRLoader


# Mock OCR response fixture
@pytest.fixture
def mock_ocr_response() -> Dict[str, Any]:
    """Mock response from LiteLLM OCR endpoint."""
    return {
        "pages": [
            {
                "index": 0,
                "markdown": "# Page 1\n\nThis is the first page.",
                "dimensions": {"width": 612, "height": 792}
            },
            {
                "index": 1,
                "markdown": "# Page 2\n\nThis is the second page.",
                "dimensions": {"width": 612, "height": 792}
            }
        ],
        "model": "azure_ai/doc-intelligence/prebuilt-layout",
        "object": "ocr"
    }


class TestLiteLLMOCRLoaderValidation:
    """Test input validation."""

    def test_no_input_source_raises_error(self) -> None:
        """Test that missing input source raises ValueError."""
        with pytest.raises(ValueError, match="Must provide exactly one"):
            LiteLLMOCRLoader()

    def test_multiple_input_sources_raises_error(self) -> None:
        """Test that multiple input sources raises ValueError."""
        with pytest.raises(ValueError, match="Must provide exactly one"):
            LiteLLMOCRLoader(
                file_path="/tmp/test.pdf",
                url_path="https://example.com/doc.pdf"
            )

    def test_invalid_mode_raises_error(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            LiteLLMOCRLoader(
                url_path="https://example.com/doc.pdf",
                mode="invalid"  # type: ignore
            )

    def test_invalid_proxy_url_raises_error(self) -> None:
        """Test that invalid proxy URL raises ValueError."""
        with pytest.raises(ValueError, match="proxy_base_url must start with"):
            LiteLLMOCRLoader(
                proxy_base_url="invalid-url",
                url_path="https://example.com/doc.pdf"
            )


class TestLiteLLMOCRLoaderDocumentPreparation:
    """Test document payload preparation."""

    def test_prepare_url_payload(self) -> None:
        """Test preparing payload for URL input."""
        loader = LiteLLMOCRLoader(url_path="https://example.com/doc.pdf")
        payload = loader._prepare_document_payload()

        assert payload == {
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }

    def test_prepare_base64_payload(self) -> None:
        """Test preparing payload for base64 input."""
        b64_content = "JVBERi0xLjQ="  # Sample base64
        loader = LiteLLMOCRLoader(base64_content=b64_content)
        payload = loader._prepare_document_payload()

        assert payload["type"] == "document_url"
        assert payload["document_url"].startswith("data:application/pdf;base64,")
        assert b64_content in payload["document_url"]

    def test_prepare_base64_with_data_uri(self) -> None:
        """Test preparing payload for base64 with data URI."""
        data_uri = "data:application/pdf;base64,JVBERi0xLjQ="
        loader = LiteLLMOCRLoader(base64_content=data_uri)
        payload = loader._prepare_document_payload()

        assert payload == {
            "type": "document_url",
            "document_url": data_uri
        }

    def test_prepare_bytes_payload(self) -> None:
        """Test preparing payload for bytes input."""
        test_bytes = b"test content"
        loader = LiteLLMOCRLoader(bytes_content=test_bytes)
        payload = loader._prepare_document_payload()

        expected_b64 = base64.b64encode(test_bytes).decode("utf-8")
        assert payload["type"] == "document_url"
        assert expected_b64 in payload["document_url"]
        assert payload["document_url"].startswith("data:application/pdf;base64,")

    def test_prepare_file_payload(self, tmp_path: Path) -> None:
        """Test preparing payload for file input."""
        # Create a temporary PDF file
        test_file = tmp_path / "test.pdf"
        test_content = b"PDF content here"
        test_file.write_bytes(test_content)

        loader = LiteLLMOCRLoader(file_path=str(test_file))
        payload = loader._prepare_document_payload()

        expected_b64 = base64.b64encode(test_content).decode("utf-8")
        assert payload["type"] == "document_url"
        assert expected_b64 in payload["document_url"]
        assert "application/pdf" in payload["document_url"]

    def test_prepare_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        loader = LiteLLMOCRLoader(file_path="/nonexistent/file.pdf")

        with pytest.raises(FileNotFoundError):
            loader._prepare_document_payload()


class TestLiteLLMOCRLoaderResponseProcessing:
    """Test response processing."""

    def test_process_response_page_mode(self, mock_ocr_response: Dict[str, Any]) -> None:
        """Test processing response in page mode."""
        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            mode="page"
        )
        documents = loader._process_response(mock_ocr_response)

        assert len(documents) == 2

        # Check first page
        assert documents[0].page_content == "# Page 1\n\nThis is the first page."
        assert documents[0].metadata["page"] == 0
        assert documents[0].metadata["width"] == 612
        assert documents[0].metadata["height"] == 792
        assert documents[0].metadata["source"] == "https://example.com/doc.pdf"
        assert documents[0].metadata["model"] == "azure_ai/doc-intelligence/prebuilt-layout"

        # Check second page
        assert documents[1].page_content == "# Page 2\n\nThis is the second page."
        assert documents[1].metadata["page"] == 1

    def test_process_response_single_mode(self, mock_ocr_response: Dict[str, Any]) -> None:
        """Test processing response in single mode."""
        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            mode="single"
        )
        documents = loader._process_response(mock_ocr_response)

        assert len(documents) == 1

        expected_content = (
            "# Page 1\n\nThis is the first page.\n\n"
            "# Page 2\n\nThis is the second page."
        )
        assert documents[0].page_content == expected_content
        assert documents[0].metadata["total_pages"] == 2
        assert documents[0].metadata["source"] == "https://example.com/doc.pdf"
        assert documents[0].metadata["model"] == "azure_ai/doc-intelligence/prebuilt-layout"

    def test_process_response_missing_pages_field(self) -> None:
        """Test that missing pages field raises ValueError."""
        loader = LiteLLMOCRLoader(url_path="https://example.com/doc.pdf")

        with pytest.raises(ValueError, match="missing 'pages' field"):
            loader._process_response({"object": "ocr"})


class TestLiteLLMOCRLoaderLoad:
    """Test synchronous loading."""

    @patch("httpx.Client")
    def test_load_success(
        self,
        mock_client_class: MagicMock,
        mock_ocr_response: Dict[str, Any]
    ) -> None:
        """Test successful synchronous load."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = mock_ocr_response
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Load documents
        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            mode="page"
        )
        documents = loader.load()

        # Verify results
        assert len(documents) == 2
        assert documents[0].page_content == "# Page 1\n\nThis is the first page."

        # Verify HTTP call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:4000/ocr"
        assert call_args[1]["json"]["model"] == "azure-document"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"

    @patch("httpx.Client")
    def test_load_with_auth(
        self,
        mock_client_class: MagicMock,
        mock_ocr_response: Dict[str, Any]
    ) -> None:
        """Test load with authentication."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = mock_ocr_response

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Load documents
        loader = LiteLLMOCRLoader(
            proxy_base_url="https://my-proxy.com",
            api_key="test-key",
            url_path="https://example.com/doc.pdf",
            model="custom-model"
        )
        loader.load()

        # Verify auth header
        call_args = mock_client.post.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
        assert call_args[1]["json"]["model"] == "custom-model"

    @patch("httpx.Client")
    def test_load_http_error(self, mock_client_class: MagicMock) -> None:
        """Test load with HTTP error."""
        import httpx

        # Setup mock to raise HTTPStatusError
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=mock_response
        )
        mock_client_class.return_value = mock_client

        # Load should raise RuntimeError
        loader = LiteLLMOCRLoader(url_path="https://example.com/doc.pdf")

        with pytest.raises(RuntimeError, match="HTTP error from LiteLLM proxy"):
            loader.load()

    @patch("httpx.Client")
    def test_load_connection_error(self, mock_client_class: MagicMock) -> None:
        """Test load with connection error."""
        import httpx

        # Setup mock to raise RequestError
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.RequestError("Connection failed")
        mock_client_class.return_value = mock_client

        # Load should raise RuntimeError
        loader = LiteLLMOCRLoader(url_path="https://example.com/doc.pdf")

        with pytest.raises(RuntimeError, match="Failed to connect"):
            loader.load()


class TestLiteLLMOCRLoaderAsyncLoad:
    """Test asynchronous loading."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_aload_success(
        self,
        mock_client_class: MagicMock,
        mock_ocr_response: Dict[str, Any]
    ) -> None:
        """Test successful asynchronous load."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = mock_ocr_response
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        # Load documents
        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            mode="single"
        )
        documents = await loader.aload()

        # Verify results
        assert len(documents) == 1
        assert "# Page 1" in documents[0].page_content
        assert "# Page 2" in documents[0].page_content

        # Verify HTTP call
        mock_client.post.assert_called_once()


class TestLiteLLMOCRLoaderLazyLoad:
    """Test lazy loading."""

    @patch("httpx.Client")
    def test_lazy_load(
        self,
        mock_client_class: MagicMock,
        mock_ocr_response: Dict[str, Any]
    ) -> None:
        """Test lazy loading yields documents."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = mock_ocr_response

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Lazy load documents
        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            mode="page"
        )
        documents = list(loader.lazy_load())

        # Verify results
        assert len(documents) == 2
        assert documents[0].page_content == "# Page 1\n\nThis is the first page."
        assert documents[1].page_content == "# Page 2\n\nThis is the second page."


class TestLiteLLMOCRLoaderTimeoutAndRetry:
    """Test configurable timeout and retry logic."""

    def test_default_timeout_and_retries(self) -> None:
        """Test default timeout and max_retries values."""
        loader = LiteLLMOCRLoader(url_path="https://example.com/doc.pdf")
        assert loader.timeout == 300.0
        assert loader.max_retries == 3

    def test_custom_timeout_and_retries(self) -> None:
        """Test custom timeout and max_retries values."""
        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            timeout=60.0,
            max_retries=5,
        )
        assert loader.timeout == 60.0
        assert loader.max_retries == 5

    @patch("httpx.Client")
    def test_timeout_passed_to_client(
        self,
        mock_client_class: MagicMock,
        mock_ocr_response: Dict[str, Any],
    ) -> None:
        """Test that custom timeout is passed to httpx.Client."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_ocr_response

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            timeout=60.0,
        )
        loader.load()

        mock_client_class.assert_called_with(timeout=60.0)

    @patch("time.sleep")
    @patch("httpx.Client")
    def test_retry_on_request_error(
        self,
        mock_client_class: MagicMock,
        mock_sleep: MagicMock,
        mock_ocr_response: Dict[str, Any],
    ) -> None:
        """Test that transient RequestError triggers retries."""
        import httpx

        mock_success_response = MagicMock()
        mock_success_response.json.return_value = mock_ocr_response

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        # Fail twice, then succeed on third attempt
        mock_client.post.side_effect = [
            httpx.RequestError("Connection reset"),
            httpx.RequestError("Connection reset"),
            mock_success_response,
        ]
        mock_client_class.return_value = mock_client

        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            max_retries=3,
        )
        documents = loader.load()

        assert len(documents) == 1
        assert mock_client.post.call_count == 3
        # Exponential backoff: sleep(1), sleep(2)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("time.sleep")
    @patch("httpx.Client")
    def test_retry_on_http_status_error(
        self,
        mock_client_class: MagicMock,
        mock_sleep: MagicMock,
        mock_ocr_response: Dict[str, Any],
    ) -> None:
        """Test that HTTPStatusError triggers retries."""
        import httpx

        mock_error_response = MagicMock()
        mock_error_response.status_code = 503
        mock_error_response.text = "Service Unavailable"

        mock_success_response = MagicMock()
        mock_success_response.json.return_value = mock_ocr_response

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = [
            httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_error_response
            ),
            mock_success_response,
        ]
        mock_client_class.return_value = mock_client

        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            max_retries=3,
        )
        documents = loader.load()

        assert len(documents) == 1
        assert mock_client.post.call_count == 2
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(1)

    @patch("time.sleep")
    @patch("httpx.Client")
    def test_retries_exhausted_raises_error(
        self,
        mock_client_class: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Test that exhausted retries raise the last error."""
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.RequestError("Connection failed")
        mock_client_class.return_value = mock_client

        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            max_retries=3,
        )

        with pytest.raises(RuntimeError, match="Failed to connect"):
            loader.load()

        assert mock_client.post.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("httpx.Client")
    def test_max_retries_one_no_retry(
        self,
        mock_client_class: MagicMock,
    ) -> None:
        """Test that max_retries=1 means no retries on failure."""
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.RequestError("Connection failed")
        mock_client_class.return_value = mock_client

        loader = LiteLLMOCRLoader(
            url_path="https://example.com/doc.pdf",
            max_retries=1,
        )

        with pytest.raises(RuntimeError, match="Failed to connect"):
            loader.load()

        assert mock_client.post.call_count == 1
