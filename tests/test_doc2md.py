"""Tests for doc2md utility functions."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import doc2md


class TestEncodeImageBytes:
    def test_basic_encoding(self):
        data = b"hello"
        result = doc2md.encode_image_bytes(data, "image/png")
        assert result.startswith("data:image/png;base64,")
        assert "aGVsbG8=" in result  # base64 of "hello"

    def test_custom_mime(self):
        result = doc2md.encode_image_bytes(b"\x00", "image/jpeg")
        assert result.startswith("data:image/jpeg;base64,")


class TestAssembleDocument:
    def test_single_page(self):
        result = doc2md.assemble_document(["# Hello"], "test.pdf")
        assert result == "# Hello"

    def test_multiple_pages(self):
        pages = ["# Page 1", "# Page 2"]
        result = doc2md.assemble_document(pages, "test.pdf")
        assert "<!-- page 1 -->" in result
        assert "<!-- page 2 -->" in result
        assert "\n\n---\n\n" in result

    def test_blank_pages_skipped(self):
        pages = ["# Page 1", "<!-- blank page -->", "# Page 3"]
        result = doc2md.assemble_document(pages, "test.pdf")
        assert "<!-- page 1 -->" in result
        assert "<!-- page 3 -->" in result
        assert "blank page" not in result.split("---")[0]
        # blank page entry itself is filtered
        assert result.count("---") == 1


class TestLoadImageFile:
    def test_png_file(self):
        # 1x1 red PNG
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            f.flush()
            result = doc2md.load_image_file(f.name)
        os.unlink(f.name)
        assert result.startswith("data:image/png;base64,")

    def test_jpeg_mime(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff")
            f.flush()
            result = doc2md.load_image_file(f.name)
        os.unlink(f.name)
        assert result.startswith("data:image/jpeg;base64,")


class TestGetClient:
    def test_missing_key_exits(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove NVIDIA_API_KEY if set
            os.environ.pop("NVIDIA_API_KEY", None)
            with pytest.raises(SystemExit):
                doc2md.get_client(api_key=None)

    def test_explicit_key(self):
        client = doc2md.get_client(api_key="test-key-123")
        assert client.api_key == "test-key-123"


class TestConvertImageRetry:
    @patch("doc2md.time.sleep")
    def test_retries_on_server_error(self, mock_sleep):
        client = MagicMock()

        error = Exception("Server error")
        mock_response = MagicMock()
        mock_response.status_code = 500
        error.response = mock_response

        client.chat.completions.create.side_effect = [
            error,
            error,
            MagicMock(
                __iter__=lambda self: iter([
                    MagicMock(choices=[
                        MagicMock(delta=MagicMock(content="# Hello"))
                    ])
                ])
            ),
        ]

        result = doc2md.convert_image_to_markdown(
            client, "data:image/png;base64,abc"
        )
        assert "Hello" in result
        assert client.chat.completions.create.call_count == 3

    def test_no_retry_on_client_error(self):
        client = MagicMock()

        error = Exception("Bad request")
        mock_response = MagicMock()
        mock_response.status_code = 400
        error.response = mock_response

        client.chat.completions.create.side_effect = error

        with pytest.raises(Exception, match="Bad request"):
            doc2md.convert_image_to_markdown(
                client, "data:image/png;base64,abc"
            )
        assert client.chat.completions.create.call_count == 1


class TestVersion:
    def test_version_defined(self):
        assert hasattr(doc2md, "__version__")
        assert doc2md.__version__
