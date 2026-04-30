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

    def test_version_bumped(self):
        assert doc2md.__version__ == "0.2.0"


# ---------------------------------------------------------------------------
# Feature 5: Page range parsing
# ---------------------------------------------------------------------------

class TestParsePageRanges:
    def test_single_page(self):
        result = doc2md.parse_page_ranges("5", 20)
        assert result == [4]  # 0-based

    def test_range(self):
        result = doc2md.parse_page_ranges("1-5", 20)
        assert result == [0, 1, 2, 3, 4]

    def test_open_start(self):
        result = doc2md.parse_page_ranges("-3", 10)
        assert result == [0, 1, 2]

    def test_open_end(self):
        result = doc2md.parse_page_ranges("8-", 10)
        assert result == [7, 8, 9]

    def test_combined(self):
        result = doc2md.parse_page_ranges("1-3,7,10-12", 15)
        assert result == [0, 1, 2, 6, 9, 10, 11]

    def test_clamps_to_total(self):
        result = doc2md.parse_page_ranges("1-100", 5)
        assert result == [0, 1, 2, 3, 4]

    def test_out_of_range_single(self):
        result = doc2md.parse_page_ranges("99", 5)
        assert result == []

    def test_deduplicates(self):
        result = doc2md.parse_page_ranges("1-3,2-4", 10)
        assert result == [0, 1, 2, 3]

    def test_empty_parts_ignored(self):
        result = doc2md.parse_page_ranges("1,,3", 10)
        assert result == [0, 2]

    def test_reversed_range_ignored(self):
        result = doc2md.parse_page_ranges("5-2", 10)
        assert result == []


# ---------------------------------------------------------------------------
# Feature 2: Markdown validation
# ---------------------------------------------------------------------------

class TestValidateMarkdown:
    def test_valid_markdown(self):
        md = "# Title\n\nSome text.\n\n## Section\n\n- item 1\n- item 2\n"
        warnings = doc2md.validate_markdown(md)
        assert warnings == []

    def test_unbalanced_code_fences(self):
        md = "# Title\n\n```python\nprint('hello')\n"
        warnings = doc2md.validate_markdown(md)
        assert any("code fence" in w.lower() for w in warnings)

    def test_balanced_code_fences(self):
        md = "```python\nprint('hello')\n```\n"
        warnings = doc2md.validate_markdown(md)
        fence_warnings = [w for w in warnings if "code fence" in w.lower()]
        assert fence_warnings == []

    def test_inconsistent_table_columns(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n"
        warnings = doc2md.validate_markdown(md)
        assert any("column" in w.lower() for w in warnings)

    def test_consistent_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        warnings = doc2md.validate_markdown(md)
        table_warnings = [w for w in warnings if "column" in w.lower()]
        assert table_warnings == []

    def test_heading_hierarchy_skip(self):
        md = "# Title\n\n### Skipped h2\n"
        warnings = doc2md.validate_markdown(md)
        assert any("heading hierarchy" in w.lower() for w in warnings)

    def test_heading_hierarchy_ok(self):
        md = "# Title\n\n## Section\n\n### Sub\n"
        warnings = doc2md.validate_markdown(md)
        heading_warnings = [w for w in warnings if "heading" in w.lower()]
        assert heading_warnings == []

    def test_very_short_output(self):
        md = "hello"
        warnings = doc2md.validate_markdown(md)
        assert any("short" in w.lower() for w in warnings)

    def test_blank_page_not_flagged(self):
        md = "<!-- blank page -->"
        warnings = doc2md.validate_markdown(md)
        short_warnings = [w for w in warnings if "short" in w.lower()]
        assert short_warnings == []


# ---------------------------------------------------------------------------
# Feature 3: Confidence scoring
# ---------------------------------------------------------------------------

class TestExtractConfidence:
    def test_extracts_score(self):
        md = "# Title\n\nSome content.\n\n<!-- confidence: 0.92 -->"
        cleaned, score = doc2md.extract_confidence(md)
        assert score == pytest.approx(0.92)
        assert "confidence" not in cleaned
        assert "# Title" in cleaned

    def test_no_confidence(self):
        md = "# Title\n\nSome content."
        cleaned, score = doc2md.extract_confidence(md)
        assert score is None
        assert cleaned == md

    def test_clamps_above_one(self):
        md = "text\n<!-- confidence: 1.50 -->"
        _, score = doc2md.extract_confidence(md)
        assert score == 1.0

    def test_negative_not_matched(self):
        # Negative values don't match the pattern (model won't produce them)
        md = "text\n<!-- confidence: -0.5 -->"
        _, score = doc2md.extract_confidence(md)
        assert score is None

    def test_whitespace_variations(self):
        md = "text\n<!--confidence:0.85-->"
        _, score = doc2md.extract_confidence(md)
        assert score == pytest.approx(0.85)


class TestConfidenceLabel:
    def test_high(self):
        assert doc2md.confidence_label(0.95) == "high"

    def test_medium(self):
        assert doc2md.confidence_label(0.75) == "medium"

    def test_low(self):
        assert doc2md.confidence_label(0.55) == "low"

    def test_very_low(self):
        assert doc2md.confidence_label(0.30) == "very low"


# ---------------------------------------------------------------------------
# Feature 1: Office format support
# ---------------------------------------------------------------------------

class TestSupportedExtensions:
    def test_docx_supported(self):
        assert ".docx" in doc2md.SUPPORTED_EXTENSIONS

    def test_pptx_supported(self):
        assert ".pptx" in doc2md.SUPPORTED_EXTENSIONS

    def test_office_extensions(self):
        assert ".docx" in doc2md.OFFICE_EXTENSIONS
        assert ".pptx" in doc2md.OFFICE_EXTENSIONS


class TestFindLibreOffice:
    @patch("shutil.which", return_value="/usr/bin/libreoffice")
    def test_finds_libreoffice(self, mock_which):
        result = doc2md._find_libreoffice()
        assert result == "/usr/bin/libreoffice"

    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    @patch("doc2md.sys")
    def test_returns_none_when_missing(self, mock_sys, mock_isfile, mock_which):
        mock_sys.platform = "linux"
        result = doc2md._find_libreoffice()
        assert result is None
