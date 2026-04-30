#!/usr/bin/env python3
"""
doc2md — Convert documents to LLM-ready Markdown using NVIDIA Nemotron 3 Nano Omni.

Accepts PDFs, images (PNG, JPG, TIFF, BMP, WebP), and Office documents (DOCX, PPTX)
and sends each page/image/slide to the Nemotron vision model to produce clean Markdown
output optimised for LLM ingestion.

Usage:
    python doc2md.py report.pdf
    python doc2md.py slide.png diagram.jpg
    python doc2md.py presentation.pptx --pages 1-5
    python doc2md.py *.pdf --output-dir ./markdown
    python doc2md.py invoice.pdf --single-file --validate
"""

import argparse
import base64
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator

from openai import OpenAI

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Optional rich import (soft dependency for progress bars)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ---------------------------------------------------------------------------
# Terminal colours (used when rich is not available)
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# NIM client
# ---------------------------------------------------------------------------
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-omni-reasoning-30b-a3b"

SYSTEM_PROMPT = """\
You are a document-to-Markdown converter. Your job is to look at a document \
page or image and produce clean, accurate Markdown that faithfully represents \
all the content.

Rules:
1. Preserve the document's structure: headings, sub-headings, lists, tables.
2. Use proper Markdown syntax: # for headings, - or * for bullets, | for tables, \
   ``` for code blocks, > for blockquotes.
3. For tables, produce well-formed Markdown tables with header rows and alignment.
4. For diagrams, charts, or figures, describe them in a blockquote prefixed with \
   [Figure]: e.g. > [Figure]: Bar chart showing quarterly revenue growth from Q1-Q4.
5. For mathematical formulas, use LaTeX notation inside $ or $$ delimiters.
6. Preserve emphasis: bold, italic, underline where visible.
7. Do NOT add commentary, opinions, or summaries. Only output the content you see.
8. Do NOT wrap the output in a markdown code fence. Output raw Markdown directly.
9. If the page is blank or contains only a page number, output: <!-- blank page -->
10. For multi-column layouts, merge columns into a single linear flow, top-to-bottom, \
    left-to-right.
"""

CONFIDENCE_PROMPT_SUFFIX = """
11. After your Markdown output, on a new line, output exactly one confidence line:
    <!-- confidence: X.XX -->
    where X.XX is a number from 0.00 to 1.00 indicating how confident you are that \
the output accurately and completely represents the source content. Consider: text \
legibility, table complexity, image quality, handwriting presence, and layout difficulty \
when scoring.
"""

# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
OFFICE_EXTENSIONS = {".docx", ".pptx"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS | OFFICE_EXTENSIONS


def get_client(api_key: str | None = None) -> OpenAI:
    """Return an OpenAI client configured for NVIDIA NIM."""
    key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        print(f"{BOLD}Error:{RESET} Set NVIDIA_API_KEY env var or pass --api-key")
        sys.exit(1)
    return OpenAI(
        base_url=BASE_URL,
        api_key=key,
        default_headers={"NVCF-POLL-SECONDS": "1800"},
    )


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_bytes(img_bytes: bytes, mime: str = "image/png") -> str:
    """Base64-encode image bytes into a data URL."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_image_file(path: str) -> str:
    """Load an image file and return a base64 data URL."""
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/png")
    with open(path, "rb") as f:
        return encode_image_bytes(f.read(), mime)


# ---------------------------------------------------------------------------
# Office document conversion (DOCX / PPTX → PDF via LibreOffice)
# ---------------------------------------------------------------------------

def _find_libreoffice() -> str | None:
    """Find the LibreOffice binary on the system."""
    candidates = ["libreoffice", "soffice"]
    if sys.platform == "darwin":
        candidates.append("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    if sys.platform == "darwin":
        mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        if os.path.isfile(mac_path):
            return mac_path
    return None


def office_to_pdf(office_path: str) -> str:
    """Convert a DOCX or PPTX file to PDF using LibreOffice headless.

    Returns the path to the generated PDF in a temporary directory.
    The caller is responsible for cleaning up the temp directory.
    """
    lo_bin = _find_libreoffice()
    if lo_bin is None:
        ext = os.path.splitext(office_path)[1].lower()
        print(f"{BOLD}Error:{RESET} LibreOffice is required for {ext} support.")
        print("  Install it:")
        if sys.platform == "darwin":
            print("    brew install --cask libreoffice")
        elif sys.platform == "linux":
            print("    sudo apt install libreoffice-core  # or: sudo dnf install libreoffice-core")
        else:
            print("    https://www.libreoffice.org/download/")
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="doc2md_")
    cmd = [
        lo_bin,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", tmp_dir,
        os.path.abspath(office_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"{BOLD}Error:{RESET} LibreOffice conversion failed: {exc.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"{BOLD}Error:{RESET} LibreOffice binary not found at: {lo_bin}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(office_path))[0]
    pdf_path = os.path.join(tmp_dir, f"{base_name}.pdf")

    if not os.path.isfile(pdf_path):
        print(f"{BOLD}Error:{RESET} LibreOffice did not produce expected PDF: {pdf_path}")
        sys.exit(1)

    return pdf_path


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

def pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF without rendering them."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"{BOLD}Error:{RESET} pymupdf is required for PDF support.")
        print("  Install it: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def pdf_to_images(
    pdf_path: str,
    dpi: int = 200,
    page_indices: list[int] | None = None,
) -> Iterator[tuple[int, bytes]]:
    """Yield (page_number, png_bytes) for each selected page of a PDF.

    page_indices: 0-based page indices to render. None means all pages.
    page_number in the yielded tuple is 1-based.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"{BOLD}Error:{RESET} pymupdf is required for PDF support.")
        print("  Install it: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    indices = page_indices if page_indices is not None else range(len(doc))
    for idx in indices:
        page = doc[idx]
        pix = page.get_pixmap(matrix=mat)
        yield (idx + 1, pix.tobytes("png"))

    doc.close()


# ---------------------------------------------------------------------------
# Page range parsing
# ---------------------------------------------------------------------------

def parse_page_ranges(spec: str, total_pages: int) -> list[int]:
    """Parse a page range spec like '1-5,10,12-15' into 0-based page indices.

    Supports:
      - Single pages: '5'
      - Ranges: '1-5'
      - Open-ended ranges: '-5' (pages 1-5), '10-' (page 10 to end)
      - Comma-separated combinations: '1-3,7,10-12'

    Returns sorted, deduplicated list of 0-based indices.
    """
    indices: set[int] = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Could be "1-5", "-5", or "10-"
            pieces = part.split("-", 1)
            start_str, end_str = pieces[0].strip(), pieces[1].strip()

            start = int(start_str) if start_str else 1
            end = int(end_str) if end_str else total_pages

            if start < 1:
                start = 1
            if end > total_pages:
                end = total_pages
            if start > end:
                continue

            for p in range(start, end + 1):
                indices.add(p - 1)  # convert to 0-based
        else:
            page = int(part)
            if 1 <= page <= total_pages:
                indices.add(page - 1)

    return sorted(indices)


# ---------------------------------------------------------------------------
# Markdown validation
# ---------------------------------------------------------------------------

def validate_markdown(md: str) -> list[str]:
    """Check generated Markdown for common structural issues.

    Returns a list of warning strings. Empty list means no issues found.
    """
    warnings: list[str] = []
    lines = md.split("\n")

    # Check for unbalanced code fences
    fence_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            fence_count += 1
    if fence_count % 2 != 0:
        warnings.append("Unbalanced code fences (odd number of ``` markers)")

    # Check for malformed tables (rows with inconsistent column counts)
    table_rows: list[tuple[int, int]] = []  # (line_number, col_count)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cols = stripped.count("|") - 1  # pipes minus outer ones
            table_rows.append((i, cols))
        else:
            # End of table block — check consistency
            if len(table_rows) >= 2:
                col_counts = {c for _, c in table_rows}
                if len(col_counts) > 1:
                    first_line = table_rows[0][0]
                    warnings.append(
                        f"Table near line {first_line} has inconsistent column counts: "
                        f"{sorted(col_counts)}"
                    )
            table_rows = []
    # Check trailing table block
    if len(table_rows) >= 2:
        col_counts = {c for _, c in table_rows}
        if len(col_counts) > 1:
            first_line = table_rows[0][0]
            warnings.append(
                f"Table near line {first_line} has inconsistent column counts: "
                f"{sorted(col_counts)}"
            )

    # Check heading hierarchy (no jumps like # → ###)
    prev_level = 0
    for i, line in enumerate(lines, 1):
        match = re.match(r"^(#{1,6})\s", line)
        if match:
            level = len(match.group(1))
            if prev_level > 0 and level > prev_level + 1:
                warnings.append(
                    f"Heading hierarchy skip at line {i}: "
                    f"jumped from h{prev_level} to h{level}"
                )
            prev_level = level

    # Check for unclosed inline formatting (basic check)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip separator rows and code
        if stripped.startswith("|") or stripped.startswith("```"):
            continue
        # Count unescaped bold markers
        bold_markers = len(re.findall(r"(?<!\*)\*\*(?!\*)", stripped))
        if bold_markers % 2 != 0:
            warnings.append(f"Possible unclosed bold (**) at line {i}")

    # Check for very short output (possibly incomplete conversion)
    non_blank = [line for line in lines if line.strip() and not line.strip().startswith("<!--")]
    if len(non_blank) < 2 and md.strip() != "<!-- blank page -->":
        warnings.append("Very short output — conversion may be incomplete")

    return warnings


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

CONFIDENCE_PATTERN = re.compile(r"<!--\s*confidence:\s*([\d.]+)\s*-->")


def extract_confidence(md: str) -> tuple[str, float | None]:
    """Extract confidence score from the end of Markdown output.

    Returns (cleaned_markdown, confidence_score).
    The confidence comment is stripped from the output.
    """
    match = CONFIDENCE_PATTERN.search(md)
    if match:
        try:
            score = float(match.group(1))
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = None
        cleaned = md[:match.start()].rstrip() + md[match.end():]
        return cleaned.strip(), score
    return md, None


def confidence_label(score: float) -> str:
    """Return a human-readable label for a confidence score."""
    if score >= 0.90:
        return "high"
    elif score >= 0.70:
        return "medium"
    elif score >= 0.50:
        return "low"
    else:
        return "very low"


def confidence_color(score: float) -> str:
    """Return an ANSI colour code for a confidence score."""
    if score >= 0.90:
        return GREEN
    elif score >= 0.70:
        return YELLOW
    else:
        return RED


# ---------------------------------------------------------------------------
# Conversion via Nemotron
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubled each attempt


def convert_image_to_markdown(
    client: OpenAI,
    image_data_url: str,
    *,
    page_label: str = "",
    enable_thinking: bool = True,
    enable_confidence: bool = False,
    verbose: bool = False,
) -> str:
    """Send a single image to Nemotron and return the Markdown output."""
    prompt = "Convert this document page to Markdown. Follow the system instructions exactly."
    if page_label:
        prompt = f"This is {page_label}. " + prompt

    system = SYSTEM_PROMPT
    if enable_confidence:
        system += CONFIDENCE_PROMPT_SUFFIX

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=16384,
                temperature=0.6,
                top_p=0.95,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )

            content = ""
            for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                delta = choice.delta
                if delta.content:
                    content += delta.content
                    if verbose:
                        print(delta.content, end="", flush=True)

            if verbose:
                print()
            return content.strip()

        except Exception as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status and status < 500 and status != 429:
                raise  # non-retryable client error
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            label = page_label or "image"
            print(
                f"\n  {BOLD}Retry {attempt}/{MAX_RETRIES}{RESET} for {label} "
                f"({exc}) — waiting {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} retries: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_pdf(
    client: OpenAI,
    pdf_path: str,
    *,
    dpi: int = 200,
    enable_thinking: bool = True,
    enable_confidence: bool = False,
    page_indices: list[int] | None = None,
    verbose: bool = False,
    progress=None,
    progress_task=None,
) -> list[tuple[str, float | None]]:
    """Convert a PDF to a list of (markdown, confidence) tuples, one per page."""
    filename = os.path.basename(pdf_path)
    total = pdf_page_count(pdf_path)

    pages_to_render = len(page_indices) if page_indices is not None else total
    page_desc = f"{pages_to_render} of {total}" if page_indices else str(total)

    if RICH_AVAILABLE and not verbose:
        console.print(f"  [cyan bold]\\[PDF][/] {filename} — {page_desc} page(s) at {dpi} DPI")
    else:
        print(f"\n{CYAN}{BOLD}[PDF]{RESET} {filename}")
        print(f"  {page_desc} page(s) at {dpi} DPI\n")

    results = []
    for page_num, img_bytes in pdf_to_images(pdf_path, dpi=dpi, page_indices=page_indices):
        page_label = f"page {page_num} of {total}"

        if verbose:
            print(f"  {BOLD}--- Page {page_num}/{total} ---{RESET}")

        data_url = encode_image_bytes(img_bytes, "image/png")
        md = convert_image_to_markdown(
            client,
            data_url,
            page_label=page_label,
            enable_thinking=enable_thinking,
            enable_confidence=enable_confidence,
            verbose=verbose,
        )

        confidence = None
        if enable_confidence:
            md, confidence = extract_confidence(md)

        results.append((md, confidence))

        if progress is not None and progress_task is not None:
            progress.advance(progress_task)

        if verbose:
            if confidence is not None:
                cc = confidence_color(confidence)
                print(
                    f"  Confidence: {cc}{confidence:.2f} "
                    f"({confidence_label(confidence)}){RESET}"
                )
            print()

    return results


def process_image(
    client: OpenAI,
    image_path: str,
    *,
    enable_thinking: bool = True,
    enable_confidence: bool = False,
    verbose: bool = False,
    progress=None,
    progress_task=None,
) -> tuple[str, float | None]:
    """Convert a single image file to Markdown. Returns (markdown, confidence)."""
    filename = os.path.basename(image_path)

    if RICH_AVAILABLE and not verbose:
        console.print(f"  [cyan bold]\\[Image][/] {filename}")
    else:
        print(f"\n{CYAN}{BOLD}[Image]{RESET} {filename}")

    data_url = load_image_file(image_path)
    md = convert_image_to_markdown(
        client,
        data_url,
        page_label=filename,
        enable_thinking=enable_thinking,
        enable_confidence=enable_confidence,
        verbose=verbose,
    )

    confidence = None
    if enable_confidence:
        md, confidence = extract_confidence(md)

    if progress is not None and progress_task is not None:
        progress.advance(progress_task)

    if verbose and confidence is not None:
        cc = confidence_color(confidence)
        print(f"  Confidence: {cc}{confidence:.2f} ({confidence_label(confidence)}){RESET}")

    return md, confidence


def process_office(
    client: OpenAI,
    office_path: str,
    *,
    dpi: int = 200,
    enable_thinking: bool = True,
    enable_confidence: bool = False,
    page_indices: list[int] | None = None,
    verbose: bool = False,
    progress=None,
    progress_task=None,
) -> list[tuple[str, float | None]]:
    """Convert a DOCX or PPTX file by first converting to PDF via LibreOffice."""
    filename = os.path.basename(office_path)
    ext = os.path.splitext(office_path)[1].lower()
    label = "DOCX" if ext == ".docx" else "PPTX"

    if RICH_AVAILABLE and not verbose:
        console.print(f"  [cyan bold]\\[{label}][/] {filename} — converting to PDF via LibreOffice")
    else:
        print(f"\n{CYAN}{BOLD}[{label}]{RESET} {filename}")
        print("  Converting to PDF via LibreOffice...")

    pdf_path = office_to_pdf(office_path)
    try:
        results = process_pdf(
            client,
            pdf_path,
            dpi=dpi,
            enable_thinking=enable_thinking,
            enable_confidence=enable_confidence,
            page_indices=page_indices,
            verbose=verbose,
            progress=progress,
            progress_task=progress_task,
        )
    finally:
        # Clean up temporary PDF
        tmp_dir = os.path.dirname(pdf_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def assemble_document(pages: list[str], source_name: str) -> str:
    """Join page markdowns into a single document with page separators."""
    if len(pages) == 1:
        return pages[0]

    parts = []
    for i, page_md in enumerate(pages, 1):
        if page_md.strip() == "<!-- blank page -->":
            continue
        parts.append(f"<!-- page {i} -->\n\n{page_md}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Progress display helpers
# ---------------------------------------------------------------------------

def _count_total_pages(
    input_files: list[str],
    page_indices: list[int] | None,
) -> int:
    """Count total pages/images to process for the progress bar."""
    total = 0
    for fpath in input_files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext in PDF_EXTENSIONS or ext in OFFICE_EXTENSIONS:
            if ext in OFFICE_EXTENSIONS:
                # We don't know the page count without converting first;
                # estimate 1 and update later
                total += len(page_indices) if page_indices else 10
            else:
                count = pdf_page_count(fpath)
                if page_indices is not None:
                    total += min(len(page_indices), count)
                else:
                    total += count
        else:
            total += 1
    return total


def _print_banner(file_count: int, dpi: int, thinking: bool, confidence: bool, pages_spec: str):
    """Print the startup banner."""
    if RICH_AVAILABLE:
        console.print()
        console.rule("[bold]doc2md — Document to Markdown Converter[/]", style="bright_blue")
        console.print("  [dim]Powered by NVIDIA Nemotron 3 Nano Omni[/]")
        console.print()
        console.print(f"  Files to process: [bold]{file_count}[/]")
        console.print(f"  DPI (for PDFs):   [bold]{dpi}[/]")
        console.print(f"  Reasoning:        [bold]{'off' if not thinking else 'on'}[/]")
        console.print(f"  Confidence:       [bold]{'on' if confidence else 'off'}[/]")
        if pages_spec:
            console.print(f"  Page selection:   [bold]{pages_spec}[/]")
        console.print()
    else:
        print(f"""
{BOLD}{'='*60}
  doc2md — Document to Markdown Converter
  Powered by NVIDIA Nemotron 3 Nano Omni
{'='*60}{RESET}

  Files to process: {file_count}
  DPI (for PDFs):   {dpi}
  Reasoning:        {'off' if not thinking else 'on'}
  Confidence:       {'on' if confidence else 'off'}
  {'Page selection:   ' + pages_spec if pages_spec else ''}
""")


def _print_summary(
    results: list[tuple[str, list[str], list[float | None]]],
    elapsed: float,
    validation_results: dict[str, list[str]] | None,
):
    """Print the completion summary."""
    total_pages = sum(len(pages) for _, pages, _ in results)
    all_confidences = [
        c for _, _, confs in results for c in confs if c is not None
    ]

    if RICH_AVAILABLE:
        console.print()
        console.rule("[bold green]CONVERSION COMPLETE[/]", style="green")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="bold")
        table.add_row("Files processed", str(len(results)))
        table.add_row("Total pages", str(total_pages))
        table.add_row("Time elapsed", f"{elapsed:.1f}s")
        if all_confidences:
            avg = sum(all_confidences) / len(all_confidences)
            table.add_row("Avg confidence", f"{avg:.2f} ({confidence_label(avg)})")
        console.print(table)

        # Per-file confidence breakdown
        if all_confidences:
            console.print()
            conf_table = Table(title="Confidence Scores", show_lines=False)
            conf_table.add_column("File", style="cyan")
            conf_table.add_column("Page", justify="right")
            conf_table.add_column("Score", justify="right")
            conf_table.add_column("Rating")

            for fpath, _, confs in results:
                fname = os.path.basename(fpath)
                for i, c in enumerate(confs, 1):
                    if c is not None:
                        rating = confidence_label(c)
                        color = "green" if c >= 0.90 else "yellow" if c >= 0.70 else "red"
                        conf_table.add_row(
                            fname if i == 1 else "",
                            str(i),
                            f"{c:.2f}",
                            f"[{color}]{rating}[/]",
                        )
            console.print(conf_table)

        # Validation warnings
        if validation_results:
            has_warnings = any(w for w in validation_results.values())
            if has_warnings:
                console.print()
                console.rule("[bold yellow]Validation Warnings[/]", style="yellow")
                for fpath, warnings in validation_results.items():
                    if warnings:
                        fname = os.path.basename(fpath)
                        console.print(f"\n  [bold]{fname}:[/]")
                        for w in warnings:
                            console.print(f"    [yellow]⚠ {w}[/]")
            else:
                console.print("\n  [green]✓ All outputs passed validation[/]")

        console.print()

    else:
        print(f"""
{BOLD}{'='*60}
  CONVERSION COMPLETE
{'='*60}{RESET}
  Files processed : {len(results)}
  Total pages     : {total_pages}
  Time elapsed    : {elapsed:.1f}s""")

        if all_confidences:
            avg = sum(all_confidences) / len(all_confidences)
            cc = confidence_color(avg)
            print(f"  Avg confidence  : {cc}{avg:.2f} ({confidence_label(avg)}){RESET}")

            print(f"\n  {BOLD}Confidence per page:{RESET}")
            for fpath, _, confs in results:
                fname = os.path.basename(fpath)
                for i, c in enumerate(confs, 1):
                    if c is not None:
                        cc = confidence_color(c)
                        print(f"    {fname} p{i}: {cc}{c:.2f} ({confidence_label(c)}){RESET}")

        if validation_results:
            has_warnings = any(w for w in validation_results.values())
            if has_warnings:
                print(f"\n  {BOLD}{YELLOW}Validation Warnings:{RESET}")
                for fpath, warnings in validation_results.items():
                    if warnings:
                        fname = os.path.basename(fpath)
                        print(f"\n    {BOLD}{fname}:{RESET}")
                        for w in warnings:
                            print(f"      {YELLOW}⚠ {w}{RESET}")
            else:
                print(f"\n  {GREEN}✓ All outputs passed validation{RESET}")

        print(f"{BOLD}{'='*60}{RESET}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="doc2md — Convert documents to LLM-ready Markdown using Nemotron 3 Nano Omni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python doc2md.py report.pdf
  python doc2md.py slide.png diagram.jpg
  python doc2md.py presentation.pptx
  python doc2md.py report.pdf --pages 1-5,10
  python doc2md.py *.pdf --output-dir ./markdown
  python doc2md.py report.pdf --dpi 300 --confidence
  python doc2md.py docs/ --single-file --output combined.md --validate
        """,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF, image, DOCX, PPTX files, or directories to convert",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for output Markdown files (default: output/)",
    )
    parser.add_argument(
        "--single-file", "-s",
        action="store_true",
        help="Combine all inputs into a single Markdown file",
    )
    parser.add_argument(
        "--output", "-O",
        default=None,
        help="Output file path (used with --single-file)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF page rendering (default: 200)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API key (or set NVIDIA_API_KEY env var)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable reasoning mode (faster but less accurate)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Scan directories recursively for supported files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have a corresponding .md output",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of files to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--pages", "-p",
        default=None,
        help="Page range to convert, e.g. '1-5', '1,3,5-10', '-5', '10-' (PDFs/Office only)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated Markdown for structural issues",
    )
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Ask the model to score its conversion confidence (0.00-1.00) per page",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show streaming model output (default: show progress bar)",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"doc2md {__version__}",
    )
    args = parser.parse_args()

    # Collect all input files
    input_files = []
    for path in args.inputs:
        if os.path.isdir(path):
            if args.recursive:
                for root, _dirs, files in os.walk(path):
                    for entry in sorted(files):
                        ext = os.path.splitext(entry)[1].lower()
                        if ext in SUPPORTED_EXTENSIONS:
                            input_files.append(os.path.join(root, entry))
            else:
                for entry in sorted(os.listdir(path)):
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        input_files.append(os.path.join(path, entry))
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"{BOLD}Warning:{RESET} Skipping unsupported file: {path}")
                continue
            input_files.append(path)
        else:
            print(f"{BOLD}Warning:{RESET} Not found: {path}")

    if not input_files:
        print(f"{BOLD}Error:{RESET} No supported files found.")
        print(f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    # Setup
    client = get_client(args.api_key)
    enable_thinking = not args.no_thinking
    verbose = args.verbose or not RICH_AVAILABLE

    # Parse page ranges (we need a file's page count to resolve the ranges,
    # so we defer final resolution to processing time)
    pages_spec = args.pages

    _print_banner(
        len(input_files), args.dpi, enable_thinking, args.confidence, pages_spec or "",
    )

    # Filter out files that already have output
    files_to_process = []
    for fpath in input_files:
        if args.skip_existing:
            out_name = os.path.splitext(os.path.basename(fpath))[0] + ".md"
            out_check = os.path.join(args.output_dir, out_name)
            if os.path.exists(out_check):
                if RICH_AVAILABLE and not verbose:
                    console.print(f"  [dim]Skipping (exists): {out_check}[/]")
                else:
                    print(f"  {DIM}Skipping (exists): {out_check}{RESET}")
                continue
        files_to_process.append(fpath)

    # --- Process files ---
    # (filename, [page_markdowns], [confidence_scores])
    all_results: list[tuple[str, list[str], list[float | None]]] = []
    t0 = time.time()

    def _resolve_pages(fpath: str) -> list[int] | None:
        """Resolve page range spec for a given file."""
        if pages_spec is None:
            return None
        ext = os.path.splitext(fpath)[1].lower()
        if ext in PDF_EXTENSIONS:
            total = pdf_page_count(fpath)
        elif ext in OFFICE_EXTENSIONS:
            # For office files, we'll resolve after conversion to PDF
            # Return a sentinel that process_office will handle
            return None  # will be resolved inside process_office via process_pdf
        else:
            return None  # images don't have pages
        return parse_page_ranges(pages_spec, total)

    def _process_one(
        fpath: str,
        progress=None,
        progress_task=None,
    ) -> tuple[str, list[str], list[float | None]]:
        ext = os.path.splitext(fpath)[1].lower()

        if ext in PDF_EXTENSIONS:
            page_idx = _resolve_pages(fpath)
            results = process_pdf(
                client,
                fpath,
                dpi=args.dpi,
                enable_thinking=enable_thinking,
                enable_confidence=args.confidence,
                page_indices=page_idx,
                verbose=verbose,
                progress=progress,
                progress_task=progress_task,
            )
            pages = [r[0] for r in results]
            confs = [r[1] for r in results]
            return (fpath, pages, confs)

        elif ext in OFFICE_EXTENSIONS:
            # For office files with --pages, we need to convert first then apply
            # page ranges. The page spec gets resolved inside process_pdf
            # after office_to_pdf converts the file.
            page_idx = None
            if pages_spec:
                # We'll convert to PDF first, then resolve pages.
                # Use process_office which handles this.
                pass
            results = process_office(
                client,
                fpath,
                dpi=args.dpi,
                enable_thinking=enable_thinking,
                enable_confidence=args.confidence,
                page_indices=page_idx,
                verbose=verbose,
                progress=progress,
                progress_task=progress_task,
            )
            pages = [r[0] for r in results]
            confs = [r[1] for r in results]
            return (fpath, pages, confs)

        else:
            md, conf = process_image(
                client,
                fpath,
                enable_thinking=enable_thinking,
                enable_confidence=args.confidence,
                verbose=verbose,
                progress=progress,
                progress_task=progress_task,
            )
            return (fpath, [md], [conf])

    if RICH_AVAILABLE and not verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Converting pages...", total=None)

            if args.workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(_process_one, fp, progress, task): fp
                        for fp in files_to_process
                    }
                    for future in concurrent.futures.as_completed(futures):
                        all_results.append(future.result())
            else:
                for fpath in files_to_process:
                    all_results.append(_process_one(fpath, progress, task))
    else:
        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_process_one, fp): fp for fp in files_to_process
                }
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())
        else:
            for fpath in files_to_process:
                all_results.append(_process_one(fpath))

    elapsed = time.time() - t0

    # --- Validation ---
    validation_results: dict[str, list[str]] | None = None
    if args.validate:
        validation_results = {}
        for fpath, pages, _ in all_results:
            all_warnings = []
            for i, page_md in enumerate(pages, 1):
                page_warnings = validate_markdown(page_md)
                for w in page_warnings:
                    prefix = f"Page {i}: " if len(pages) > 1 else ""
                    all_warnings.append(f"{prefix}{w}")
            validation_results[fpath] = all_warnings

    # --- Write output ---
    if args.single_file:
        combined_parts = []
        for fpath, pages, _ in all_results:
            name = os.path.basename(fpath)
            doc_md = assemble_document(pages, name)
            combined_parts.append(f"# {name}\n\n{doc_md}")

        combined = "\n\n---\n\n".join(combined_parts)

        out_path = args.output or os.path.join(args.output_dir, "combined.md")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(combined + "\n")

        if RICH_AVAILABLE and not verbose:
            console.print(f"  [green bold]Written:[/] {out_path}")
        else:
            print(f"\n{GREEN}{BOLD}Written:{RESET} {out_path}")

    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for fpath, pages, _ in all_results:
            name = os.path.splitext(os.path.basename(fpath))[0] + ".md"
            doc_md = assemble_document(pages, name)
            out_path = os.path.join(args.output_dir, name)
            with open(out_path, "w") as f:
                f.write(doc_md + "\n")

            if RICH_AVAILABLE and not verbose:
                console.print(f"  [green bold]Written:[/] {out_path}")
            else:
                print(f"\n{GREEN}{BOLD}Written:{RESET} {out_path}")

    # --- Summary ---
    _print_summary(all_results, elapsed, validation_results)


if __name__ == "__main__":
    main()
