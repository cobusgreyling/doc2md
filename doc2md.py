#!/usr/bin/env python3
"""
doc2md — Convert documents to LLM-ready Markdown using NVIDIA Nemotron 3 Nano Omni.

Accepts PDFs and images (PNG, JPG, TIFF, BMP), sends each page/image to the
Nemotron vision model, and produces clean Markdown output optimised for LLM
ingestion: proper headings, tables, lists, code blocks, and image descriptions.

Usage:
    python doc2md.py report.pdf
    python doc2md.py slide.png diagram.jpg
    python doc2md.py *.pdf --output-dir ./markdown
    python doc2md.py invoice.pdf --single-file
"""

import argparse
import base64
import io
import os
import sys
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


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
# PDF handling
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[bytes]:
    """Convert each page of a PDF to a PNG image using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"{BOLD}Error:{RESET} pymupdf is required for PDF support.")
        print("  Install it: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))

    doc.close()
    return images


# ---------------------------------------------------------------------------
# Conversion via Nemotron
# ---------------------------------------------------------------------------

def convert_image_to_markdown(
    client: OpenAI,
    image_data_url: str,
    *,
    page_label: str = "",
    enable_thinking: bool = True,
) -> str:
    """Send a single image to Nemotron and return the Markdown output."""
    prompt = "Convert this document page to Markdown. Follow the system instructions exactly."
    if page_label:
        prompt = f"This is {page_label}. " + prompt

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
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
            print(delta.content, end="", flush=True)

    print()
    return content.strip()


def process_pdf(client: OpenAI, pdf_path: str, *, dpi: int = 200) -> list[str]:
    """Convert a PDF to a list of Markdown strings, one per page."""
    filename = os.path.basename(pdf_path)
    print(f"\n{CYAN}{BOLD}[PDF]{RESET} {filename}")

    images = pdf_to_images(pdf_path, dpi=dpi)
    print(f"  {len(images)} page(s) extracted at {dpi} DPI\n")

    pages = []
    for i, img_bytes in enumerate(images, 1):
        page_label = f"page {i} of {len(images)}"
        print(f"  {BOLD}--- Page {i}/{len(images)} ---{RESET}")
        data_url = encode_image_bytes(img_bytes, "image/png")
        md = convert_image_to_markdown(client, data_url, page_label=page_label)
        pages.append(md)
        print()

    return pages


def process_image(client: OpenAI, image_path: str) -> str:
    """Convert a single image file to Markdown."""
    filename = os.path.basename(image_path)
    print(f"\n{CYAN}{BOLD}[Image]{RESET} {filename}")

    data_url = load_image_file(image_path)
    md = convert_image_to_markdown(client, data_url, page_label=filename)
    return md


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
  python doc2md.py *.pdf --output-dir ./markdown
  python doc2md.py report.pdf --dpi 300
  python doc2md.py docs/ --single-file --output combined.md
        """,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF files, image files, or directories to convert",
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
    args = parser.parse_args()

    # Collect all input files
    input_files = []
    for path in args.inputs:
        if os.path.isdir(path):
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

    print(f"""
{BOLD}{'='*60}
  doc2md — Document to Markdown Converter
  Powered by NVIDIA Nemotron 3 Nano Omni
{'='*60}{RESET}

  Files to process: {len(input_files)}
  DPI (for PDFs):   {args.dpi}
  Reasoning:        {'off' if args.no_thinking else 'on'}
""")

    # Process each file
    all_results: list[tuple[str, list[str]]] = []  # (filename, [page_markdowns])
    t0 = time.time()

    for fpath in input_files:
        ext = os.path.splitext(fpath)[1].lower()

        if ext in PDF_EXTENSIONS:
            pages = process_pdf(client, fpath, dpi=args.dpi)
            all_results.append((fpath, pages))
        elif ext in IMAGE_EXTENSIONS:
            md = process_image(client, fpath)
            all_results.append((fpath, [md]))

    elapsed = time.time() - t0

    # Write output
    if args.single_file:
        combined_parts = []
        for fpath, pages in all_results:
            name = os.path.basename(fpath)
            doc_md = assemble_document(pages, name)
            combined_parts.append(f"# {name}\n\n{doc_md}")

        combined = "\n\n---\n\n".join(combined_parts)

        out_path = args.output or os.path.join(args.output_dir, "combined.md")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(combined + "\n")
        print(f"\n{GREEN}{BOLD}Written:{RESET} {out_path}")

    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for fpath, pages in all_results:
            name = os.path.splitext(os.path.basename(fpath))[0] + ".md"
            doc_md = assemble_document(pages, name)
            out_path = os.path.join(args.output_dir, name)
            with open(out_path, "w") as f:
                f.write(doc_md + "\n")
            print(f"\n{GREEN}{BOLD}Written:{RESET} {out_path}")

    # Summary
    total_pages = sum(len(pages) for _, pages in all_results)
    print(f"""
{BOLD}{'='*60}
  CONVERSION COMPLETE
{'='*60}{RESET}
  Files processed : {len(all_results)}
  Total pages     : {total_pages}
  Time elapsed    : {elapsed:.1f}s
{BOLD}{'='*60}{RESET}
""")


if __name__ == "__main__":
    main()
