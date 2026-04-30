# doc2md

![doc2md — Convert PDFs & Images to Clean, LLM-Ready Markdown Using NVIDIA Nemotron 3 Nano Omni](header.jpg)

Convert PDFs and images to clean, LLM-ready Markdown using NVIDIA Nemotron 3 Nano Omni.

```
  ┌──────────────┐      ┌───────────────────────┐      ┌──────────────┐
  │  PDF / Image │ ───► │  Nemotron 3 Nano Omni │ ───► │  Markdown    │
  │              │      │  (Vision + Reasoning) │      │  (.md files) │
  └──────────────┘      └───────────────────────┘      └──────────────┘
```

## Why

LLMs need clean text. PDFs and scanned documents are not clean text. This tool bridges the gap: it sends each page to Nemotron's vision model and gets back structured Markdown with proper headings, tables, lists, code blocks, and figure descriptions.

The output is designed to drop straight into RAG pipelines, LLM context windows, or knowledge bases.

## Requirements

- Python 3.10+

## Setup

```bash
pip install -r requirements.txt
```

Or install as a package (makes `doc2md` available as a CLI command):

```bash
pip install .
```

Set your NVIDIA API key:

```bash
export NVIDIA_API_KEY=nvapi-your-key-here
```

You can get an API key from the [NVIDIA API Catalog](https://build.nvidia.com/).

## Usage

### Convert a PDF

```bash
python doc2md.py report.pdf
```

Each page becomes Markdown. Output goes to `output/report.md`.

### Convert images

```bash
python doc2md.py slide.png diagram.jpg whiteboard.jpeg
```

Each image becomes a separate `.md` file in `output/`.

### Convert a directory of documents

```bash
python doc2md.py ./documents/
```

Picks up all PDFs and images in the directory.

### Combine everything into one file

```bash
python doc2md.py *.pdf --single-file --output knowledge_base.md
```

### Higher DPI for dense documents

```bash
python doc2md.py small-text.pdf --dpi 300
```

### Faster mode (skip reasoning)

```bash
python doc2md.py quick-scan.pdf --no-thinking
```

### Recursive directory scanning

```bash
python doc2md.py ./documents/ --recursive
```

### Skip already-converted files

```bash
python doc2md.py ./documents/ --skip-existing
```

### Parallel processing

```bash
python doc2md.py *.pdf --workers 4
```

## Options

| Flag | Description |
|---|---|
| `--output-dir`, `-o` | Output directory (default: `output/`) |
| `--single-file`, `-s` | Combine all inputs into one Markdown file |
| `--output`, `-O` | Output file path (with `--single-file`) |
| `--dpi` | PDF rendering DPI (default: 200) |
| `--api-key` | NVIDIA API key (or use `NVIDIA_API_KEY` env var) |
| `--no-thinking` | Disable reasoning mode for faster conversion |
| `--recursive`, `-r` | Scan directories recursively |
| `--skip-existing` | Skip files with existing `.md` output |
| `--workers`, `-w` | Number of files to process in parallel (default: 1) |
| `--version`, `-V` | Show version and exit |

## Supported Formats

**Documents:** PDF

**Images:** PNG, JPG/JPEG, TIFF/TIF, BMP, WebP

## How It Works

1. **PDFs** are rendered page-by-page into images using PyMuPDF
2. Each image is base64-encoded and sent to Nemotron 3 Nano Omni via the NVIDIA NIM API
3. The model extracts all visible content and converts it to structured Markdown
4. Output is assembled with page separators and written to disk

The model uses reasoning mode by default, which means it analyses document structure before producing output. This gives better results for complex layouts (multi-column, nested tables, mixed content). Use `--no-thinking` to skip reasoning for simple documents.

## What the Output Looks Like

The converter produces Markdown with:

- **Headings** preserved from the document hierarchy
- **Tables** as proper Markdown tables with alignment
- **Lists** (ordered and unordered) faithfully reproduced
- **Code blocks** wrapped in fenced blocks with language hints
- **Figures/charts** described in blockquotes: `> [Figure]: Description`
- **Math** in LaTeX notation: `$E = mc^2$`
- **Page breaks** as horizontal rules with page-number comments

## Model

This tool uses [NVIDIA Nemotron 3 Nano Omni](https://build.nvidia.com/) — a 30B-parameter mixture-of-experts model (3B active per inference) that natively processes text and images in a single forward pass. It leads on OCR benchmarks and produces structured output via an OpenAI-compatible API.

## License

MIT
