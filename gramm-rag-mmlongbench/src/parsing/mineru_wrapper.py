"""
src/parsing/mineru_wrapper.py
─────────────────────────────
MinerU PDF parsing wrapper for GraMM-RAG.

MinerU (magic-pdf) converts PDFs into structured JSON with:
  - bounding boxes per element
  - element types: text / table / figure / equation / section header
  - reading order

VRAM: ~6GB. Fits on RTX 5070 (8GB). Run parse_all_pdfs() overnight for
large benchmarks (~4 hrs for MP-DocVQA full set).

Install:
    pip install magic-pdf[full]
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: str, output_dir: str) -> dict:
    """
    Parse a single PDF with MinerU and return structured JSON.

    Args:
        pdf_path:   Absolute path to the PDF file.
        output_dir: Directory to write MinerU output artifacts (images, etc.)

    Returns:
        Parsed result dict with keys: elements, metadata.
        Each element has: text, type, bbox, page_no, reading_order.
    """
    try:
        # MinerU imports — only imported when actually called so the rest of the
        # codebase can be imported even if magic-pdf is not installed.
        #
        # API NOTE: this targets magic-pdf 1.x (PymuDocDataset + doc_analyze).
        # The 0.x `magic_pdf.pipe.UNIPipe` entry point this wrapper originally
        # used was removed upstream and does not exist in >=1.0.
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    except ImportError as e:
        raise ImportError(
            "MinerU (magic-pdf >=1.0) is not installed. "
            "Run: pip install magic-pdf[full]"
        ) from e

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # MinerU writes figure/table crops here; embeddings.py resolves element
    # 'image_path' values relative to this same directory.
    image_writer = FileBasedDataWriter(str(out))

    pdf_bytes = open(pdf_path, "rb").read()
    dataset = PymuDocDataset(pdf_bytes)

    # ocr=False -> text-extraction mode. MMLongBench PDFs are digital-born, so
    # the text layer is authoritative; OCR mode is both slower and lossier here.
    infer_result = doc_analyze(dataset, ocr=False)
    pipe_result = infer_result.pipe_txt_mode(image_writer)

    middle = pipe_result.get_middle_json()
    if isinstance(middle, str):
        middle = json.loads(middle)
    pages = middle.get("pdf_info", [])

    # Normalise MinerU output into our internal format
    elements = []
    reading_order_counter = 0
    for page_idx, page in enumerate(pages):
        # para_blocks is the post-processed, reading-order-sorted block list.
        for block in page.get("para_blocks", []):
            el_type = _normalise_type(block.get("type", "text"))
            text = _block_text(block)
            image_path = _block_image_path(block)
            # Keep image-bearing nodes even when they carry no text: figure and
            # table nodes are embedded by SigLIP from their crops, not their text.
            if not text and not image_path:
                continue
            elements.append({
                "id": f"p{page_idx + 1}_e{reading_order_counter}",
                "type": el_type,
                "text": text,
                "bbox": block.get("bbox", []),         # [x0, y0, x1, y1] in pts
                # 1-BASED page numbers. MinerU's pdf_info index is 0-based, but
                # MMLongBench's gold `evidence_pages` are 1-based, and APPA
                # compares retrieved page_no against those values directly.
                # Verified empirically over 308 answerable questions across the
                # 73 pilot docs by locating each gold answer string in the parsed
                # page text: 1-based matched 58 vs 23 for 0-based (and 34 vs 10
                # restricted to answers >=10 chars). Emitting 0-based here would
                # silently shift every APPA comparison by one page and deflate the
                # headline retrieval-localisation metric. The MP-DocVQA parser
                # (src/parsing/mpdocvqa_parser.py) is likewise 1-based.
                "page_no": page_idx + 1,
                "reading_order": reading_order_counter,
                "image_path": image_path,
            })
            reading_order_counter += 1

    result = {
        "doc_id": Path(pdf_path).stem,
        "pdf_path": str(pdf_path),
        "num_pages": len(pages),
        "elements": elements,
    }
    return result


def _block_text(block: dict) -> str:
    """
    Flatten a MinerU 1.x block into plain text.

    Structure is block -> lines -> spans -> content. Table and image blocks nest
    one level deeper under 'blocks' (e.g. table_body / table_caption), so those
    are walked recursively.
    """
    parts: list[str] = []

    def walk(node: dict) -> None:
        for line in node.get("lines", []):
            for span in line.get("spans", []):
                # Equation spans carry LaTeX rather than plain content.
                content = span.get("content") or span.get("latex") or ""
                if content:
                    parts.append(content)
        for sub in node.get("blocks", []):
            walk(sub)

    walk(block)
    return " ".join(p.strip() for p in parts if p.strip()).strip()


def _block_image_path(block: dict) -> str:
    """Return the crop filename for an image/table block, or '' if it has none."""

    def walk(node: dict) -> str:
        for line in node.get("lines", []):
            for span in line.get("spans", []):
                p = span.get("image_path")
                if p:
                    return p
        for sub in node.get("blocks", []):
            found = walk(sub)
            if found:
                return found
        return ""

    return walk(block)


def _normalise_type(raw_type: str) -> str:
    """Map MinerU element types to GraMM-RAG canonical types."""
    mapping = {
        "text": "text",
        "para": "text",
        "title": "section",
        "interline_equation": "equation",
        "table": "table",
        "figure": "figure",
        "image": "figure",
        # magic-pdf 1.x block-type names. Captions/footnotes are kept as text so
        # the builder's caption edges (text -> figure) still have endpoints.
        "image_body": "figure",
        "image_caption": "text",
        "image_footnote": "text",
        "table_body": "table",
        "table_caption": "text",
        "table_footnote": "text",
        "index": "section",
        "list": "text",
        "discarded": "text",
    }
    return mapping.get(raw_type.lower(), "text")


def parse_all_pdfs(
    pdf_dir: str,
    parsed_dir: str,
    max_docs: Optional[int] = None,   # ← EXTEND: set to None for full paper run
) -> None:
    """
    Batch-parse all PDFs in pdf_dir, saving JSON to parsed_dir.

    Args:
        pdf_dir:   Directory containing PDF files.
        parsed_dir: Output directory for parsed JSON files.
        max_docs:  Maximum number of PDFs to process. Set to None for all.
                   # LIMIT: Currently capped to save time during dev/testing.
                   # ← EXTEND: remove or increase for full paper run.
    """
    pdf_paths = sorted(Path(pdf_dir).glob("**/*.pdf"))

    # LIMIT: cap number of documents for end-to-end testing
    # ← EXTEND: remove the slice below for the full paper run
    if max_docs is not None:
        pdf_paths = pdf_paths[:max_docs]
        logger.info(f"[LIMIT] Capped to {max_docs} PDFs for dev testing. "
                    "Extend max_docs for full run.")

    logger.info(f"Parsing {len(pdf_paths)} PDFs from {pdf_dir}")
    Path(parsed_dir).mkdir(parents=True, exist_ok=True)

    for i, pdf_path in enumerate(pdf_paths):
        doc_id = pdf_path.stem
        out_path = Path(parsed_dir) / f"{doc_id}.json"

        if out_path.exists():
            logger.info(f"[{i+1}/{len(pdf_paths)}] Skipping (cached): {doc_id}")
            continue

        logger.info(f"[{i+1}/{len(pdf_paths)}] Parsing: {doc_id}")
        try:
            art_dir = Path(parsed_dir) / "artifacts" / doc_id
            result = parse_pdf(str(pdf_path), str(art_dir))
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to parse {doc_id}: {e}")


def load_parsed(parsed_dir: str, doc_id: str) -> Optional[dict]:
    """Load a cached parsed JSON result for a given doc_id."""
    path = Path(parsed_dir) / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
