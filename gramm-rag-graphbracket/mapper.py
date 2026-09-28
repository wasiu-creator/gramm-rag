"""
Canonical parse -> comparator-input mapping for the graph bracket (WP-1).

ONE implementation, consumed by every comparator, so "same MinerU parse to every
text-side ingester" is enforced by construction rather than asserted in prose.
Reinventing a predicate per-arm is the C-8/C-9 failure class: both were silent
divergences from canonical semantics that produced plausible, wrong numbers.

Our archived parse schema (cache/parsed/<doc>.json):
    {doc_id, pdf_path, num_pages, elements: [
        {id, type: section|text|figure|table, text, bbox, page_no (1-based),
         reading_order, image_path, temporal_markers}]}

MAPPING DECISIONS, stated because they are deviations a reader must be able to audit:

D1. `table` elements carry text == "" in our parse -- the table body exists ONLY as a
    cropped image. So tables contribute NOTHING to any text-side ingester. This is a
    property of the shared parse, not a choice made against a comparator: GraMM's own
    text arm sees the same emptiness. Tables therefore map to RAG-Anything as *images*
    (img_path), never as {"type":"table","table_body":...} with fabricated content.
D2. `figure` elements DO carry caption text, which the text side sees; the figure image
    additionally goes to RAG-Anything's multimodal path.
D3. page_no is 1-based in our parse, page_idx is 0-based in MinerU's content_list.
D4. Ordering is by `reading_order` -- the same field the reading_order edge type is built
    from, so text-side ingest sees the document in the order the graph encodes.
"""
from pathlib import Path
from typing import Any


def load_parse(path: str | Path) -> dict:
    import json
    d = json.loads(Path(path).read_text())
    if "elements" not in d:
        raise ValueError(f"{path}: no 'elements' key -- not a GraMM parse")
    return d


def _ordered(parsed: dict) -> list[dict]:
    return sorted(parsed["elements"], key=lambda e: e.get("reading_order", 0))


def elements_to_text(parsed: dict) -> str:
    """Reading-order text for TEXT-SIDE ingesters (LightRAG).

    Sections are emitted as markdown headings so a chunker can see structure; text and
    figure captions as paragraphs. Tables emit nothing (D1) -- deliberately, and the
    pilot log records how much content that silently removes.
    """
    out = []
    for e in _ordered(parsed):
        t = (e.get("text") or "").strip()
        if not t:
            continue
        out.append(f"## {t}" if e.get("type") == "section" else t)
    return "\n\n".join(out)


def elements_to_content_list(parsed: dict, artifacts_dir: str | Path) -> list[dict]:
    """MinerU content_list for RAG-Anything's insert_content_list().

    Schema per RAGAnything.insert_content_list docstring (v1.2.8):
      text     {"type":"text",  "text":str, "page_idx":int}
      image    {"type":"image", "img_path":str, "image_caption":[str], "page_idx":int}
    """
    art = Path(artifacts_dir)
    cl: list[dict] = []
    for e in _ordered(parsed):
        page_idx = max(0, int(e.get("page_no", 1)) - 1)          # D3
        text = (e.get("text") or "").strip()
        img = (e.get("image_path") or "").strip()
        etype = e.get("type")

        if etype in ("section", "text"):
            if text:
                cl.append({"type": "text", "text": text, "page_idx": page_idx})
        elif etype in ("figure", "table"):
            if img and (art / img).exists():
                cl.append({"type": "image",
                           "img_path": str((art / img).resolve()),
                           "image_caption": [text] if text else [],
                           "page_idx": page_idx})                 # D1, D2
            elif text:
                cl.append({"type": "text", "text": text, "page_idx": page_idx})
        elif text:
            cl.append({"type": "text", "text": text, "page_idx": page_idx})
    return cl


def parse_stats(parsed: dict, artifacts_dir: str | Path) -> dict[str, Any]:
    """What each side actually receives -- so 'harness identity' is a measured claim."""
    art = Path(artifacts_dir)
    els = parsed["elements"]
    by_type: dict[str, int] = {}
    for e in els:
        by_type[e.get("type", "?")] = by_type.get(e.get("type", "?"), 0) + 1
    txt = elements_to_text(parsed)
    cl = elements_to_content_list(parsed, art)
    return {
        "n_elements": len(els),
        "elements_by_type": by_type,
        "n_pages": parsed.get("num_pages", 0),
        "text_chars": len(txt),
        "content_list_items": len(cl),
        "content_list_text_items": sum(1 for c in cl if c["type"] == "text"),
        "content_list_image_items": sum(1 for c in cl if c["type"] == "image"),
        "images_missing_on_disk": sum(
            1 for e in els if e.get("image_path") and not (art / e["image_path"]).exists()),
        "tables_dropped_from_text": sum(
            1 for e in els if e.get("type") == "table" and not (e.get("text") or "").strip()),
    }
