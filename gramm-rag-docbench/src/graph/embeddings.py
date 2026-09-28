"""
src/graph/embeddings.py
───────────────────────
Node embedding computation for GraMM-RAG.

Step 4: Embed all text nodes with E5-Mistral-7B (4-bit quantized, ~4GB VRAM)
        and all figure/image nodes with SigLIP-SO400M (~1.5GB VRAM).

⚠ Sequential processing is REQUIRED on RTX 5070 (8GB VRAM):
   1. Load E5-Mistral → embed all text → unload → save
   2. Load SigLIP     → embed all images → unload → save
   3. Load projection layers (tiny) → project to 256-dim → save

Both models running simultaneously would exceed 8GB VRAM.
"""

import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECTION_DIM = 256   # all nodes projected to this common dimension

# Hard cap on characters fed to the text encoder per node. ~2k chars is roughly
# 500 tokens, comfortably inside E5-Mistral's window and well above the 99th
# percentile element length in this corpus, so virtually nothing is lost.
MAX_TEXT_CHARS = 2000


# ─── Linear projection layers ──────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Single-layer linear projection to PROJECTION_DIM."""
    def __init__(self, in_dim: int, out_dim: int = PROJECTION_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ─── Text embedding (E5-Mistral-7B 4-bit or E5-large-v2 fallback) ────────────

# D-32: opt-in guard for the dimension-changing fallback. Off by default so a
# transient OOM can never silently mix 1024-dim and 4096-dim vectors in one
# corpus. Set GRAMM_ALLOW_E5_FALLBACK=1 only when re-embedding EVERY document
# with E5-large-v2 on a VRAM-limited machine.
ALLOW_DIM_CHANGING_FALLBACK = os.environ.get("GRAMM_ALLOW_E5_FALLBACK") == "1"


# D-33: process-wide singleton for the text encoder. Keyed by device so a CPU
# and CUDA run in the same process cannot collide.
_TEXT_MODEL_CACHE: dict = {}


def _get_cached_text_model(device: str = "cuda"):
    """Load E5-Mistral once per process and reuse it for every document.

    Returns the same (model, dim) tuple on subsequent calls. See D-33: the
    per-document load pattern leaked VRAM until the GPU OOM'd mid-corpus.
    """
    key = str(device)
    if key not in _TEXT_MODEL_CACHE:
        _TEXT_MODEL_CACHE[key] = load_text_model(
            use_4bit=(device == "cuda"), device=device
        )
    return _TEXT_MODEL_CACHE[key]


def load_text_model(use_4bit: bool = True, device: str = "cuda"):
    """
    Load E5-Mistral-7B-Instruct with 4-bit quantization (~4GB VRAM).
    Falls back to E5-large-v2 (~1.3GB) if 4-bit loading fails.
    """
    try:
        from sentence_transformers import SentenceTransformer
        if use_4bit:
            logger.info("Loading E5-Mistral-7B-Instruct (4-bit quantized)")
            model = SentenceTransformer(
                "intfloat/e5-mistral-7b-instruct",
                model_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                },
                device=device,
            )
            embed_dim = 4096
        else:
            raise RuntimeError("Skipping 4-bit, using fallback.")
    except Exception as e:
        # D-32: the E5-large-v2 fallback is a *deployment* fallback for machines
        # that cannot host the 4-bit 7B model. It must NEVER fire mid-corpus:
        # E5-large-v2 emits 1024-dim vectors, so a document embedded by the
        # fallback lands in a different vector space from the rest of the corpus
        # while every downstream stage (projection to 256-dim, FAISS, HGT) still
        # "works". On the first full run a transient CUDA OOM triggered this for
        # doc 150, whose retrieval was silently meaningless.
        #
        # Transient OOM is NOT a reason to change embedding models. Re-raise so
        # the document is recorded as failed and can be re-embedded on a clean
        # GPU. Only an explicit opt-in allows the dimension change.
        if not ALLOW_DIM_CHANGING_FALLBACK:
            logger.error(
                f"E5-Mistral 4-bit failed ({e}). REFUSING the 1024-dim "
                f"E5-large-v2 fallback: it would mix embedding spaces within one "
                f"corpus. Set GRAMM_ALLOW_E5_FALLBACK=1 only for a full re-embed "
                f"of every document on a VRAM-limited machine."
            )
            raise
        logger.warning(f"E5-Mistral 4-bit failed ({e}). Using E5-large-v2 fallback.")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/e5-large-v2", device=device)
        embed_dim = 1024

    return model, embed_dim


def embed_text_nodes(
    elements: list,
    model,
    batch_size: int = 16,   # ← EXTEND: increase to 32+ when VRAM allows
    device: str = "cuda",
) -> torch.Tensor:
    """
    Embed all text-type elements.

    Args:
        elements:   List of parsed element dicts (from mineru_wrapper).
        model:      Loaded SentenceTransformer model.
        batch_size: Inference batch size.
                    # LIMIT: set small (16) to fit VRAM during dev.
                    # ← EXTEND: increase for faster throughput in full runs.

    Returns:
        Tensor of shape [N_text, embed_dim].
    """
    text_elems = [e for e in elements if e["type"] in ("text", "section", "equation")]
    texts = []
    for e in text_elems:
        # E5-Mistral expects "Instruct: <task>\nQuery: <text>" format
        #
        # Truncate to MAX_TEXT_CHARS first. MinerU occasionally emits a single
        # very large block (the corpus worst case is a 19,089-char element in
        # 2307.09288v2.pdf, ~5k tokens); batching several of those together
        # allocates an attention tensor big enough to kill the process natively
        # — the CUDA allocator aborts rather than raising, so it surfaces only as
        # an opaque "Kernel died" with no Python traceback.
        #
        # Measured on this corpus (23,622 text-like nodes): p50=103, p99=1,431,
        # p99.5=1,765 chars. A 2,000-char cap truncates 88 nodes (0.37%), i.e.
        # 3.3% of total corpus characters, all in the tail of a few very long
        # blocks. Retrieval granularity is the page/element, so a node's identity
        # is carried by its opening text; the dropped tails do not change which
        # page a node points at.
        texts.append(f"Represent this document passage: {e['text'][:MAX_TEXT_CHARS]}")

    if not texts:
        return torch.zeros(0, PROJECTION_DIM)

    logger.info(f"Embedding {len(texts)} text nodes (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return embeddings.cpu()


# ─── Image embedding (SigLIP-SO400M) ──────────────────────────────────────────

def load_image_model(device: str = "cuda"):
    """
    Load SigLIP-SO400M (~1.5GB VRAM). Safe to run on RTX 5070 alone.
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("Run: pip install open-clip-torch")

    logger.info("Loading SigLIP-SO400M (ViT-SO400M-14-SigLIP)")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP", pretrained="webli"
    )
    model = model.to(device).eval()
    embed_dim = 1152   # SigLIP-SO400M output dimension
    return model, preprocess, embed_dim


def embed_image_nodes(
    elements: list,
    model,
    preprocess,
    image_root: str,
    batch_size: int = 8,    # ← EXTEND: increase when VRAM allows
    device: str = "cuda",
) -> torch.Tensor:
    """
    Embed all figure/table elements using their extracted image crops.

    Args:
        elements:   List of parsed element dicts.
        model:      Loaded SigLIP model.
        preprocess: SigLIP image transform.
        image_root: Directory containing image crops (MinerU artifact dir).
        batch_size: Inference batch size.
                    # LIMIT: 8 to fit VRAM.
                    # ← EXTEND: increase for faster throughput.

    Returns:
        Tensor of shape [N_figure, embed_dim].
    """
    from PIL import Image

    fig_elems = [e for e in elements if e["type"] in ("figure", "table")]
    if not fig_elems:
        return torch.zeros(0, 1152)

    # Blank placeholder must match the preprocessed crop shape, or torch.stack
    # below raises on the mixed batch. Derive it from the transform rather than
    # hard-coding 224 (SigLIP-SO400M is 224; the -384 variant is not).
    _blank = preprocess(Image.new("RGB", (224, 224)))

    images = []
    for e in fig_elems:
        rel = e.get("image_path", "")
        # MinerU sometimes tags a figure/table region without writing a crop,
        # leaving image_path empty (32/2,884 elements = 1.1% of this corpus,
        # across 9 documents). `Path(root) / ""` collapses to `root`, whose
        # .exists() is True, so the old check passed and PIL then died with
        # IsADirectoryError -- killing the whole embedding phase for the doc.
        # Treat an empty path as a missing crop and fall through to the blank.
        img_path = Path(image_root) / rel if rel else None
        if img_path is not None and img_path.is_file():
            images.append(preprocess(Image.open(img_path).convert("RGB")))
        else:
            # Create blank placeholder if image crop not found
            logger.warning(f"Image not found for element {e['id']}, using blank.")
            images.append(_blank)

    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(images[i : i + batch_size]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = model.encode_image(batch)
        embeddings.append(feats.cpu())

    return torch.cat(embeddings, dim=0)


# ─── Project all embeddings to 256-dim ────────────────────────────────────────

def project_embeddings(
    text_emb: torch.Tensor,
    img_emb: torch.Tensor,
    text_proj: ProjectionHead,
    img_proj: ProjectionHead,
) -> tuple:
    """Apply linear projection to common 256-dim space.

    Both encoders emit float16 — E5-Mistral because it is loaded 4-bit
    quantized, SigLIP because `embed_image_nodes` runs under autocast — while
    `ProjectionHead`'s nn.Linear defaults to float32. Feeding one to the other
    raises "mat1 and mat2 must have the same dtype, but got Half and Float".
    Cast the inputs to the projection's own dtype rather than casting the
    module, so the stored 256-dim features stay float32 like the HGT expects.
    """
    text_proj.eval()
    img_proj.eval()

    def _apply(proj, emb):
        if emb.shape[0] == 0:
            return emb
        w_dtype = next(proj.parameters()).dtype
        return proj(emb.to(w_dtype))

    with torch.no_grad():
        text_out = _apply(text_proj, text_emb)
        img_out = _apply(img_proj, img_emb)
    return text_out, img_out


# ─── Main driver ───────────────────────────────────────────────────────────────

def compute_and_save_embeddings(
    parsed: dict,
    embeddings_dir: str = None,   # canonical param name
    image_root: str = "",
    device: str = None,
    force: bool = False,
    # ── Aliases used by 03_build_graphs.ipynb ────────────────────────────────
    emb_dir: str = None,          # alias for embeddings_dir
    text_batch_size: int = 16,    # LIMIT: 16 for 8GB VRAM; ← EXTEND: 32+
    image_batch_size: int = 8,    # LIMIT: 8 for 8GB VRAM;  ← EXTEND: 16+
) -> dict:
    """
    Full embedding pipeline for one document. Saves .pt tensors to disk.

    Outputs:
      embeddings/{doc_id}_text.pt  — shape [N_text, 256]
      embeddings/{doc_id}_img.pt   — shape [N_fig,  256]

    Sequential processing: text model → unload → image model → unload → project.

    Args:
        parsed:         Parsed document dict (from mineru_wrapper/mpdocvqa_parser).
        embeddings_dir: Output directory for .pt files.
        emb_dir:        Alias for embeddings_dir (accepted from notebook).
        image_root:     Root dir for image crops (MinerU artifacts).
        device:         'cuda' or 'cpu' (auto-detected if None).
        force:          Recompute even if cached.
        text_batch_size: # LIMIT: batch size for text encoding (fit 8GB VRAM).
                         # ← EXTEND: increase for faster throughput.
        image_batch_size: # LIMIT: batch size for image encoding.
                          # ← EXTEND: increase when VRAM allows.
    """
    # Resolve embeddings_dir from either canonical param or alias
    out_dir = embeddings_dir or emb_dir
    if out_dir is None:
        raise ValueError("Provide either embeddings_dir or emb_dir.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    doc_id = parsed["doc_id"]
    out_text = Path(out_dir) / f"{doc_id}_text.pt"
    out_img  = Path(out_dir) / f"{doc_id}_img.pt"
    # Raw (un-projected) E5 text embeddings, kept for semantic vector retrieval.
    # The projected _text.pt feeds the HGT graph; _text_raw.pt feeds FAISS so
    # queries can be embedded into the SAME E5 space (the projection layer is
    # random/per-doc/unsaved and cannot be reproduced for a query).
    out_text_raw = Path(out_dir) / f"{doc_id}_text_raw.pt"

    if (out_text.exists() and out_img.exists() and out_text_raw.exists()
            and not force):
        logger.info(f"Embeddings cached for {doc_id}, skipping.")
        return {
            "text": torch.load(out_text, weights_only=True),
            "img":  torch.load(out_img,  weights_only=True),
            "text_raw": torch.load(out_text_raw, weights_only=True),
        }

    elements = parsed["elements"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Text embeddings ──────────────────────────────────────────────
    # D-33: reuse ONE cached E5-Mistral instance across documents. The previous
    # code loaded the 7B model per document and relied on `del text_model` +
    # empty_cache() to reclaim it, but `del` only drops the local reference --
    # SentenceTransformer retains the module, so VRAM grew monotonically. After
    # ~140 docs the A100 hit 38.6/39.5 GiB and 6 documents died with CUDA OOM
    # (see D-32). Loading once also removes ~25s of per-document load time.
    text_model, text_dim = _get_cached_text_model(device=device)
    raw_text_emb = embed_text_nodes(
        elements, text_model, batch_size=text_batch_size, device=device
    )
    if device == "cuda":
        torch.cuda.empty_cache()   # frees per-batch activations, keeps weights
    logger.info(f"Text embeddings: {raw_text_emb.shape}")

    # Persist the RAW E5 text embeddings (pre-projection) for vector retrieval.
    torch.save(raw_text_emb, out_text_raw)

    # ── Step 2: Image embeddings ─────────────────────────────────────────────
    has_images = any(e.get("type") in ("figure", "image") for e in elements)
    if has_images:
        img_model, preprocess, img_dim = load_image_model(device=device)
        raw_img_emb = embed_image_nodes(
            elements, img_model, preprocess, image_root,
            batch_size=image_batch_size, device=device
        )
        del img_model
        if device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"Image embeddings: {raw_img_emb.shape}")
    else:
        # No figure/image elements — skip SigLIP loading entirely (saves ~5s per doc)
        raw_img_emb = torch.zeros(0, PROJECTION_DIM)
        img_dim = PROJECTION_DIM
        logger.info("No image elements — skipping SigLIP (image embeddings: empty)")

    # ── Step 3: Project to 256-dim ───────────────────────────────────────────
    text_proj = ProjectionHead(text_dim if text_dim > 0 and raw_text_emb.shape[0] > 0
                               else (raw_text_emb.shape[-1] if raw_text_emb.shape[0] > 0
                                     else PROJECTION_DIM))
    img_proj  = ProjectionHead(img_dim  if img_dim  > 0 and raw_img_emb.shape[0]  > 0
                               else (raw_img_emb.shape[-1]  if raw_img_emb.shape[0]  > 0
                                     else PROJECTION_DIM))

    # Projection layers start randomly initialised; jointly trained with HGT in Step 8.
    text_emb, img_emb = project_embeddings(raw_text_emb, raw_img_emb, text_proj, img_proj)

    torch.save(text_emb, out_text)
    torch.save(img_emb,  out_img)
    logger.info(f"Saved embeddings for {doc_id} → {out_dir}")

    return {"text": text_emb, "img": img_emb, "text_raw": raw_text_emb}
