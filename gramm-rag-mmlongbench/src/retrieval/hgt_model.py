"""
src/retrieval/hgt_model.py
──────────────────────────
Heterogeneous Graph Transformer (HGT) retrieval module for GraMM-RAG.

Step 8: Implements DocumentHGT using PyG's HGTConv layers.
        Trained with InfoNCE contrastive loss using MMLongBench-Doc
        evidence page annotations as positive pairs.

Architecture:
  - Input: 256-dim projected node features (all types)
  - 2x HGTConv layers (256 hidden, 8 heads)
  - Output: 256-dim contextualised node embeddings

VRAM: ~2.1GB. Comfortably fits on RTX 5070 (8GB).
Training: 50 epochs, AdamW (lr=1e-3, wd=1e-4), batch of 32
          (1 query + 1 positive + 15 hard negatives).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


# ─── Node types and edge types (must match builder.py) ─────────────────────────
NODE_TYPES = ["text", "table", "figure", "equation", "section"]
EDGE_TYPES = [
    ("text",    "spatial",       "table"),
    ("text",    "reading_order", "text"),
    ("section", "hierarchy",     "text"),
    ("text",    "temporal",      "text"),
    ("text",    "coref",         "text"),
    ("text",    "crosspage",     "text"),
    ("text",    "caption",       "figure"),
    ("text",    "kg",            "text"),
]
METADATA = (NODE_TYPES, EDGE_TYPES)


class DocumentHGT(nn.Module):
    """
    Heterogeneous Graph Transformer for document understanding.

    Performs message passing across all 8 edge types simultaneously,
    allowing the model to learn cross-modal, cross-page, and
    hierarchical document structure.
    """

    def __init__(
        self,
        hidden_channels: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        metadata: tuple = METADATA,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Input projection per node type (all are already 256-dim, but we
        # add a small linear to make the model trainable end-to-end)
        self.node_lin = nn.ModuleDict({
            nt: Linear(-1, hidden_channels) for nt in NODE_TYPES
        })

        # HGT convolution layers
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            for _ in range(num_layers)
        ])

        # Final normalisation
        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_channels) for nt in NODE_TYPES
        })

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
    ) -> dict:
        """
        Args:
            x_dict:          {node_type: Tensor[N, 256]} — raw node features
            edge_index_dict: {(src, rel, dst): Tensor[2, E]} — edge indices

        Returns:
            {node_type: Tensor[N, 256]} — contextualised embeddings
        """
        # Input projection.
        #
        # Empty node types must stay in x_dict as correctly-shaped zero rows.
        # HGTConv indexes every node type named in its metadata when it builds
        # the key/value tensors, so dropping a key raises KeyError on some other
        # type as its offset bookkeeping desynchronises. On this corpus the
        # `equation` type is always empty (MinerU formula recognition is off,
        # see R-13), so the old `if x.shape[0] > 0` filter broke every graph.
        #
        # The filter existed because node_lin is a PyG lazy `Linear(-1, ...)`,
        # which cannot infer its input width from a 0-row tensor. So project the
        # non-empty types normally, and emit zeros of the output width for the
        # empty ones without ever calling the lazy layer on them.
        projected = {}
        for nt, x in x_dict.items():
            if x.shape[0] > 0:
                projected[nt] = F.gelu(self.node_lin[nt](x))
            else:
                projected[nt] = x.new_zeros((0, self.hidden_channels))
        x_dict = projected

        # HGT message passing.
        #
        # HGTConv returns None for any node type that receives no message, and
        # omits nothing else — but iterating `x_dict_new` alone silently drops
        # those types from the running dict. `section` is only ever an edge
        # SOURCE (section -> hierarchy -> text) and never a destination, so it
        # receives no messages and disappeared after layer 1; layer 2 then hit
        # `KeyError: 'section'` when HGTConv looked up its source features.
        # (`equation` is likewise absent as a source on this corpus.)
        #
        # Iterate the incoming x_dict instead, and treat a None/missing conv
        # output as "no update" — i.e. carry the residual through unchanged, so
        # every node type survives all layers with its features intact.
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            updated = {}
            for nt, x_prev in x_dict.items():
                delta = x_dict_new.get(nt)
                x_out = x_prev if delta is None else delta + x_prev
                updated[nt] = F.dropout(
                    x_out, p=self.dropout, training=self.training,
                )
            x_dict = updated

        # Layer norm
        x_dict = {
            nt: self.norms[nt](x)
            for nt, x in x_dict.items()
            if nt in self.norms
        }

        return x_dict

    def encode_query(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Project a raw query embedding (E5 space) into the same space the
        node features enter the HGT, via the TRAINABLE text input layer
        (node_lin['text'] + GELU), then L2-normalise.

        This makes the query question-conditioned and trained jointly with
        the nodes (Phase 5 InfoNCE), replacing the old identity-normalise
        which silently assumed the query was already in node space.

        Note: node_lin is a lazy Linear(-1, hidden) — it is initialised by
        the first forward()/encode_nodes() call, which retrieval performs
        before encode_query(), so the layer exists when this runs.
        """
        # R-38: same device guard as encode_nodes. embed_query() returns a CPU
        # tensor (it goes through numpy), while this module lives on CUDA.
        dev = next(self.parameters()).device
        if query_emb.device != dev:
            query_emb = query_emb.to(dev)
        if "text" in self.node_lin:
            q = F.gelu(self.node_lin["text"](query_emb))
        else:
            q = query_emb
        return F.normalize(q, dim=-1)

    def encode_nodes(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Forward + L2-normalise all node embeddings."""
        # R-38: move inputs onto the module's own device. Callers differ —
        # Phase 7/9 load a graph with `load_graph()` and pass its CPU tensors
        # straight in, while the HGT itself was moved to CUDA. That raises
        # "Expected all tensors to be on the same device ... wrapper_CUDA_addmm"
        # at the first node_lin. Doing it here covers every call site (training,
        # reward tuning, self-healing retrieval, evaluation) instead of relying
        # on each one to remember.
        dev = next(self.parameters()).device
        x_dict = {nt: (x.to(dev) if x.device != dev else x)
                  for nt, x in x_dict.items()}
        edge_index_dict = {et: (ei.to(dev) if ei.device != dev else ei)
                           for et, ei in edge_index_dict.items()}

        x_filtered = {nt: x for nt, x in x_dict.items() if x.shape[0] > 0}
        edge_filtered = {
            et: ei for et, ei in edge_index_dict.items()
            if et[0] in x_filtered and et[2] in x_filtered and ei.shape[1] > 0
        }
        if not x_filtered:
            return {}
        # No edges after filtering (e.g. ablation zeroed all edge types) —
        # fall back to input projections only, skip message passing.
        if not edge_filtered:
            with torch.no_grad():
                out = {nt: F.normalize(F.gelu(self.node_lin[nt](x)), dim=-1)
                       for nt, x in x_filtered.items()}
            return out
        out = self.forward(x_filtered, edge_filtered)
        return {nt: F.normalize(x, dim=-1) for nt, x in out.items()}


# ─── InfoNCE contrastive loss ──────────────────────────────────────────────────

def info_nce_loss(
    q_emb: torch.Tensor,          # [D]  query embedding
    pos_emb: torch.Tensor,        # [D]  positive node embedding
    neg_embs: torch.Tensor,       # [K, D]  hard negative embeddings
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE (NT-Xent) contrastive loss.

    Args:
        q_emb:       L2-normalised query embedding, shape [D].
        pos_emb:     L2-normalised positive embedding, shape [D].
        neg_embs:    L2-normalised negative embeddings, shape [K, D].
        temperature: Logit scaling factor. 0.07 is standard.

    Returns:
        Scalar loss tensor.
    """
    # Positive similarity
    pos_sim = F.cosine_similarity(q_emb.unsqueeze(0), pos_emb.unsqueeze(0))  # [1]
    # Negative similarities
    neg_sims = F.cosine_similarity(q_emb.unsqueeze(0), neg_embs)             # [K]

    logits = torch.cat([pos_sim, neg_sims]) / temperature                    # [K+1]
    label = torch.zeros(1, dtype=torch.long, device=q_emb.device)            # positive at idx 0
    return F.cross_entropy(logits.unsqueeze(0), label)


# ─── Retrieval at inference time ──────────────────────────────────────────────

@torch.no_grad()
def retrieve_top_k(
    query_emb: torch.Tensor,          # [D]
    node_emb_dict: dict,              # {node_type: Tensor[N, D]}
    top_k: int = 10,
) -> list[dict]:
    """
    FAISS-free top-k retrieval by cosine similarity over all node types.

    Args:
        query_emb:     L2-normalised query embedding.
        node_emb_dict: Dict of contextualised node embeddings per type.
        top_k:         Number of top results to return.

    Returns:
        List of dicts: {node_type, local_idx, score}, sorted by score desc.
    """
    candidates = []
    q = F.normalize(query_emb.unsqueeze(0), dim=-1)  # [1, D]

    for node_type, embs in node_emb_dict.items():
        if embs.shape[0] == 0:
            continue
        # R-33: align device (and dtype) before comparing. Callers do not agree
        # on where the query lives: Phase 9 builds it with embed_query() on the
        # GPU, while Phase 7's reward tuning loads it straight off disk with
        # torch.load(...) — a CPU tensor — and node_emb_dict is always on the
        # device the HGT ran on. Mismatched devices raise
        # "Expected all tensors to be on the same device", which under headless
        # nbconvert surfaces only as an opaque "Kernel died".
        if q.device != embs.device or q.dtype != embs.dtype:
            q_cmp = q.to(device=embs.device, dtype=embs.dtype)
        else:
            q_cmp = q
        sims = F.cosine_similarity(q_cmp, embs)  # [N]
        for local_idx, score in enumerate(sims.tolist()):
            candidates.append({
                "node_type": node_type,
                "local_idx": local_idx,
                "score": score,
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]
