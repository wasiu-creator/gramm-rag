# GraMM-RAG — MMLongBench-Doc end-to-end pipeline
# Base: CUDA 12.4 + Python 3.11 (Lambda-compatible)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HOME=/app/.cache/huggingface

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        git wget curl build-essential libgl1 libglib2.0-0 \
        poppler-utils tesseract-ocr pandoc \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch (CUDA 12.4) ───────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# ── PyTorch Geometric ─────────────────────────────────────────────────────────
# torch-scatter/torch-sparse are intentionally NOT installed: the codebase only
# uses pure torch_geometric APIs (HGTConv, Linear, HeteroData), and modern PyG
# (>=2.5) implements these with native torch scatter ops. The separate compiled
# packages have no prebuilt wheel for this torch+cu124 combo and would fall back
# to a source build that fails under PEP 517 build isolation (no torch visible).
RUN pip install torch-geometric

# ── Copy and install project requirements ─────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ── spaCy model ───────────────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_lg

# ── Copy project (src/ at root, notebook, data JSON) ─────────────────────────
COPY src/                  /app/src/
COPY data/                 /app/data/
COPY gen_notebook.py       /app/gen_notebook.py
COPY mmlongbench_e2e.ipynb /app/mmlongbench_e2e.ipynb
COPY download_pdfs.py      /app/download_pdfs.py

# ── Create directory structure (PDFs + artefacts mounted as volumes) ──────────
RUN mkdir -p \
    /app/data/mmlongbench/pdfs \
    /app/parsed \
    /app/embeddings \
    /app/graphs \
    /app/results/models/hgt_mmlb \
    /app/results/models/router_mmlb \
    /app/results/figures \
    /app/.cache/huggingface

WORKDIR /app

# ── Jupyter Lab on port 8888 ─────────────────────────────────────────────────
EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
