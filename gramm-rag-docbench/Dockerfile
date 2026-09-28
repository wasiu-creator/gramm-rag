# GraMM-RAG — DocBench end-to-end pipeline
# Base: CUDA 12.4 + Python 3.11 (JetStream2 / Lambda-compatible)
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
# D-8 (R-1): torch-scatter/torch-sparse dropped -- no wheel for torch 2.6+cu124,
# and the codebase only uses pure torch_geometric APIs.
RUN pip install torch-geometric

# ── Copy and install project requirements ─────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ── spaCy model ───────────────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_lg

# ── Copy project (flat layout: src/ at root alongside notebook) ───────────────
COPY src/               /app/src/
COPY gen_notebook.py    /app/gen_notebook.py
COPY docbench_e2e.ipynb /app/docbench_e2e.ipynb

# ── Create directory structure (data + artefacts mounted as volumes) ──────────
RUN mkdir -p \
    /app/data/docbench/data \
    /app/parsed \
    /app/embeddings \
    /app/graphs \
    /app/results/models/hgt_docbench \
    /app/results/models/router_docbench \
    /app/results/figures \
    /app/results/page_images \
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
