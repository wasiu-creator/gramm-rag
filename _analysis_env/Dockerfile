# Slim CPU-only environment for the $0 post-processing analyses (Phase C T-C0).
#
# Deliberately NOT the benchmark images: A1-A5 are pure post-processing of archived
# result JSONs. They need no GPU, no torch, no MinerU. Building the CUDA 12.4 +
# torch-cu124 + PyG + MinerU image to run them would write tens of GB to the boot
# volume (Docker data-root is /var/lib/docker on /dev/sda1 here) for no benefit, and
# would put the analyses on the GPU's critical path -- breaking the whole point of
# running Phase A analyses concurrently with the ColPali arm (R-32).
#
# The project's own scorer (src/evaluation/metrics.py -> anls_score) is pure stdlib,
# so it imports fine here; the repo is bind-mounted at runtime rather than copied.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
        numpy \
        pandas \
        matplotlib \
        nbconvert \
        nbformat \
        ipykernel \
        openai \
        scipy

# Headless: no display, and the notebook sets %matplotlib inline itself. Do NOT set
# MPLBACKEND=Agg -- doing that in the MP-DocVQA notebook silently suppressed every
# inline figure, which is why the joint notebook carries an explicit warning against it.
ENV MPLCONFIGDIR=/tmp/mpl

WORKDIR /work
