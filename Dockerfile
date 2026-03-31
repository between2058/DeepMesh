# =============================================================================
# DeepMesh API — Docker Image
#
# Target hardware : NVIDIA RTX Pro 6000 (Blackwell, sm_120)
# CUDA toolkit    : 12.8.1
# cuDNN           : 9
# Python          : 3.10
# PyTorch         : 2.7.1 + cu128
#
# Build:
#   docker build -t deepmesh:latest .
#
# Run (quick test):
#   docker run --gpus all -p 8193:8193 deepmesh:latest
#
# NOTE: Model weights must be mounted or pre-downloaded.
#       See docker-compose.yml for volume mapping.
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# -- Build-time arguments ------------------------------------------------------
# sm_120 = RTX Pro 6000 (Blackwell). Add more archs for multi-GPU compat:
# "8.0;8.6;8.9;9.0;10.0;12.0"
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

# Parallel compile jobs for ninja. flash-attn is capped separately at 4.
ARG MAX_JOBS=4

# -- Proxy (build-time + runtime) ----------------------------------------------
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy="localhost,127.0.0.1"

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    no_proxy=${no_proxy} \
    NO_PROXY=${no_proxy}

# -- Environment variables -----------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache \
    HUGGINGFACE_HUB_CACHE=/hf_cache

# -- System packages -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -- Python 3.10 as default interpreter ----------------------------------------
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

# -- Ensure /usr/local/cuda-12.8 exists (CUDA_HOME points here) ----------------
RUN test -d /usr/local/cuda-12.8 \
 || ln -sf /usr/local/cuda /usr/local/cuda-12.8

WORKDIR /app

# =============================================================================
# STEP 1 — PyTorch 2.7.1 + CUDA 12.8
#
# Exact versions confirmed working on RTX Pro 6000 (sm_120).
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# STEP 2 — Pure-Python / non-CUDA packages
# =============================================================================
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# =============================================================================
# STEP 3 — Lock torch back to 2.7.1 cu128 (ABI finalise point)
#
# requirements-api.txt deps may downgrade torch to the CPU PyPI version.
# Force-reinstall here so every CUDA extension below compiles and runs
# against the exact same torch ABI.
# =============================================================================
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir "numpy<2.0"

# =============================================================================
# STEP 4 — Clone flash-attention source
#
# flash-attn, rotary-emb, and dropout-layer-norm are all built from
# the flash-attention repository. rotary-emb and dropout-layer-norm
# are NOT on PyPI — they live in csrc/ and must be compiled locally.
# =============================================================================
RUN git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention

# -- 4a: flash-attn main package ----------------------------------------------
RUN cd /tmp/flash-attention \
 && MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation .

# -- 4b: rotary-emb (csrc/rotary) ---------------------------------------------
#    Required by lit_gpt/fused_rotary_embedding.py
RUN cd /tmp/flash-attention/csrc/rotary \
 && pip install --no-cache-dir --no-build-isolation .

# -- 4c: dropout-layer-norm (csrc/layer_norm) ----------------------------------
#    Required by lit_gpt/rmsnorm.py (FusedRMSNorm)
RUN cd /tmp/flash-attention/csrc/layer_norm \
 && pip install --no-cache-dir --no-build-isolation .

# -- 4d: cleanup source -------------------------------------------------------
RUN rm -rf /tmp/flash-attention

# =============================================================================
# STEP 5 — xformers (built from source for sm_120)
#
# Required by DeepMesh's LLaMAMLP (xformers.ops.SwiGLU).
# =============================================================================
RUN pip install --no-cache-dir --no-build-isolation \
    xformers

# =============================================================================
# STEP 6 — triton (for fused kernels)
# =============================================================================
RUN pip install --no-cache-dir triton

# =============================================================================
# Application source
# =============================================================================
COPY lit_gpt/        /app/lit_gpt/
COPY miche/          /app/miche/
COPY sft/            /app/sft/
COPY deepmesh_api.py /app/deepmesh_api.py

RUN mkdir -p /hf_cache /app/outputs /app/logs /app/weights

# -- Port ----------------------------------------------------------------------
EXPOSE 8193

# -- Health check --------------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=600s \
    --retries=5 \
    CMD curl -f http://localhost:8193/health || exit 1

# -- Entrypoint ----------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "deepmesh_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8193", \
     "--workers", "1", \
     "--log-level", "info"]
