# Multi-stage Dockerfile for Hierarchical Reasoning Model
# Supports both FlashAttention 2 (Ampere GPUs) and FlashAttention 3 (Hopper GPUs)

# Build argument for FlashAttention version
ARG FLASH_ATTN_VERSION=2

# ============================================================================
# Stage 1: Base Image with CUDA and System Dependencies
# ============================================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.6 \
    PATH=/usr/local/cuda-12.6/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH \
    PYTHONUNBUFFERED=1 \
    PYTHON_VERSION=3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# ============================================================================
# Stage 2: Install uv and Python Dependencies
# ============================================================================
FROM base AS builder

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Create src directory structure (required for build)
RUN mkdir -p src/hierarchical_reasoning_model && \
    touch src/hierarchical_reasoning_model/__init__.py

# Install core dependencies
RUN uv sync --frozen --no-dev

# Install PyTorch with CUDA 12.6 support
RUN uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install FlashAttention based on version
ARG FLASH_ATTN_VERSION
RUN if [ "$FLASH_ATTN_VERSION" = "3" ]; then \
        echo "Installing FlashAttention 3 for Hopper GPUs" && \
        git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
        cd /tmp/flash-attention/hopper && \
        uv pip install . && \
        rm -rf /tmp/flash-attention; \
    else \
        echo "Installing FlashAttention 2 for Ampere GPUs" && \
        uv pip install flash-attn --no-build-isolation; \
    fi

# Install training dependencies
RUN uv sync --frozen --extra train --extra data

# ============================================================================
# Stage 3: Final Runtime Image
# ============================================================================
FROM base AS runtime

# Copy uv from builder
COPY --from=builder /uv /uvx /bin/

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/
COPY pyproject.toml README.md LICENSE ./

# Install the package in editable mode
RUN . /app/.venv/bin/activate && \
    uv pip install -e .

# Set environment variables for runtime
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src:$PYTHONPATH

# Create directories for data and checkpoints
RUN mkdir -p /app/data /app/checkpoints /app/wandb

# Set default command
CMD ["python", "-c", "import hierarchical_reasoning_model; print('HRM Docker image ready!')"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
