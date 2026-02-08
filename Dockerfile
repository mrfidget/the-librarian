# ---------------------------------------------------------------------------
# Stage 1: Builder — installs everything, downloads model weights
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    libmagic-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Set explicit cache directories for model downloads ---
ENV HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf_cache/sentence_transformers

RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $SENTENCE_TRANSFORMERS_HOME

# --- 1. Download sqlite-vec extension ---
# Latest release as of writing. Check https://github.com/asg017/sqlite-vec/releases for updates.
RUN mkdir -p /opt/sqlite-extensions && \
    curl -L https://github.com/asg017/sqlite-vec/releases/download/v0.1.1/sqlite-vec-0.1.1-loadable-linux-x86_64.tar.gz \
    -o /tmp/sqlite-vec.tar.gz && \
    tar -xzf /tmp/sqlite-vec.tar.gz -C /opt/sqlite-extensions && \
    rm /tmp/sqlite-vec.tar.gz

# --- 2. Install torch CPU-only first (pins the index for this package) ---
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

# --- 3. Install sentence-transformers (picks up the CPU torch already present) ---
RUN pip install --no-cache-dir sentence-transformers

# --- 4. Install everything else from the standard index ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- 5. Pre-download model weights so the image is self-contained ---
RUN python -c "from transformers import CLIPModel, CLIPProcessor; from sentence_transformers import SentenceTransformer; print('Downloading CLIP model...'); CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); print('Downloading MiniLM embedding model...'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Models cached.')"

# ---------------------------------------------------------------------------
# Stage 2: Runtime — only what is needed to run
# ---------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv (includes Python packages)
COPY --from=builder /opt/venv /opt/venv

# Copy sqlite-vec extension
COPY --from=builder /opt/sqlite-extensions /opt/sqlite-extensions

# Copy HuggingFace model cache (this is the critical part for offline operation)
COPY --from=builder /opt/hf_cache /opt/hf_cache

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SQLITE_VEC_PATH=/opt/sqlite-extensions/vec0 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf_cache/sentence_transformers \
    HF_HUB_OFFLINE=1

WORKDIR /app

RUN mkdir -p /data/staging /data/database /data/backups /data/library/text /data/library/images /data/library/pdfs

ENV DATABASE_PATH=/data/database/metadata.db \
    VECTOR_DB_PATH=/data/database/vectors.db \
    STAGING_PATH=/data/staging \
    BACKUP_PATH=/data/backups \
    LIBRARY_PATH=/data/library

# Copy application code LAST so code changes don't invalidate earlier layers
COPY . /app/

CMD ["python", "main.py", "--help"]