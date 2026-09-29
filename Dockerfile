# Dockerfile — RAG Document Q&A Chatbot (Flask application only)
#
# Design decisions:
#   - Python 3.11 slim base: matches the project's minimum requirement and is
#     significantly smaller than the full image (~150 MB vs ~900 MB).
#   - Multi-stage NOT used: the heavy dependencies (torch, sentence-transformers)
#     are runtime dependencies — they cannot be pruned after install.
#   - Gunicorn: production-grade WSGI server; Flask's built-in server is
#     single-threaded and not suitable for production.
#   - Ollama is NOT included: the LLM runs externally (host or separate service).
#     OLLAMA_BASE_URL is passed in at runtime via environment variable.
#   - ChromaDB data is mounted as a volume so it persists across container
#     restarts and can be inspected from the host.
#   - Secrets are never baked into the image; all config comes from env vars.
#   - A non-root user is created for the application process (security best
#     practice — avoids running as UID 0 inside the container).
#
# Build:
#   docker build -t rag-chatbot .
#
# Run (Ollama on host):
#   docker run -p 5000:5000 \
#     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     -e SECRET_KEY=<your-secret> \
#     -v $(pwd)/uploads:/app/uploads \
#     -v $(pwd)/chroma_db:/app/chroma_db \
#     -v hf-cache:/app/.cache/huggingface \
#     rag-chatbot

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="rag-document-qa-chatbot"
LABEL description="RAG Document Q&A Chatbot — Flask + Gunicorn"

# ── System dependencies ───────────────────────────────────────────────────────
# gcc and g++ are required to compile some Python packages (e.g. chromadb's
# hnswlib bindings).  They are kept in the image because they may be needed
# at runtime by chromadb's native extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root application user ─────────────────────────────────────────────────
# Running as root inside a container is a security risk.  Create a dedicated
# user/group for the application process.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the requirements file first to leverage Docker layer caching.
# The heavy dependencies (torch, sentence-transformers) are only re-downloaded
# when requirements.txt changes — not on every code change.
COPY requirements.txt .

# Upgrade pip to avoid old-pip install quirks, then install all deps.
# --no-cache-dir reduces the image size by not storing the pip download cache.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source code ───────────────────────────────────────────────────
# Copy everything else (respects .dockerignore — excludes .env, chroma_db/,
# uploads/, tests/, .venv/, __pycache__, etc.)
COPY . .

# ── Runtime directories ───────────────────────────────────────────────────────
# Create the upload, ChromaDB, and HuggingFace model-cache directories and
# hand them to appuser so the application can write to them at runtime.
#
# HuggingFace downloads the embedding model on first run and caches it.
# Without a writable cache dir, the first request would fail with
# PermissionError because the default cache path is ~/. and no home dir
# exists for the non-root user (--no-create-home).
# We redirect the cache to /app/.cache/huggingface (inside the app tree,
# fully owned by appuser).  The same directory can be mounted as a volume
# to persist the model across container restarts:
#   -v hf-cache:/app/.cache/huggingface
RUN mkdir -p /app/uploads /app/chroma_db /app/.cache/huggingface \
    && chown -R appuser:appgroup /app

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Environment variable defaults ────────────────────────────────────────────
# These are safe, non-secret defaults.
# Override at container start time with -e or --env-file.
#
# FLASK_DEBUG=false — never run debug mode in the Docker image.
# CHROMA_DB_PATH    — absolute path inside the container (not relative).
# UPLOAD_FOLDER     — absolute path inside the container (not relative).
# OLLAMA_BASE_URL   — override to http://host.docker.internal:11434 on Docker
#                     Desktop, or to the Ollama service hostname in Docker
#                     Compose / EC2.
ENV FLASK_DEBUG=false \
    CHROMA_DB_PATH=/app/chroma_db \
    UPLOAD_FOLDER=/app/uploads \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /api/health stops responding.
# --interval: check every 30 s
# --timeout:  fail the check if no response within 10 s
# --start-period: allow 60 s for model loading before health checks start
# --retries: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" \
    || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
# Gunicorn configuration:
#   -w 2        — 2 worker processes (safe for RAM-constrained instances;
#                 each worker loads the full model, so don't over-provision)
#   --threads 2 — 2 threads per worker for I/O concurrency (PDF upload, Ollama
#                 HTTP calls are I/O-bound)
#   -b 0.0.0.0:5000 — bind to all interfaces inside the container
#   --timeout 120   — give Ollama LLM calls up to 120 s to complete
#   --access-logfile - — log access to stdout (captured by Docker)
#   --error-logfile - — log errors to stderr (captured by Docker)
#   "app:create_app()" — Flask app factory pattern
CMD ["gunicorn", \
     "-w", "2", \
     "--threads", "2", \
     "-b", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:create_app()"]
