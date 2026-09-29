"""
app/config.py — Application configuration.

All settings are read from environment variables so that:
  - No secrets are hard-coded in source code.
  - The same codebase runs in development, Docker, and AWS without changes.
  - Settings can be tuned without touching application logic.

Load a .env file at startup (see run.py / create_app) and these values
will be picked up automatically during local development.

Interview explanation:
  Separating configuration from code is a standard best practice.
  In production you would inject these as environment variables (e.g. via
  AWS Secrets Manager, Docker --env-file, or EC2 user-data) rather than
  committing a real .env file to source control.
"""

import os
from pathlib import Path


# ── Base paths ────────────────────────────────────────────────────────────────

# The root of the project (one level above this file).
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Central configuration class.

    All attributes read from environment variables with sensible defaults
    that work out-of-the-box for local development.
    """

    # ── Flask ─────────────────────────────────────────────────────────────────

    # SECRET_KEY is used by Flask to sign session cookies.
    # Must be set to a long random string in production.
    # Never commit a real value to source control.
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # Show detailed error pages in development; disable in production.
    DEBUG: bool = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    # ── File upload ───────────────────────────────────────────────────────────

    # Directory where uploaded PDF files are saved.
    UPLOAD_FOLDER: str = os.getenv(
        "UPLOAD_FOLDER", str(BASE_DIR / "uploads")
    )

    # Maximum allowed upload size in bytes (default: 50 MB).
    # Flask rejects requests larger than this before they reach route code.
    MAX_CONTENT_LENGTH: int = int(
        os.getenv("MAX_UPLOAD_SIZE_MB", "50")
    ) * 1024 * 1024

    # Only these file extensions are accepted.
    ALLOWED_EXTENSIONS: set = {"pdf"}

    # ── Vector database ───────────────────────────────────────────────────────

    # Directory where ChromaDB persists its data on disk.
    # Change this to an absolute path when running inside Docker.
    CHROMA_DB_PATH: str = os.getenv(
        "CHROMA_DB_PATH", str(BASE_DIR / "chroma_db")
    )

    # Name of the ChromaDB collection that stores document chunks.
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "documents"
    )

    # ── Embedding model ───────────────────────────────────────────────────────

    # HuggingFace model used to convert text into vector embeddings.
    # Runs entirely locally — no API key required.
    # all-MiniLM-L6-v2 is small (80 MB), fast, and produces 384-dim vectors.
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Document chunking ─────────────────────────────────────────────────────

    # chunk_size: approximate character count per chunk.
    # 1000 chars ≈ 200 words, which is enough context for a focused answer
    # and fits within all-MiniLM-L6-v2's 256-token limit.
    # Adjust upward for more context, downward for faster retrieval.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))

    # chunk_overlap: characters shared between consecutive chunks.
    # Prevents answers from being split across chunk boundaries.
    # 200 chars (20% of chunk_size) is a common starting point.
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    # Number of document chunks retrieved per query (top-K similarity search).
    # 4 gives enough context for most questions without overloading the LLM
    # prompt.  Increase if answers are incomplete; decrease if too slow.
    TOP_K: int = int(os.getenv("TOP_K", "4"))

    # ── LLM ───────────────────────────────────────────────────────────────────

    # Which LLM provider to use.  Only "ollama" is implemented in v1.
    # Adding a new provider later means adding a branch in app/rag/llm.py
    # without changing anything else.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama model name.  Must be pulled locally before use:
    #   ollama pull llama3.2
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Base URL of the locally running Ollama HTTP service.
    # When running Flask in Docker, change this to http://host.docker.internal:11434
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
