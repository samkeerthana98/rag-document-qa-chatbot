"""
app/rag/embeddings.py — Local HuggingFace embedding model wrapper.

Embeddings are the core mechanism that makes semantic search possible.
This module wraps HuggingFace sentence-transformers so the rest of the
pipeline does not depend on any specific embedding library.

Interview explanation:
  An embedding is a list of numbers (a vector) that represents the meaning
  of a piece of text.  Texts with similar meanings produce vectors that are
  close together in high-dimensional space.  This is what allows ChromaDB
  to find document chunks relevant to a question — it compares the question's
  vector to each chunk's vector and returns the closest ones.

  all-MiniLM-L6-v2 produces 384-dimensional vectors.  It is:
    - Small (~80 MB download)
    - Fast on CPU
    - Good enough for document Q&A tasks
    - Runs entirely locally — no API key, no cost per call

  Keeping embeddings in a separate module means we can swap the model later
  (e.g. to a larger model or a different provider) by changing only this file.

Public API:
  get_embedding_model(model_name) -> HuggingFaceEmbeddings
      Returns a configured embedding model instance.
      The model is loaded once and should be reused across requests.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load and return the HuggingFace embedding model.

    The model is cached after the first load using @lru_cache.  This means:
      - The first call downloads the model (if not cached) and loads it into
        memory.  This can take 5–30 seconds.
      - All subsequent calls return the already-loaded model instantly.
      - The model stays in memory for the lifetime of the process.

    Why lru_cache?
      Loading a sentence-transformer model involves reading ~80 MB of weights
      from disk and initialising PyTorch.  Doing this on every request would
      make every upload/query slow.  Caching it in-process means the cost is
      paid only once at startup (or on first use).

    Args:
        model_name: HuggingFace model identifier, e.g.
                    "sentence-transformers/all-MiniLM-L6-v2".
                    Read from app config; passed as argument so the cache key
                    changes if the model name changes.

    Returns:
        A LangChain HuggingFaceEmbeddings instance ready for use.
    """
    logger.info("Loading embedding model: %s", model_name)
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        # Run on CPU by default.  If a GPU is present, sentence-transformers
        # will use it automatically.
        model_kwargs={"device": "cpu"},
        # Normalise embeddings so cosine similarity == dot product.
        # This is standard practice for similarity search.
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded: %s", model_name)
    return model
