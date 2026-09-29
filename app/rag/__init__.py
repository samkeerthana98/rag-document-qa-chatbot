"""
app/rag/ — RAG (Retrieval-Augmented Generation) pipeline components.

This package contains all the building blocks of the RAG pipeline:

  document_processor.py  — PDF loading and text chunking
  embeddings.py          — Local HuggingFace embedding model
  vector_store.py        — ChromaDB storage and search
  retriever.py           — Question → similarity search → relevant chunks
  llm.py                 — Ollama / LangChain LLM integration

Each module is intentionally small and focused on one responsibility so that:
  - Each part can be tested independently.
  - Any component can be swapped without rewriting the whole pipeline.
    (e.g. replace all-MiniLM-L6-v2 with a different embedding model by only
    changing embeddings.py)

These modules will be implemented in Phases 4–8.
"""
