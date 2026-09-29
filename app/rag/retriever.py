"""
app/rag/retriever.py — Similarity search over ChromaDB.

This module is the bridge between a user's question and the relevant document
chunks.  It converts the question into a vector embedding, searches ChromaDB
for the closest stored chunk vectors, and returns the results formatted for
the LLM prompt and the frontend source display.

Interview explanation:
  This is the "Retrieval" part of Retrieval-Augmented Generation.

  Step-by-step:
    1. The user types a question: "What is AWS IAM?"
    2. The question is converted to a 384-dim vector using the same
       HuggingFace model that was used to embed the document chunks.
       (Using the same model is essential — the vectors must live in the
       same mathematical space to be comparable.)
    3. ChromaDB computes the cosine distance between the question vector
       and every stored chunk vector.
    4. The top-K closest chunks are returned — these are the parts of the
       document most likely to contain the answer.
    5. The chunks + metadata are returned to the LLM in Phase 8.

  Why cosine similarity?
    Cosine similarity measures the angle between two vectors rather than
    their magnitude.  This means a short question and a long paragraph
    can still match well if they discuss the same topic.

  Why top-K = 4?
    4 chunks give the LLM enough context to answer most questions without
    making the prompt too long.  Configurable via TOP_K env var.

Public API:
  retrieve(collection, question, embedding_model, top_k) -> list[RetrievedChunk]

  RetrievedChunk is a dataclass with:
    - text:        The chunk's text content
    - source:      Original PDF filename
    - page_number: 1-based page number
    - chunk_index: Position of this chunk within the document
    - distance:    Cosine distance from the question (lower = more similar)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single document chunk returned by similarity search.

    Using a dataclass (rather than a plain dict) gives us:
      - Named attributes with type hints — easier to read and maintain.
      - Clear contract: every caller knows exactly what fields to expect.
      - Easy to extend later (e.g. add a relevance_score field).

    Attributes:
        text:        The raw text of this chunk.
        source:      Original PDF filename (e.g. "report.pdf").
        page_number: 1-based page number for display to the user.
        chunk_index: Zero-based position of this chunk within its document.
        distance:    Cosine distance between question and chunk vectors.
                     Lower = more similar.  Range: [-1, 1] for cosine,
                     typically [0, 1] for normalised embeddings.
    """

    text: str
    source: str
    page_number: int
    chunk_index: int
    distance: float


# ── Core retrieval function ───────────────────────────────────────────────────

def retrieve(
    collection: chromadb.Collection,
    question: str,
    embedding_model: HuggingFaceEmbeddings,
    top_k: int = 4,
) -> List[RetrievedChunk]:
    """Find the most relevant document chunks for a question.

    Converts the question to an embedding, queries ChromaDB for the
    closest stored chunks, and returns them as RetrievedChunk objects.

    Args:
        collection:      An open ChromaDB Collection (already populated with
                         document chunks from vector_store.add_chunks).
        question:        The user's natural-language question.
        embedding_model: The loaded HuggingFace embedding model — must be the
                         same model used to embed the stored chunks.
        top_k:           Maximum number of chunks to return.  Defaults to 4.
                         Read from app config (TOP_K env var).

    Returns:
        List of RetrievedChunk objects, sorted by distance ascending
        (most relevant first).  May be shorter than top_k if the collection
        has fewer stored chunks.

    Raises:
        ValueError: If question is empty or top_k is not a positive integer.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")
    if top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    # ── Guard: empty collection ───────────────────────────────────────────────
    total_stored = collection.count()
    if total_stored == 0:
        logger.warning("Retrieval requested but collection is empty")
        return []

    # Clamp top_k to the number of stored chunks to avoid ChromaDB errors.
    # e.g. if only 2 chunks are stored and top_k=4, ChromaDB would raise.
    effective_top_k = min(top_k, total_stored)
    if effective_top_k < top_k:
        logger.debug(
            "top_k clamped from %d to %d (collection only has %d chunks)",
            top_k,
            effective_top_k,
            total_stored,
        )

    # ── Step 1: Embed the question ────────────────────────────────────────────
    logger.info("Embedding question for retrieval: '%s'", question[:80])
    question_vector = embedding_model.embed_query(question)

    # ── Step 2: Query ChromaDB ────────────────────────────────────────────────
    logger.info(
        "Querying ChromaDB for top-%d chunks (collection size: %d)",
        effective_top_k,
        total_stored,
    )
    result = collection.query(
        query_embeddings=[question_vector],
        n_results=effective_top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ── Step 3: Parse and format results ─────────────────────────────────────
    # ChromaDB returns nested lists because query() supports multiple
    # query_embeddings at once.  We sent one embedding, so we take index [0].
    ids        = result.get("ids", [[]])[0]
    documents  = result.get("documents", [[]])[0]
    metadatas  = result.get("metadatas", [[]])[0]
    distances  = result.get("distances", [[]])[0]

    if not ids:
        logger.info("No results returned by ChromaDB query")
        return []

    chunks: List[RetrievedChunk] = []
    for doc_text, meta, dist in zip(documents, metadatas, distances):
        chunk = RetrievedChunk(
            text=doc_text,
            source=meta.get("source", "unknown"),
            page_number=int(meta.get("page_number", 1)),
            chunk_index=int(meta.get("chunk_index", 0)),
            distance=float(dist),
        )
        chunks.append(chunk)

    logger.info(
        "Retrieved %d chunk(s); closest distance: %.4f",
        len(chunks),
        chunks[0].distance if chunks else float("inf"),
    )
    return chunks


# ── Convenience: deduplicate sources ─────────────────────────────────────────

def unique_sources(chunks: List[RetrievedChunk]) -> List[dict]:
    """Extract deduplicated source references from retrieved chunks.

    Used by the /api/ask route to build the 'sources' field in the response.
    Deduplicates by (filename, page_number) — if two chunks from the same
    page were retrieved, the page is listed only once.

    Args:
        chunks: List of RetrievedChunk objects from retrieve().

    Returns:
        List of dicts with 'filename' and 'page' keys, in the order they
        first appear in the chunks list (most relevant first).

    Example:
        [{"filename": "report.pdf", "page": 3},
         {"filename": "report.pdf", "page": 7}]
    """
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.source, chunk.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({"filename": chunk.source, "page": chunk.page_number})
    return sources
