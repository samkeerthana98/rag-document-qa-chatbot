"""
app/rag/vector_store.py — ChromaDB persistent vector store.

This module manages storing and retrieving document chunks in ChromaDB.

Interview explanation:
  A vector database stores embeddings (numerical vectors) alongside the
  original text and metadata.  When a user asks a question, the question is
  also converted to a vector and the database returns the stored chunks whose
  vectors are closest to the question vector.  This is called similarity search
  or nearest-neighbour search.

  ChromaDB is an embedded vector database — it runs inside the same Python
  process as Flask and persists data to a local directory.  There is no
  separate server to manage, which keeps the architecture simple.

  Why upsert instead of insert?
    If a user uploads the same PDF twice, we don't want duplicate chunks.
    Upsert means "insert if new, replace if the ID already exists".  We
    generate deterministic IDs from the filename + chunk index, so re-uploading
    the same document overwrites the previous chunks cleanly.

Public API:
  get_vector_store(db_path, collection_name, embedding_model) -> VectorStore
  add_chunks(vector_store, chunks, filename) -> int
  get_document_count(vector_store) -> int
  delete_document(vector_store, filename) -> None
"""

import hashlib
import logging
from typing import List

import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


# ── ChromaDB client and collection ────────────────────────────────────────────

def get_chroma_client(db_path: str) -> chromadb.PersistentClient:
    """Create or open a persistent ChromaDB client.

    The client manages the connection to the on-disk database.
    Calling this with the same path always opens the same database.

    Args:
        db_path: Absolute path to the directory where ChromaDB stores data.
                 The directory is created automatically if it doesn't exist.

    Returns:
        A ChromaDB PersistentClient connected to db_path.
    """
    logger.debug("Opening ChromaDB at '%s'", db_path)
    return chromadb.PersistentClient(path=db_path)


def get_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
) -> chromadb.Collection:
    """Get or create a named ChromaDB collection.

    A collection is like a table in a relational database — it groups related
    embeddings together.  We use one collection for all documents.

    get_or_create_collection is idempotent: calling it multiple times with the
    same name always returns the same collection without error.

    Args:
        client:          An open ChromaDB PersistentClient.
        collection_name: Name for the collection (from app config).

    Returns:
        A ChromaDB Collection ready for upsert/query.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        # cosine distance is standard for normalised sentence embeddings.
        # ChromaDB 1.5.x uses "hnsw:space" in metadata to set the distance.
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(
        "Collection '%s' opened (%d chunks stored)", collection_name, collection.count()
    )
    return collection


# ── Chunk ID generation ───────────────────────────────────────────────────────

def _make_chunk_id(filename: str, chunk_index: int) -> str:
    """Generate a deterministic, unique ID for a document chunk.

    The ID is built from the filename and the chunk's position within the
    document.  Using deterministic IDs means that re-uploading the same file
    always produces the same IDs, so upsert replaces old chunks rather than
    creating duplicates.

    We use an MD5 hash of the filename to keep IDs short and safe for use as
    ChromaDB string keys.  MD5 is fine here — we're not using it for security,
    just for a stable short identifier.

    Args:
        filename:    The sanitised filename (e.g. "report.pdf").
        chunk_index: Zero-based position of the chunk within the document.

    Returns:
        A string like "a1b2c3d4e5f6...7890_0" (hash_chunkindex).
    """
    name_hash = hashlib.md5(filename.encode()).hexdigest()
    return f"{name_hash}_{chunk_index}"


# ── Store chunks ──────────────────────────────────────────────────────────────

def add_chunks(
    collection: chromadb.Collection,
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    filename: str,
) -> int:
    """Embed document chunks and store them in ChromaDB.

    Steps:
      1. Delete any existing chunks for this filename (deduplication).
      2. Generate embeddings for all chunk texts.
      3. Upsert chunks with their embeddings and metadata.

    Why delete-then-upsert rather than pure upsert?
      Pure upsert by ID handles re-uploads of the same number of chunks well,
      but if a re-upload produces fewer chunks (e.g. after editing the PDF),
      the old extra chunks would remain.  Deleting first guarantees a clean
      slate for this document without affecting other documents.

    Args:
        collection:      An open ChromaDB Collection.
        chunks:          List of LangChain Documents from document_processor.
        embedding_model: The loaded HuggingFace embedding model.
        filename:        The sanitised filename — used as the deletion key.

    Returns:
        The number of chunks stored.
    """
    if not chunks:
        logger.warning("add_chunks called with empty chunk list for '%s'", filename)
        return 0

    # ── Step 1: Delete existing chunks for this document ─────────────────────
    # This prevents duplicates when the same document is re-uploaded.
    existing = collection.get(where={"source": filename})
    if existing and existing.get("ids"):
        count_before = len(existing["ids"])
        collection.delete(where={"source": filename})
        logger.info(
            "Deleted %d existing chunks for '%s' before re-indexing",
            count_before,
            filename,
        )

    # ── Step 2: Generate embeddings ───────────────────────────────────────────
    texts = [chunk.page_content for chunk in chunks]
    logger.info(
        "Generating embeddings for %d chunks from '%s'...", len(texts), filename
    )
    embeddings = embedding_model.embed_documents(texts)
    logger.info("Embeddings generated (%d x %d)", len(embeddings), len(embeddings[0]))

    # ── Step 3: Build IDs and metadata lists ─────────────────────────────────
    ids = [_make_chunk_id(filename, i) for i in range(len(chunks))]

    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = {
            # Always store these fields so retrieval can surface them.
            "source": chunk.metadata.get("source", filename),
            "page_number": int(chunk.metadata.get("page_number", 1)),
            "page": int(chunk.metadata.get("page", 0)),
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        metadatas.append(meta)

    # ── Step 4: Upsert into ChromaDB ─────────────────────────────────────────
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    stored = collection.count()
    logger.info(
        "Stored %d chunks for '%s'. Collection total: %d",
        len(chunks),
        filename,
        stored,
    )
    return len(chunks)


# ── Utility ───────────────────────────────────────────────────────────────────

def get_document_count(collection: chromadb.Collection) -> int:
    """Return the total number of chunks stored in the collection.

    Args:
        collection: An open ChromaDB Collection.

    Returns:
        Total chunk count across all documents.
    """
    return collection.count()


def delete_document(collection: chromadb.Collection, filename: str) -> int:
    """Remove all chunks belonging to a specific document.

    Args:
        collection: An open ChromaDB Collection.
        filename:   The sanitised filename to remove (e.g. "report.pdf").

    Returns:
        Number of chunks deleted.
    """
    existing = collection.get(where={"source": filename})
    if not existing or not existing.get("ids"):
        logger.info("No chunks found for '%s' — nothing to delete", filename)
        return 0

    count = len(existing["ids"])
    collection.delete(where={"source": filename})
    logger.info("Deleted %d chunks for '%s'", count, filename)
    return count


def list_documents(collection: chromadb.Collection) -> List[str]:
    """Return a deduplicated list of filenames stored in the collection.

    Useful for the health check or a future 'list documents' endpoint.

    Args:
        collection: An open ChromaDB Collection.

    Returns:
        Sorted list of unique source filenames.
    """
    if collection.count() == 0:
        return []

    result = collection.get(include=["metadatas"])
    sources = {
        meta["source"]
        for meta in result.get("metadatas", [])
        if meta and "source" in meta
    }
    return sorted(sources)
