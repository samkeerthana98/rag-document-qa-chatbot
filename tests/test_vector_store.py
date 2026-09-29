"""
tests/test_vector_store.py — Unit tests for app/rag/vector_store.py.

Design decisions:
  - The HuggingFace embedding model is MOCKED — we don't load 80 MB of weights
    in a unit test.  The mock returns deterministic fake vectors (lists of
    floats) with the correct shape (384 dimensions).
  - ChromaDB runs for REAL with a temporary in-process directory provided by
    pytest's tmp_path fixture.  This tests the actual storage and retrieval
    logic against a live (but ephemeral) database.

Interview explanation:
  Mocking the embedding model is correct here because:
  1. The model has no logic we need to test — it's a library call.
  2. Loading it is slow and requires network/disk on first use.
  3. We only care that our code calls it correctly and passes the result to
     ChromaDB.  The mock verifies the call happened; the real ChromaDB
     verifies the storage worked.
"""

import hashlib
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.documents import Document

from app.rag.vector_store import (
    _make_chunk_id,
    add_chunks,
    delete_document,
    get_chroma_client,
    get_collection,
    get_document_count,
    list_documents,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors


def make_fake_embedding(seed: int = 0) -> list:
    """Return a deterministic fake 384-dim unit vector for testing."""
    val = (seed % 100 + 1) / 100.0  # e.g. 0.01, 0.02 ... avoids zero vector
    return [val] * EMBEDDING_DIM


def make_mock_embedding_model(n_docs: int = 1) -> MagicMock:
    """Return a mock embedding model whose embed_documents returns fake vectors."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [
        make_fake_embedding(i) for i in range(len(texts))
    ]
    mock.embed_query.side_effect = lambda text: make_fake_embedding(0)
    return mock


@pytest.fixture()
def chroma_client(tmp_path):
    """Real ChromaDB PersistentClient pointing at a temporary directory."""
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


@pytest.fixture()
def collection(chroma_client):
    """A fresh ChromaDB collection for each test."""
    return get_collection(chroma_client, "test_collection")


@pytest.fixture()
def mock_embedding_model():
    """Mock HuggingFaceEmbeddings that returns deterministic fake vectors."""
    return make_mock_embedding_model()


@pytest.fixture()
def sample_chunks():
    """A small list of LangChain Documents simulating chunked PDF output."""
    return [
        Document(
            page_content="The first chunk of text from the document.",
            metadata={"source": "test.pdf", "page": 0, "page_number": 1, "total_pages": 2},
        ),
        Document(
            page_content="The second chunk from page two of the document.",
            metadata={"source": "test.pdf", "page": 1, "page_number": 2, "total_pages": 2},
        ),
        Document(
            page_content="A third chunk with more content for testing.",
            metadata={"source": "test.pdf", "page": 1, "page_number": 2, "total_pages": 2},
        ),
    ]


# ── _make_chunk_id ────────────────────────────────────────────────────────────

class TestMakeChunkId:
    def test_returns_string(self):
        result = _make_chunk_id("test.pdf", 0)
        assert isinstance(result, str)

    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        assert _make_chunk_id("doc.pdf", 3) == _make_chunk_id("doc.pdf", 3)

    def test_different_index_different_id(self):
        assert _make_chunk_id("doc.pdf", 0) != _make_chunk_id("doc.pdf", 1)

    def test_different_filename_different_id(self):
        assert _make_chunk_id("a.pdf", 0) != _make_chunk_id("b.pdf", 0)

    def test_contains_index(self):
        chunk_id = _make_chunk_id("doc.pdf", 7)
        assert chunk_id.endswith("_7")

    def test_contains_hash(self):
        expected_hash = hashlib.md5("doc.pdf".encode()).hexdigest()
        assert _make_chunk_id("doc.pdf", 0).startswith(expected_hash)


# ── get_chroma_client ─────────────────────────────────────────────────────────

class TestGetChromaClient:
    def test_returns_persistent_client(self, tmp_path):
        # PersistentClient is a factory function in chromadb 1.5.x — it returns
        # chromadb.api.client.Client, not a PersistentClient type.
        # We verify the returned object has the collection API instead.
        client = get_chroma_client(str(tmp_path / "db"))
        assert hasattr(client, "get_or_create_collection")
        assert hasattr(client, "delete_collection")

    def test_creates_directory(self, tmp_path):
        db_path = str(tmp_path / "new_db")
        get_chroma_client(db_path)
        import os
        assert os.path.isdir(db_path)


# ── get_collection ────────────────────────────────────────────────────────────

class TestGetCollection:
    def test_returns_collection(self, chroma_client):
        col = get_collection(chroma_client, "mycollection")
        assert col is not None

    def test_idempotent(self, chroma_client):
        """Calling get_collection twice with the same name returns the same collection."""
        col1 = get_collection(chroma_client, "same_name")
        col2 = get_collection(chroma_client, "same_name")
        assert col1.name == col2.name

    def test_collection_starts_empty(self, chroma_client):
        col = get_collection(chroma_client, "empty_col")
        assert col.count() == 0

    def test_cosine_distance_configured(self, chroma_client):
        col = get_collection(chroma_client, "cosine_col")
        assert col.metadata.get("hnsw:space") == "cosine"


# ── add_chunks ────────────────────────────────────────────────────────────────

class TestAddChunks:
    def test_returns_chunk_count(self, collection, mock_embedding_model, sample_chunks):
        result = add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        assert result == len(sample_chunks)

    def test_chunks_stored_in_collection(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        assert collection.count() == len(sample_chunks)

    def test_embedding_model_called(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        mock_embedding_model.embed_documents.assert_called_once()

    def test_embedding_model_receives_all_texts(
        self, collection, mock_embedding_model, sample_chunks
    ):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        call_args = mock_embedding_model.embed_documents.call_args[0][0]
        assert len(call_args) == len(sample_chunks)
        for chunk in sample_chunks:
            assert chunk.page_content in call_args

    def test_metadata_stored(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        result = collection.get(include=["metadatas"])
        sources = {m["source"] for m in result["metadatas"]}
        assert sources == {"test.pdf"}

    def test_page_number_in_metadata(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        result = collection.get(include=["metadatas"])
        page_numbers = {m["page_number"] for m in result["metadatas"]}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_chunk_index_in_metadata(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        result = collection.get(include=["metadatas"])
        indices = {m["chunk_index"] for m in result["metadatas"]}
        assert indices == {0, 1, 2}

    def test_deduplication_on_reupload(self, collection, mock_embedding_model, sample_chunks):
        """Re-uploading the same document must not create duplicates."""
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        assert collection.count() == 3

        # Re-upload the same document
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        # Count must still be 3, not 6
        assert collection.count() == 3

    def test_reupload_with_fewer_chunks(self, collection, mock_embedding_model, sample_chunks):
        """Re-uploading with fewer chunks removes the old extra chunks."""
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        assert collection.count() == 3

        # Re-upload with only 1 chunk
        one_chunk = sample_chunks[:1]
        add_chunks(collection, one_chunk, mock_embedding_model, "test.pdf")
        assert collection.count() == 1

    def test_two_documents_independent(
        self, collection, mock_embedding_model, sample_chunks
    ):
        """Chunks from different documents coexist without interference."""
        other_chunks = [
            Document(
                page_content="Content from a different document.",
                metadata={"source": "other.pdf", "page": 0, "page_number": 1, "total_pages": 1},
            )
        ]
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        add_chunks(collection, other_chunks, mock_embedding_model, "other.pdf")
        assert collection.count() == len(sample_chunks) + len(other_chunks)

    def test_empty_chunks_returns_zero(self, collection, mock_embedding_model):
        result = add_chunks(collection, [], mock_embedding_model, "empty.pdf")
        assert result == 0
        assert collection.count() == 0


# ── get_document_count ────────────────────────────────────────────────────────

class TestGetDocumentCount:
    def test_empty_collection(self, collection):
        assert get_document_count(collection) == 0

    def test_count_after_add(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        assert get_document_count(collection) == len(sample_chunks)


# ── delete_document ───────────────────────────────────────────────────────────

class TestDeleteDocument:
    def test_delete_existing_document(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        assert collection.count() == 3

        deleted = delete_document(collection, "test.pdf")
        assert deleted == 3
        assert collection.count() == 0

    def test_delete_nonexistent_document(self, collection):
        """Deleting a document that doesn't exist should not raise."""
        deleted = delete_document(collection, "nonexistent.pdf")
        assert deleted == 0

    def test_delete_only_target_document(
        self, collection, mock_embedding_model, sample_chunks
    ):
        """Deleting one document must not affect other documents."""
        other = [
            Document(
                page_content="Keep this document.",
                metadata={"source": "keep.pdf", "page": 0, "page_number": 1, "total_pages": 1},
            )
        ]
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        add_chunks(collection, other, mock_embedding_model, "keep.pdf")
        assert collection.count() == 4

        delete_document(collection, "test.pdf")
        assert collection.count() == 1

        # Verify the remaining chunk belongs to keep.pdf
        remaining = collection.get(include=["metadatas"])
        assert remaining["metadatas"][0]["source"] == "keep.pdf"


# ── list_documents ────────────────────────────────────────────────────────────

class TestListDocuments:
    def test_empty_collection(self, collection):
        assert list_documents(collection) == []

    def test_single_document(self, collection, mock_embedding_model, sample_chunks):
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        docs = list_documents(collection)
        assert docs == ["test.pdf"]

    def test_multiple_documents_sorted(self, collection, mock_embedding_model):
        for name in ["zebra.pdf", "alpha.pdf", "mango.pdf"]:
            chunk = [Document(
                page_content=f"Content of {name}",
                metadata={"source": name, "page": 0, "page_number": 1, "total_pages": 1},
            )]
            add_chunks(collection, chunk, mock_embedding_model, name)

        docs = list_documents(collection)
        assert docs == ["alpha.pdf", "mango.pdf", "zebra.pdf"]

    def test_no_duplicates(self, collection, mock_embedding_model, sample_chunks):
        """Multiple chunks from the same document should appear once in the list."""
        add_chunks(collection, sample_chunks, mock_embedding_model, "test.pdf")
        docs = list_documents(collection)
        assert docs.count("test.pdf") == 1
