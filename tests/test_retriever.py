"""
tests/test_retriever.py — Unit tests for app/rag/retriever.py.

Design:
  - Embedding model is MOCKED — returns deterministic fake 384-dim vectors.
    The mock lets us control which chunks appear "closer" to the question
    by assigning specific vector values to each piece of text.
  - ChromaDB runs for REAL using a tmp_path directory — tests the actual
    query logic, distance ordering, and metadata parsing.

Interview explanation:
  The retriever is the component that makes RAG intelligent.  By controlling
  the fake vectors in tests, we can verify:
    - Closer vectors (same value) rank first.
    - top_k limits are respected.
    - Metadata (filename, page_number) is correctly extracted.
    - Edge cases (empty collection, top_k > stored chunks) are handled.
"""

from dataclasses import fields
from unittest.mock import MagicMock

import chromadb
import pytest
from langchain_core.documents import Document

from app.rag.retriever import RetrievedChunk, retrieve, unique_sources
from app.rag.vector_store import add_chunks, get_collection


# ── Fixtures ──────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384


def make_vec(val: float) -> list:
    """Return a 384-dim vector filled with val (normalised unit-style)."""
    return [val] * EMBEDDING_DIM


def mock_model_factory(query_vec: list, doc_vecs: list) -> MagicMock:
    """Build a mock embedding model with controlled output.

    Args:
        query_vec: The vector to return for embed_query() calls.
        doc_vecs:  The list of vectors to return for embed_documents() calls.
    """
    mock = MagicMock()
    mock.embed_query.return_value = query_vec
    mock.embed_documents.side_effect = lambda texts: doc_vecs[: len(texts)]
    return mock


@pytest.fixture()
def chroma_col(tmp_path):
    """Fresh ChromaDB collection for each test."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return get_collection(client, "test_retriever")


def seed_collection(collection, mock_model, chunks):
    """Helper: add chunks to the collection using the mock model."""
    add_chunks(collection, chunks, mock_model, chunks[0].metadata["source"])


# ── RetrievedChunk dataclass ──────────────────────────────────────────────────

class TestRetrievedChunk:
    def test_has_required_fields(self):
        field_names = {f.name for f in fields(RetrievedChunk)}
        assert {"text", "source", "page_number", "chunk_index", "distance"} == field_names

    def test_can_be_constructed(self):
        chunk = RetrievedChunk(
            text="Hello", source="a.pdf", page_number=1, chunk_index=0, distance=0.1
        )
        assert chunk.text == "Hello"
        assert chunk.source == "a.pdf"
        assert chunk.page_number == 1
        assert chunk.distance == 0.1


# ── retrieve() — empty collection ────────────────────────────────────────────

class TestRetrieveEmptyCollection:
    def test_returns_empty_list(self, chroma_col):
        mock_model = MagicMock()
        mock_model.embed_query.return_value = make_vec(0.5)
        result = retrieve(chroma_col, "What is this?", mock_model, top_k=4)
        assert result == []

    def test_does_not_call_embed_on_empty_collection(self, chroma_col):
        mock_model = MagicMock()
        retrieve(chroma_col, "Any question", mock_model, top_k=4)
        mock_model.embed_query.assert_not_called()


# ── retrieve() — input validation ────────────────────────────────────────────

class TestRetrieveValidation:
    def test_empty_question_raises(self, chroma_col):
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="not be empty"):
            retrieve(chroma_col, "", mock_model, top_k=4)

    def test_whitespace_only_question_raises(self, chroma_col):
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="not be empty"):
            retrieve(chroma_col, "   ", mock_model, top_k=4)

    def test_zero_top_k_raises(self, chroma_col):
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="positive integer"):
            retrieve(chroma_col, "What?", mock_model, top_k=0)

    def test_negative_top_k_raises(self, chroma_col):
        mock_model = MagicMock()
        with pytest.raises(ValueError):
            retrieve(chroma_col, "What?", mock_model, top_k=-1)


# ── retrieve() — basic retrieval ─────────────────────────────────────────────

class TestRetrieveBasic:
    def _seed_two_chunks(self, collection):
        """Store 2 chunks with distinct vectors."""
        chunks = [
            Document(
                page_content="AWS IAM manages access control.",
                metadata={"source": "aws.pdf", "page": 0, "page_number": 1, "total_pages": 2},
            ),
            Document(
                page_content="S3 is an object storage service.",
                metadata={"source": "aws.pdf", "page": 1, "page_number": 2, "total_pages": 2},
            ),
        ]
        doc_vecs = [make_vec(0.1), make_vec(0.9)]
        store_model = mock_model_factory(make_vec(0.0), doc_vecs)
        add_chunks(collection, chunks, store_model, "aws.pdf")
        return chunks

    def test_returns_list_of_retrieved_chunks(self, chroma_col):
        self._seed_two_chunks(chroma_col)
        query_model = mock_model_factory(make_vec(0.1), [])
        result = retrieve(chroma_col, "What is IAM?", query_model, top_k=2)
        assert isinstance(result, list)
        assert all(isinstance(c, RetrievedChunk) for c in result)

    def test_returns_correct_count(self, chroma_col):
        self._seed_two_chunks(chroma_col)
        query_model = mock_model_factory(make_vec(0.1), [])
        result = retrieve(chroma_col, "What is IAM?", query_model, top_k=2)
        assert len(result) == 2

    def test_embed_query_called_once(self, chroma_col):
        self._seed_two_chunks(chroma_col)
        query_model = mock_model_factory(make_vec(0.1), [])
        retrieve(chroma_col, "What is IAM?", query_model, top_k=2)
        query_model.embed_query.assert_called_once_with("What is IAM?")

    def test_chunk_text_is_populated(self, chroma_col):
        self._seed_two_chunks(chroma_col)
        query_model = mock_model_factory(make_vec(0.1), [])
        result = retrieve(chroma_col, "What is IAM?", query_model, top_k=2)
        for chunk in result:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0

    def test_source_metadata_preserved(self, chroma_col):
        self._seed_two_chunks(chroma_col)
        query_model = mock_model_factory(make_vec(0.1), [])
        result = retrieve(chroma_col, "What is IAM?", query_model, top_k=2)
        for chunk in result:
            assert chunk.source == "aws.pdf"

    def test_page_number_metadata_preserved(self, chroma_col):
        self._seed_two_chunks(chroma_col)
        query_model = mock_model_factory(make_vec(0.1), [])
        result = retrieve(chroma_col, "What is IAM?", query_model, top_k=2)
        page_numbers = {c.page_number for c in result}
        assert page_numbers == {1, 2}

    def test_distance_is_float(self, chroma_col):
        self._seed_two_chunks(chroma_col)
        query_model = mock_model_factory(make_vec(0.1), [])
        result = retrieve(chroma_col, "What is IAM?", query_model, top_k=2)
        for chunk in result:
            assert isinstance(chunk.distance, float)

    def test_most_similar_chunk_returned_first(self, chroma_col):
        """Chunk with vector closest to the query vector should rank first.

        We use two orthogonal vectors (first-half vs second-half components)
        so their cosine distances are meaningfully different.
        Using [0.1]*384 and [0.9]*384 would NOT work because they are parallel
        (same direction, different magnitude) and cosine distance only measures
        angle — both would have the same distance to any query.
        """
        # v_a and v_b are approximately orthogonal: cosine(v_a, v_b) ≈ 0
        v_a = [1.0] * 192 + [0.0] * 192  # "A direction"
        v_b = [0.0] * 192 + [1.0] * 192  # "B direction" — orthogonal to v_a

        chunks = [
            Document(
                page_content="IAM chunk — should rank first.",
                metadata={"source": "test.pdf", "page": 0, "page_number": 1, "total_pages": 2},
            ),
            Document(
                page_content="S3 chunk — should rank second.",
                metadata={"source": "test.pdf", "page": 1, "page_number": 2, "total_pages": 2},
            ),
        ]

        # Store chunk 0 with v_a, chunk 1 with v_b
        store_model = mock_model_factory(make_vec(0.0), [v_a, v_b])
        add_chunks(chroma_col, chunks, store_model, "test.pdf")

        # Query with v_a — chunk 0 has distance ≈ 0, chunk 1 has distance ≈ 1
        query_model = mock_model_factory(v_a, [])
        result = retrieve(chroma_col, "Tell me about IAM", query_model, top_k=2)
        assert len(result) == 2
        assert result[0].text == "IAM chunk — should rank first."
        assert result[1].text == "S3 chunk — should rank second."


# ── retrieve() — top_k clamping ───────────────────────────────────────────────

class TestRetrieveTopKClamping:
    def test_top_k_larger_than_stored_clamped(self, chroma_col):
        """Requesting more chunks than stored should return all stored chunks."""
        chunk = [Document(
            page_content="Only one chunk in this collection.",
            metadata={"source": "one.pdf", "page": 0, "page_number": 1, "total_pages": 1},
        )]
        store_model = mock_model_factory(make_vec(0.5), [make_vec(0.5)])
        add_chunks(chroma_col, chunk, store_model, "one.pdf")

        query_model = mock_model_factory(make_vec(0.5), [])
        result = retrieve(chroma_col, "Any question", query_model, top_k=10)
        assert len(result) == 1  # not 10 — clamped to collection size

    def test_top_k_1_returns_single_chunk(self, chroma_col):
        chunks = [
            Document(page_content="A", metadata={"source": "f.pdf", "page": 0, "page_number": 1, "total_pages": 3}),
            Document(page_content="B", metadata={"source": "f.pdf", "page": 1, "page_number": 2, "total_pages": 3}),
            Document(page_content="C", metadata={"source": "f.pdf", "page": 2, "page_number": 3, "total_pages": 3}),
        ]
        store_model = mock_model_factory(make_vec(0.0), [make_vec(0.1), make_vec(0.5), make_vec(0.9)])
        add_chunks(chroma_col, chunks, store_model, "f.pdf")

        query_model = mock_model_factory(make_vec(0.1), [])
        result = retrieve(chroma_col, "Question", query_model, top_k=1)
        assert len(result) == 1


# ── unique_sources() ──────────────────────────────────────────────────────────

class TestUniqueSources:
    def _make_chunks(self, specs):
        """Build RetrievedChunk list from (source, page_number) tuples."""
        return [
            RetrievedChunk(
                text=f"chunk text {i}",
                source=src,
                page_number=pg,
                chunk_index=i,
                distance=0.1 * i,
            )
            for i, (src, pg) in enumerate(specs)
        ]

    def test_empty_input(self):
        assert unique_sources([]) == []

    def test_single_chunk(self):
        chunks = self._make_chunks([("doc.pdf", 3)])
        result = unique_sources(chunks)
        assert result == [{"filename": "doc.pdf", "page": 3}]

    def test_deduplicates_same_page(self):
        """Two chunks from the same page should appear only once."""
        chunks = self._make_chunks([("doc.pdf", 1), ("doc.pdf", 1)])
        result = unique_sources(chunks)
        assert result == [{"filename": "doc.pdf", "page": 1}]

    def test_different_pages_both_included(self):
        chunks = self._make_chunks([("doc.pdf", 1), ("doc.pdf", 2)])
        result = unique_sources(chunks)
        assert len(result) == 2
        assert {"filename": "doc.pdf", "page": 1} in result
        assert {"filename": "doc.pdf", "page": 2} in result

    def test_multiple_documents(self):
        chunks = self._make_chunks([("a.pdf", 1), ("b.pdf", 1)])
        result = unique_sources(chunks)
        assert len(result) == 2

    def test_order_preserved(self):
        """Sources should appear in the order chunks were retrieved (best match first)."""
        chunks = self._make_chunks([("first.pdf", 5), ("second.pdf", 2)])
        result = unique_sources(chunks)
        assert result[0]["filename"] == "first.pdf"
        assert result[1]["filename"] == "second.pdf"

    def test_result_has_filename_and_page_keys(self):
        chunks = self._make_chunks([("doc.pdf", 1)])
        result = unique_sources(chunks)
        assert "filename" in result[0]
        assert "page" in result[0]
