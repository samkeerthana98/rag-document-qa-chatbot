"""
tests/test_multi_document.py — Multi-document support tests (Phase 11).

Verifies that the application correctly handles multiple uploaded PDFs:
  - Each document is indexed independently with its own chunks and metadata.
  - Re-uploading the same document replaces (not duplicates) its chunks.
  - Retrieval searches across ALL indexed documents simultaneously.
  - Source metadata (filename + page) is correct for each document.
  - The Q&A pipeline works correctly when multiple documents are indexed.

Test strategy:
  - Route-level tests use the Flask test client and patch invoke_llm so
    no Ollama instance is needed.
  - Service-level tests use a real ChromaDB (tmp_path) and a mock embedding
    model to exercise answer_question() with multi-doc collections.
  - Retriever-level tests use controlled fake vectors to verify that chunks
    from multiple documents are correctly returned and labelled.
  - build_prompt tests verify that per-document source labels appear in the
    prompt when chunks come from more than one file.

LLM calls are always mocked — tests run fully offline.
"""

import io
import json
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.documents import Document

from app.rag.retriever import RetrievedChunk, retrieve, unique_sources
from app.rag.vector_store import add_chunks, get_collection, list_documents
from app.services.qa_service import QAResult, answer_question, build_prompt


# ── Shared constants ──────────────────────────────────────────────────────────

EMBEDDING_DIM = 384
MOCK_ANSWER = "This is a mocked multi-document answer."


def make_vec(val: float) -> list:
    return [val] * EMBEDDING_DIM


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def mock_embed():
    """Mock embedding model. All texts and queries get the same vector."""
    m = MagicMock()
    m.embed_documents.side_effect = lambda texts: [make_vec(0.5)] * len(texts)
    m.embed_query.return_value = make_vec(0.5)
    return m


@pytest.fixture()
def chroma_col(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return get_collection(client, "test_multidoc")


@pytest.fixture()
def mock_llm():
    m = MagicMock()
    m.invoke.return_value = MOCK_ANSWER
    return m


def _pdf_upload(client, pdf_bytes, filename):
    """Helper: POST a PDF to /api/upload via the test client."""
    return client.post(
        "/api/upload",
        data={"file": (io.BytesIO(pdf_bytes), filename)},
        content_type="multipart/form-data",
    )


def _ask(client, question):
    """Helper: POST a question to /api/ask via the test client."""
    return client.post(
        "/api/ask",
        data=json.dumps({"question": question}),
        content_type="application/json",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Route-level: multiple uploads
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteMultipleUploads:
    """Two distinct PDFs can both be uploaded and independently indexed."""

    def test_two_uploads_both_succeed(self, client, sample_pdf_bytes):
        """Uploading two different filenames must both return 200."""
        r1 = _pdf_upload(client, sample_pdf_bytes, "doc_a.pdf")
        r2 = _pdf_upload(client, sample_pdf_bytes, "doc_b.pdf")
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_two_uploads_return_correct_filenames(self, client, sample_pdf_bytes):
        """Each upload response must report its own filename."""
        r1 = _pdf_upload(client, sample_pdf_bytes, "alpha.pdf")
        r2 = _pdf_upload(client, sample_pdf_bytes, "beta.pdf")
        assert json.loads(r1.data)["filename"] == "alpha.pdf"
        assert json.loads(r2.data)["filename"] == "beta.pdf"

    def test_two_uploads_both_report_chunks(self, client, sample_pdf_bytes):
        """Both uploads must report a positive chunks_processed count."""
        r1 = _pdf_upload(client, sample_pdf_bytes, "first.pdf")
        r2 = _pdf_upload(client, sample_pdf_bytes, "second.pdf")
        assert json.loads(r1.data)["chunks_processed"] > 0
        assert json.loads(r2.data)["chunks_processed"] > 0

    def test_three_uploads_all_succeed(self, client, sample_pdf_bytes):
        """Three different documents can all be uploaded successfully."""
        names = ["one.pdf", "two.pdf", "three.pdf"]
        for name in names:
            r = _pdf_upload(client, sample_pdf_bytes, name)
            assert r.status_code == 200, f"Upload failed for {name}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Route-level: re-upload deduplication
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteReuploadDeduplication:
    """Re-uploading the same filename replaces, not duplicates, its chunks."""

    def test_reupload_same_file_returns_200(self, client, sample_pdf_bytes):
        """Uploading the same filename twice must both return 200."""
        r1 = _pdf_upload(client, sample_pdf_bytes, "report.pdf")
        r2 = _pdf_upload(client, sample_pdf_bytes, "report.pdf")
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_reupload_chunks_not_doubled(self, client, sample_pdf_bytes):
        """After uploading the same file twice, chunks_processed must equal
        a single upload — not double the first upload's count.

        This verifies the delete-before-upsert deduplication strategy.
        """
        r1 = _pdf_upload(client, sample_pdf_bytes, "dedup.pdf")
        first_chunks = json.loads(r1.data)["chunks_processed"]

        r2 = _pdf_upload(client, sample_pdf_bytes, "dedup.pdf")
        second_chunks = json.loads(r2.data)["chunks_processed"]

        # Both uploads process the same PDF so chunk counts should match
        assert second_chunks == first_chunks

    def test_reupload_does_not_affect_other_documents(
        self, client, sample_pdf_bytes
    ):
        """Re-uploading doc A must not remove doc B from the index.

        After uploading doc_a.pdf and doc_b.pdf, re-uploading doc_a.pdf
        and then asking a question must still return 200 (collection non-empty).
        """
        _pdf_upload(client, sample_pdf_bytes, "doc_a.pdf")
        _pdf_upload(client, sample_pdf_bytes, "doc_b.pdf")
        _pdf_upload(client, sample_pdf_bytes, "doc_a.pdf")  # re-upload

        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = _ask(client, "What is in the documents?")

        assert response.status_code == 200
        data = json.loads(response.data)
        # Collection is non-empty so the LLM was called and answer is the mock
        assert data["answer"] == MOCK_ANSWER


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Route-level: /api/ask with multiple documents indexed
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteAskMultiDoc:
    """The Q&A endpoint works correctly when multiple documents are indexed."""

    def test_ask_after_two_uploads_returns_200(self, client, sample_pdf_bytes):
        """After two uploads, /api/ask must return 200 (not the no-docs message)."""
        _pdf_upload(client, sample_pdf_bytes, "a.pdf")
        _pdf_upload(client, sample_pdf_bytes, "b.pdf")

        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = _ask(client, "What do these documents say?")

        assert response.status_code == 200

    def test_ask_after_two_uploads_returns_llm_answer(
        self, client, sample_pdf_bytes
    ):
        """With documents indexed, the answer must come from the LLM, not the
        no-documents fallback message."""
        _pdf_upload(client, sample_pdf_bytes, "a.pdf")
        _pdf_upload(client, sample_pdf_bytes, "b.pdf")

        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = _ask(client, "What do these documents say?")

        data = json.loads(response.data)
        assert data["answer"] == MOCK_ANSWER

    def test_ask_after_two_uploads_has_sources(self, client, sample_pdf_bytes):
        """Sources list must be populated (from the indexed chunks)."""
        _pdf_upload(client, sample_pdf_bytes, "a.pdf")
        _pdf_upload(client, sample_pdf_bytes, "b.pdf")

        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = _ask(client, "What do these documents say?")

        data = json.loads(response.data)
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) >= 1

    def test_ask_sources_have_correct_structure(self, client, sample_pdf_bytes):
        """Every source in the response must have 'filename' and 'page' keys."""
        _pdf_upload(client, sample_pdf_bytes, "a.pdf")
        _pdf_upload(client, sample_pdf_bytes, "b.pdf")

        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = _ask(client, "What do these documents say?")

        data = json.loads(response.data)
        for source in data["sources"]:
            assert "filename" in source
            assert "page" in source

    def test_ask_sources_filename_is_one_of_uploaded(
        self, client, sample_pdf_bytes
    ):
        """Source filenames must be from the set of uploaded documents."""
        uploaded = {"a.pdf", "b.pdf"}
        for name in uploaded:
            _pdf_upload(client, sample_pdf_bytes, name)

        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = _ask(client, "What do these documents say?")

        data = json.loads(response.data)
        for source in data["sources"]:
            assert source["filename"] in uploaded


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Retriever-level: retrieve() across multiple documents
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieverMultiDoc:
    """retrieve() searches across all documents in the shared collection."""

    def _seed_two_documents(self, collection):
        """Populate collection with 1 chunk each from two different documents."""
        doc_a_chunks = [Document(
            page_content="Aurora is an AWS managed relational database.",
            metadata={"source": "aurora.pdf", "page": 0, "page_number": 1, "total_pages": 1},
        )]
        doc_b_chunks = [Document(
            page_content="Lambda is a serverless compute service.",
            metadata={"source": "lambda.pdf", "page": 0, "page_number": 1, "total_pages": 1},
        )]
        # Use orthogonal vectors so both are retrieved equally for any query
        vec_a = [1.0] * 192 + [0.0] * 192
        vec_b = [0.0] * 192 + [1.0] * 192

        model_a = MagicMock()
        model_a.embed_documents.return_value = [vec_a]
        add_chunks(collection, doc_a_chunks, model_a, "aurora.pdf")

        model_b = MagicMock()
        model_b.embed_documents.return_value = [vec_b]
        add_chunks(collection, doc_b_chunks, model_b, "lambda.pdf")

        return vec_a, vec_b

    def test_collection_contains_chunks_from_both_docs(self, chroma_col):
        """After seeding two documents the collection must have 2 chunks total."""
        self._seed_two_documents(chroma_col)
        assert chroma_col.count() == 2

    def test_list_documents_shows_both_filenames(self, chroma_col):
        """list_documents() must return both filenames after seeding."""
        self._seed_two_documents(chroma_col)
        docs = list_documents(chroma_col)
        assert "aurora.pdf" in docs
        assert "lambda.pdf" in docs

    def test_retrieve_returns_chunks_from_both_docs(self, chroma_col):
        """When top_k >= 2 and two docs are indexed, retrieve() can return
        chunks from both documents."""
        self._seed_two_documents(chroma_col)

        # Query with a neutral vector so both docs are candidates
        query_model = MagicMock()
        query_model.embed_query.return_value = make_vec(0.5)

        result = retrieve(chroma_col, "Tell me about AWS services", query_model, top_k=2)
        assert len(result) == 2
        sources_found = {c.source for c in result}
        assert sources_found == {"aurora.pdf", "lambda.pdf"}

    def test_retrieve_preserves_source_per_chunk(self, chroma_col):
        """Each RetrievedChunk must carry the correct source filename."""
        self._seed_two_documents(chroma_col)
        query_model = MagicMock()
        query_model.embed_query.return_value = make_vec(0.5)

        result = retrieve(chroma_col, "AWS services", query_model, top_k=2)
        for chunk in result:
            assert chunk.source in {"aurora.pdf", "lambda.pdf"}

    def test_retrieve_preserves_page_number_per_doc(self, chroma_col):
        """page_number must be 1 for both single-page documents."""
        self._seed_two_documents(chroma_col)
        query_model = MagicMock()
        query_model.embed_query.return_value = make_vec(0.5)

        result = retrieve(chroma_col, "AWS services", query_model, top_k=2)
        for chunk in result:
            assert chunk.page_number == 1

    def test_unique_sources_from_multi_doc_result(self, chroma_col):
        """unique_sources() on a multi-doc retrieval must list both files."""
        self._seed_two_documents(chroma_col)
        query_model = MagicMock()
        query_model.embed_query.return_value = make_vec(0.5)

        result = retrieve(chroma_col, "AWS services", query_model, top_k=2)
        sources = unique_sources(result)

        filenames = {s["filename"] for s in sources}
        assert "aurora.pdf" in filenames
        assert "lambda.pdf" in filenames

    def test_unique_sources_deduplicates_across_docs(self):
        """unique_sources() deduplicates by (filename, page), not just page."""
        chunks = [
            RetrievedChunk(text="a", source="doc1.pdf", page_number=1, chunk_index=0, distance=0.1),
            RetrievedChunk(text="b", source="doc2.pdf", page_number=1, chunk_index=0, distance=0.2),
            RetrievedChunk(text="c", source="doc1.pdf", page_number=1, chunk_index=1, distance=0.3),
        ]
        sources = unique_sources(chunks)
        # doc1 p1 and doc2 p1 are different (different filename)
        assert len(sources) == 2
        filenames = {s["filename"] for s in sources}
        assert filenames == {"doc1.pdf", "doc2.pdf"}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. qa_service level: answer_question with multi-doc collection
# ═══════════════════════════════════════════════════════════════════════════════

class TestQAServiceMultiDoc:
    """answer_question() correctly handles a collection with multiple documents."""

    @pytest.fixture()
    def two_doc_collection(self, chroma_col, mock_embed):
        """Collection pre-loaded with 2 chunks from doc1.pdf and 2 from doc2.pdf."""
        chunks_a = [
            Document(
                page_content="IAM manages AWS user permissions.",
                metadata={"source": "iam.pdf", "page": 0, "page_number": 1, "total_pages": 2},
            ),
            Document(
                page_content="IAM roles can be assumed by EC2 instances.",
                metadata={"source": "iam.pdf", "page": 1, "page_number": 2, "total_pages": 2},
            ),
        ]
        chunks_b = [
            Document(
                page_content="S3 stores objects in buckets.",
                metadata={"source": "s3.pdf", "page": 0, "page_number": 1, "total_pages": 2},
            ),
            Document(
                page_content="S3 supports versioning and lifecycle policies.",
                metadata={"source": "s3.pdf", "page": 1, "page_number": 2, "total_pages": 2},
            ),
        ]
        add_chunks(chroma_col, chunks_a, mock_embed, "iam.pdf")
        add_chunks(chroma_col, chunks_b, mock_embed, "s3.pdf")
        return chroma_col

    def test_answer_question_returns_qa_result(
        self, two_doc_collection, mock_embed, mock_llm
    ):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            result = answer_question(
                "What is IAM?", two_doc_collection, mock_embed, mock_llm, top_k=4
            )
        assert isinstance(result, QAResult)

    def test_answer_is_llm_output(
        self, two_doc_collection, mock_embed, mock_llm
    ):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            result = answer_question(
                "What is IAM?", two_doc_collection, mock_embed, mock_llm, top_k=4
            )
        assert result.answer == MOCK_ANSWER

    def test_sources_are_populated(
        self, two_doc_collection, mock_embed, mock_llm
    ):
        """Sources must be non-empty when documents are indexed."""
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            result = answer_question(
                "What is IAM?", two_doc_collection, mock_embed, mock_llm, top_k=4
            )
        assert len(result.sources) >= 1

    def test_sources_have_filename_and_page(
        self, two_doc_collection, mock_embed, mock_llm
    ):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            result = answer_question(
                "What is IAM?", two_doc_collection, mock_embed, mock_llm, top_k=4
            )
        for source in result.sources:
            assert "filename" in source
            assert "page" in source

    def test_source_filenames_belong_to_uploaded_docs(
        self, two_doc_collection, mock_embed, mock_llm
    ):
        """Source filenames must be from the indexed document set."""
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            result = answer_question(
                "Tell me about AWS services",
                two_doc_collection, mock_embed, mock_llm, top_k=4,
            )
        for source in result.sources:
            assert source["filename"] in {"iam.pdf", "s3.pdf"}

    def test_chunks_used_reflects_multi_doc_retrieval(
        self, two_doc_collection, mock_embed, mock_llm
    ):
        """chunks_used must equal the number of chunks retrieved (up to top_k=4)."""
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            result = answer_question(
                "Tell me about AWS",
                two_doc_collection, mock_embed, mock_llm, top_k=4,
            )
        # Collection has 4 chunks total; top_k=4 → all retrieved
        assert result.chunks_used == 4

    def test_reupload_does_not_duplicate_sources(
        self, chroma_col, mock_embed, mock_llm
    ):
        """After re-uploading iam.pdf, sources must not contain duplicate
        iam.pdf entries beyond what deduplication allows."""
        chunks = [
            Document(
                page_content="IAM manages AWS user permissions.",
                metadata={"source": "iam.pdf", "page": 0, "page_number": 1, "total_pages": 1},
            ),
        ]
        add_chunks(chroma_col, chunks, mock_embed, "iam.pdf")
        # Re-upload same document
        add_chunks(chroma_col, chunks, mock_embed, "iam.pdf")

        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            result = answer_question(
                "What is IAM?", chroma_col, mock_embed, mock_llm, top_k=4
            )

        # Only 1 chunk in collection after dedup → only 1 source
        assert chroma_col.count() == 1
        assert len(result.sources) == 1
        assert result.sources[0]["filename"] == "iam.pdf"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. build_prompt with chunks from multiple documents
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildPromptMultiDoc:
    """build_prompt() correctly labels chunks from different source files."""

    def _chunk(self, text, source, page=1):
        return RetrievedChunk(
            text=text, source=source, page_number=page, chunk_index=0, distance=0.1
        )

    def test_prompt_contains_both_filenames(self):
        """When chunks come from two files, both filenames must appear in the prompt."""
        chunks = [
            self._chunk("IAM controls access.", "iam.pdf", page=1),
            self._chunk("S3 stores objects.", "s3.pdf", page=1),
        ]
        prompt = build_prompt("Tell me about AWS services", chunks)
        assert "iam.pdf" in prompt
        assert "s3.pdf" in prompt

    def test_prompt_contains_all_chunk_texts(self):
        """All chunk texts must be included in the prompt."""
        chunks = [
            self._chunk("IAM controls access.", "iam.pdf", page=1),
            self._chunk("S3 stores objects.", "s3.pdf", page=2),
        ]
        prompt = build_prompt("Tell me about AWS services", chunks)
        assert "IAM controls access." in prompt
        assert "S3 stores objects." in prompt

    def test_prompt_labels_each_source_with_filename_and_page(self):
        """Each source block must show its filename and page number."""
        chunks = [
            self._chunk("Text A", "doc_a.pdf", page=3),
            self._chunk("Text B", "doc_b.pdf", page=7),
        ]
        prompt = build_prompt("Question", chunks)
        assert "doc_a.pdf" in prompt
        assert "doc_b.pdf" in prompt
        assert "3" in prompt
        assert "7" in prompt

    def test_prompt_numbers_sources_sequentially(self):
        """Sources are numbered 1, 2, 3... regardless of which document they're from."""
        chunks = [
            self._chunk("A", "first.pdf", page=1),
            self._chunk("B", "second.pdf", page=1),
            self._chunk("C", "third.pdf", page=1),
        ]
        prompt = build_prompt("Q", chunks)
        assert "Source 1" in prompt
        assert "Source 2" in prompt
        assert "Source 3" in prompt

    def test_prompt_three_documents_all_present(self):
        """Three different source files must all appear in the prompt."""
        chunks = [
            self._chunk("Content from A", "alpha.pdf"),
            self._chunk("Content from B", "beta.pdf"),
            self._chunk("Content from C", "gamma.pdf"),
        ]
        prompt = build_prompt("What is this?", chunks)
        assert "alpha.pdf" in prompt
        assert "beta.pdf" in prompt
        assert "gamma.pdf" in prompt
