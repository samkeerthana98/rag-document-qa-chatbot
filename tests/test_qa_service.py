"""
tests/test_qa_service.py — Unit tests for app/services/qa_service.py.

The LLM is MOCKED — tests do not require Ollama to be running.
ChromaDB runs REAL with a temporary directory.
The embedding model is MOCKED — no model weights loaded.

This lets the test suite:
  - Run fast (no network, no GPU)
  - Run in CI without external services
  - Verify the logic of the service (prompt building, result formatting)
    without caring about the specific LLM output.
"""

from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.documents import Document

from app.rag.llm import LLMUnavailableError
from app.rag.retriever import RetrievedChunk
from app.rag.vector_store import add_chunks, get_collection
from app.services.qa_service import QAResult, answer_question, build_prompt


# ── Fixtures ──────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384


def make_vec(val: float) -> list:
    return [val] * EMBEDDING_DIM


@pytest.fixture()
def chroma_col(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return get_collection(client, "test_qa")


@pytest.fixture()
def mock_embedding_model():
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [make_vec(0.5)] * len(texts)
    mock.embed_query.return_value = make_vec(0.5)
    return mock


@pytest.fixture()
def mock_llm():
    """Mock LLM that returns a fixed answer string."""
    mock = MagicMock()
    mock.invoke.return_value = "This is a test answer from the mock LLM."
    return mock


@pytest.fixture()
def populated_collection(chroma_col, mock_embedding_model):
    """A collection pre-loaded with 2 document chunks."""
    chunks = [
        Document(
            page_content="IAM controls access to AWS services.",
            metadata={"source": "aws.pdf", "page": 0, "page_number": 1, "total_pages": 2},
        ),
        Document(
            page_content="S3 provides object storage.",
            metadata={"source": "aws.pdf", "page": 1, "page_number": 2, "total_pages": 2},
        ),
    ]
    add_chunks(chroma_col, chunks, mock_embedding_model, "aws.pdf")
    return chroma_col


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def _make_chunk(self, text, source="doc.pdf", page=1):
        return RetrievedChunk(
            text=text, source=source, page_number=page, chunk_index=0, distance=0.1
        )

    def test_returns_string(self):
        chunks = [self._make_chunk("Some context.")]
        result = build_prompt("What is this?", chunks)
        assert isinstance(result, str)

    def test_contains_question(self):
        chunks = [self._make_chunk("Some context.")]
        result = build_prompt("What is the main topic?", chunks)
        assert "What is the main topic?" in result

    def test_contains_chunk_text(self):
        chunks = [self._make_chunk("Important document content here.")]
        result = build_prompt("What?", chunks)
        assert "Important document content here." in result

    def test_contains_source_reference(self):
        chunks = [self._make_chunk("Content.", source="report.pdf", page=5)]
        result = build_prompt("Question?", chunks)
        assert "report.pdf" in result
        assert "5" in result

    def test_contains_grounding_instruction(self):
        """Prompt must instruct the LLM to answer from context only."""
        chunks = [self._make_chunk("Context.")]
        result = build_prompt("Q?", chunks)
        lower = result.lower()
        assert "context" in lower
        assert "only" in lower or "strictly" in lower

    def test_contains_not_found_instruction(self):
        """Prompt must tell LLM what to say when answer is not in context."""
        chunks = [self._make_chunk("Context.")]
        result = build_prompt("Q?", chunks)
        assert "not found" in result.lower() or "could not find" in result.lower()

    def test_multiple_chunks_all_included(self):
        chunks = [
            self._make_chunk("First chunk.", page=1),
            self._make_chunk("Second chunk.", page=2),
            self._make_chunk("Third chunk.", page=3),
        ]
        result = build_prompt("Q?", chunks)
        assert "First chunk." in result
        assert "Second chunk." in result
        assert "Third chunk." in result

    def test_chunk_source_numbered(self):
        """Each chunk should be labelled Source 1, Source 2, etc."""
        chunks = [
            self._make_chunk("A.", page=1),
            self._make_chunk("B.", page=2),
        ]
        result = build_prompt("Q?", chunks)
        assert "Source 1" in result
        assert "Source 2" in result


# ── QAResult dataclass ────────────────────────────────────────────────────────

class TestQAResult:
    def test_can_construct(self):
        r = QAResult(answer="Answer text", sources=[{"filename": "a.pdf", "page": 1}], chunks_used=2)
        assert r.answer == "Answer text"
        assert r.chunks_used == 2

    def test_default_sources_is_empty_list(self):
        r = QAResult(answer="x")
        assert r.sources == []

    def test_default_chunks_used_is_zero(self):
        r = QAResult(answer="x")
        assert r.chunks_used == 0


# ── answer_question ───────────────────────────────────────────────────────────

class TestAnswerQuestion:
    def test_returns_qa_result(self, populated_collection, mock_embedding_model, mock_llm):
        result = answer_question(
            "What does IAM do?",
            populated_collection,
            mock_embedding_model,
            mock_llm,
            top_k=2,
        )
        assert isinstance(result, QAResult)

    def test_answer_is_string(self, populated_collection, mock_embedding_model, mock_llm):
        result = answer_question(
            "What does IAM do?",
            populated_collection,
            mock_embedding_model,
            mock_llm,
        )
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_llm_invoke_called(self, populated_collection, mock_embedding_model, mock_llm):
        """The LLM must be called exactly once per question."""
        answer_question(
            "What does IAM do?",
            populated_collection,
            mock_embedding_model,
            mock_llm,
        )
        mock_llm.invoke.assert_called_once()

    def test_llm_receives_prompt_with_question(
        self, populated_collection, mock_embedding_model, mock_llm
    ):
        """The prompt sent to the LLM must contain the user's question."""
        answer_question(
            "What does IAM do?",
            populated_collection,
            mock_embedding_model,
            mock_llm,
        )
        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "What does IAM do?" in prompt_sent

    def test_sources_populated(self, populated_collection, mock_embedding_model, mock_llm):
        """Sources list must be populated from retrieved chunks."""
        result = answer_question(
            "Question",
            populated_collection,
            mock_embedding_model,
            mock_llm,
        )
        assert isinstance(result.sources, list)
        assert len(result.sources) >= 1
        assert "filename" in result.sources[0]
        assert "page" in result.sources[0]

    def test_chunks_used_positive(self, populated_collection, mock_embedding_model, mock_llm):
        result = answer_question(
            "Question",
            populated_collection,
            mock_embedding_model,
            mock_llm,
        )
        assert result.chunks_used >= 1

    def test_empty_question_raises(self, populated_collection, mock_embedding_model, mock_llm):
        with pytest.raises(ValueError, match="not be empty"):
            answer_question("", populated_collection, mock_embedding_model, mock_llm)

    def test_whitespace_question_raises(self, populated_collection, mock_embedding_model, mock_llm):
        with pytest.raises(ValueError):
            answer_question("   ", populated_collection, mock_embedding_model, mock_llm)

    def test_empty_collection_returns_no_docs_message(
        self, chroma_col, mock_embedding_model, mock_llm
    ):
        """When no documents are uploaded, return a friendly message without calling LLM."""
        result = answer_question(
            "What is IAM?",
            chroma_col,  # empty collection
            mock_embedding_model,
            mock_llm,
        )
        assert isinstance(result, QAResult)
        assert result.chunks_used == 0
        assert result.sources == []
        # LLM must NOT be called — no context means no hallucination
        mock_llm.invoke.assert_not_called()
        # Message should guide the user
        assert "upload" in result.answer.lower() or "no document" in result.answer.lower()

    def test_llm_unavailable_error_propagates(
        self, populated_collection, mock_embedding_model
    ):
        """LLMUnavailableError from invoke_llm must propagate to the caller."""
        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = Exception("connection refused")

        with patch("app.services.qa_service.invoke_llm") as mock_invoke:
            mock_invoke.side_effect = LLMUnavailableError("Ollama is not running")
            with pytest.raises(LLMUnavailableError):
                answer_question(
                    "What is IAM?",
                    populated_collection,
                    mock_embedding_model,
                    failing_llm,
                )
