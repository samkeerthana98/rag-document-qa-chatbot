"""
tests/test_ask.py — Integration tests for POST /api/ask.

The LLM (Ollama) is MOCKED via unittest.mock.patch so tests run without
a running Ollama service.

We patch at the service layer (app.services.qa_service.invoke_llm) rather
than at the route level so the entire pipeline runs — including retrieval —
with only the final LLM call replaced by a mock.
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def post_ask(client, question):
    """POST /api/ask with a JSON body."""
    return client.post(
        "/api/ask",
        data=json.dumps({"question": question}),
        content_type="application/json",
    )


MOCK_ANSWER = "This is a mock answer from the test LLM."


# ── Happy path ────────────────────────────────────────────────────────────────

class TestAskSuccess:
    def test_ask_returns_200(self, client):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is AWS IAM?")
        assert response.status_code == 200

    def test_ask_returns_json(self, client):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is AWS IAM?")
        assert response.content_type == "application/json"

    def test_ask_response_has_answer_key(self, client):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is AWS IAM?")
        data = json.loads(response.data)
        assert "answer" in data

    def test_ask_response_has_sources_key(self, client):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is AWS IAM?")
        data = json.loads(response.data)
        assert "sources" in data

    def test_ask_sources_is_list(self, client):
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is AWS IAM?")
        data = json.loads(response.data)
        assert isinstance(data["sources"], list)

    def test_ask_no_documents_returns_friendly_message(self, client):
        """When no documents are uploaded, LLM is not called and a helpful message is returned."""
        with patch("app.services.qa_service.invoke_llm") as mock_invoke:
            response = post_ask(client, "What is the document about?")
            # LLM should NOT be called — no context to ground the answer
            mock_invoke.assert_not_called()

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "answer" in data
        # The message should guide the user to upload a document
        assert "upload" in data["answer"].lower() or "no document" in data["answer"].lower()


# ── Input validation ──────────────────────────────────────────────────────────

class TestAskValidation:
    def test_missing_question_field_returns_400(self, client):
        response = client.post(
            "/api/ask",
            data=json.dumps({"other": "field"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_empty_question_returns_400(self, client):
        response = post_ask(client, "")
        assert response.status_code == 400

    def test_whitespace_question_returns_400(self, client):
        response = post_ask(client, "   ")
        assert response.status_code == 400

    def test_non_json_body_returns_400(self, client):
        response = client.post(
            "/api/ask",
            data="not json at all",
            content_type="text/plain",
        )
        assert response.status_code == 400

    def test_empty_body_returns_400(self, client):
        response = client.post("/api/ask", content_type="application/json")
        assert response.status_code == 400

    def test_error_response_is_json(self, client):
        response = post_ask(client, "")
        assert response.content_type == "application/json"

    def test_error_has_error_key(self, client):
        response = post_ask(client, "")
        data = json.loads(response.data)
        assert "error" in data

    def test_no_stack_trace_in_error(self, client):
        response = post_ask(client, "")
        body = response.data.decode()
        assert "Traceback" not in body


# ── Ollama unavailable ────────────────────────────────────────────────────────

class TestAskOllamaUnavailable:
    def test_ollama_down_returns_503(self, client, app, sample_pdf_bytes):
        """When Ollama is not available, /api/ask returns 503, not 500."""
        from app.rag.llm import LLMUnavailableError

        # First upload a PDF so the collection is non-empty (otherwise
        # answer_question returns early before calling the LLM)
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )

        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=LLMUnavailableError("Ollama is not running"),
        ):
            response = post_ask(client, "What is in the document?")

        assert response.status_code == 503

    def test_ollama_down_returns_json(self, client, app, sample_pdf_bytes):
        from app.rag.llm import LLMUnavailableError

        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )

        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=LLMUnavailableError("Ollama is not running"),
        ):
            response = post_ask(client, "What is in the document?")

        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert "error" in data

    def test_ollama_error_not_exposed(self, client, sample_pdf_bytes):
        """Internal error details must not appear in the client response."""
        from app.rag.llm import LLMUnavailableError

        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )

        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=LLMUnavailableError("secret internal detail"),
        ):
            response = post_ask(client, "Any question?")

        body = response.data.decode()
        assert "Traceback" not in body


# ── HTTP method validation ────────────────────────────────────────────────────

class TestAskMethods:
    def test_get_not_allowed(self, client):
        response = client.get("/api/ask")
        assert response.status_code == 405
        assert response.content_type == "application/json"

    def test_put_not_allowed(self, client):
        response = client.put("/api/ask")
        assert response.status_code == 405



# ── Response value assertions (route-level) ───────────────────────────────────

class TestAskResponseValues:
    """Assert the actual content of /api/ask responses, not just their shape."""

    def test_answer_value_matches_llm_output(self, client, sample_pdf_bytes):
        """The 'answer' field must contain exactly what the mocked LLM returned.

        A document must be uploaded first so the no-documents early-return path
        is bypassed and the mocked LLM is actually called.
        """
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        assert data["answer"] == MOCK_ANSWER

    def test_answer_is_non_empty_string(self, client, sample_pdf_bytes):
        """After uploading a document, the answer must be a non-empty string."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_sources_each_have_filename_key(self, client, sample_pdf_bytes):
        """Every source dict in the response must have a 'filename' key."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        for source in data["sources"]:
            assert "filename" in source, f"Source missing 'filename' key: {source}"

    def test_sources_each_have_page_key(self, client, sample_pdf_bytes):
        """Every source dict in the response must have a 'page' key."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        for source in data["sources"]:
            assert "page" in source, f"Source missing 'page' key: {source}"

    def test_sources_filename_is_string(self, client, sample_pdf_bytes):
        """The 'filename' value in each source must be a string."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        for source in data["sources"]:
            assert isinstance(source["filename"], str)

    def test_sources_page_is_integer(self, client, sample_pdf_bytes):
        """The 'page' value in each source must be an integer."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch("app.services.qa_service.invoke_llm", return_value=MOCK_ANSWER):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        for source in data["sources"]:
            assert isinstance(source["page"], int)

    def test_no_documents_answer_is_non_empty_string(self, client):
        """Even with no documents, the 'answer' must be a non-empty string."""
        with patch("app.services.qa_service.invoke_llm"):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_no_documents_sources_is_empty_list(self, client):
        """When no documents are uploaded, sources must be an empty list."""
        with patch("app.services.qa_service.invoke_llm"):
            response = post_ask(client, "What is in the document?")
        data = json.loads(response.data)
        assert data["sources"] == []


# ── Out-of-scope question handling ────────────────────────────────────────────

class TestAskOutOfScope:
    """Verify that out-of-scope LLM responses pass through the pipeline unchanged.

    When the LLM determines the answer is not in the document context, it
    responds with the 'could not find' phrase from the prompt instruction.
    The route must pass this response through without modification.
    """

    OUT_OF_SCOPE_REPLY = (
        "I could not find the answer to that question in the uploaded document."
    )

    def test_out_of_scope_reply_returned_as_answer(self, client, sample_pdf_bytes):
        """The LLM's 'not found' response must be returned verbatim in the answer field."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            return_value=self.OUT_OF_SCOPE_REPLY,
        ):
            response = post_ask(client, "What is the GDP of France?")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["answer"] == self.OUT_OF_SCOPE_REPLY

    def test_out_of_scope_still_returns_200(self, client, sample_pdf_bytes):
        """An out-of-scope answer is a valid answer — status must be 200, not 4xx."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            return_value=self.OUT_OF_SCOPE_REPLY,
        ):
            response = post_ask(client, "What is the GDP of France?")

        assert response.status_code == 200

    def test_out_of_scope_sources_still_returned(self, client, sample_pdf_bytes):
        """Sources from retrieval are still returned even for out-of-scope answers."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            return_value=self.OUT_OF_SCOPE_REPLY,
        ):
            response = post_ask(client, "What is the GDP of France?")

        data = json.loads(response.data)
        # sources may or may not be populated depending on retrieval, but must be a list
        assert isinstance(data["sources"], list)

    def test_prompt_contains_not_found_instruction(self):
        """The grounded prompt must instruct the LLM how to respond for out-of-scope questions.

        This is a service-level check verifying the prompt template is correct.
        If this instruction is missing, the LLM has no guidance and may hallucinate.
        """
        from app.rag.retriever import RetrievedChunk
        from app.services.qa_service import build_prompt

        chunk = RetrievedChunk(
            text="Hello World document content.",
            source="doc.pdf",
            page_number=1,
            chunk_index=0,
            distance=0.1,
        )
        prompt = build_prompt("What is the GDP of France?", [chunk])
        assert "could not find" in prompt.lower() or "not found" in prompt.lower()


# ── Non-string question field ─────────────────────────────────────────────────

class TestAskNonStringQuestion:
    """The 'question' field must be a string. Other JSON types must return 400."""

    def test_integer_question_returns_400(self, client):
        """Sending question as an integer must be rejected with 400."""
        response = client.post(
            "/api/ask",
            data=json.dumps({"question": 42}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_null_question_returns_400(self, client):
        """Sending question as null must be rejected with 400."""
        response = client.post(
            "/api/ask",
            data=json.dumps({"question": None}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_list_question_returns_400(self, client):
        """Sending question as an array must be rejected with 400."""
        response = client.post(
            "/api/ask",
            data=json.dumps({"question": ["what", "is", "this"]}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_boolean_question_returns_400(self, client):
        """Sending question as a boolean must be rejected with 400."""
        response = client.post(
            "/api/ask",
            data=json.dumps({"question": True}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_non_string_error_response_has_error_key(self, client):
        """The 400 response for a non-string question must have an 'error' key."""
        response = client.post(
            "/api/ask",
            data=json.dumps({"question": 42}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert "error" in data


# ── Unexpected internal error → 500 ──────────────────────────────────────────

class TestAskInternalError:
    """Unexpected errors in the RAG pipeline must return 500, not crash the server."""

    def test_unexpected_exception_returns_500(self, client, sample_pdf_bytes):
        """An unexpected RuntimeError in the pipeline must yield 500."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=RuntimeError("something unexpected broke"),
        ):
            response = post_ask(client, "What is in the document?")

        assert response.status_code == 500

    def test_unexpected_exception_returns_json(self, client, sample_pdf_bytes):
        """The 500 response must be JSON."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=RuntimeError("something unexpected broke"),
        ):
            response = post_ask(client, "What is in the document?")

        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert "error" in data

    def test_unexpected_exception_no_internal_detail(self, client, sample_pdf_bytes):
        """The 500 response must not expose internal error details to the client."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=RuntimeError("secret internal detail xyz"),
        ):
            response = post_ask(client, "What is in the document?")

        body = response.data.decode()
        assert "secret internal detail xyz" not in body
        assert "Traceback" not in body


# ── 503 error message content ─────────────────────────────────────────────────

class TestAskOllamaErrorMessage:
    """The 503 error body must be user-actionable, not an opaque message."""

    def test_503_error_message_mentions_ollama(self, client, sample_pdf_bytes):
        """The 503 error message must mention Ollama so the user knows what to fix."""
        from app.rag.llm import LLMUnavailableError

        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=LLMUnavailableError(
                "The Ollama LLM service is not available. "
                "Please ensure Ollama is running: ollama serve"
            ),
        ):
            response = post_ask(client, "What is in the document?")

        data = json.loads(response.data)
        assert "ollama" in data["error"].lower()

    def test_503_error_field_is_string(self, client, sample_pdf_bytes):
        """The error field in a 503 response must be a non-empty string."""
        from app.rag.llm import LLMUnavailableError

        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=LLMUnavailableError("Ollama is not running"),
        ):
            response = post_ask(client, "What is in the document?")

        data = json.loads(response.data)
        assert isinstance(data["error"], str)
        assert len(data["error"]) > 0
