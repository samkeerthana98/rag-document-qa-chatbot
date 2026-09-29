"""
tests/test_upload.py — Integration tests for POST /api/upload.

These tests exercise the full HTTP request/response cycle through Flask,
including file validation, saving, and document processing.

All tests use the Flask test client (no real server needed).
File I/O goes to the temporary directory provided by the app fixture.

Fixtures used (all defined in conftest.py):
  client           — Flask test client
  app              — Flask application with tmp upload/chroma dirs
  sample_pdf_bytes — minimal valid PDF as raw bytes
  not_a_pdf_bytes  — bytes that fail the magic-byte check
"""

import io
import json
import os

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def upload(client, file_bytes, filename, content_type="application/pdf"):
    """Send a multipart POST /api/upload request."""
    return client.post(
        "/api/upload",
        data={"file": (io.BytesIO(file_bytes), filename)},
        content_type="multipart/form-data",
    )


# ── Happy path ────────────────────────────────────────────────────────────────

class TestUploadSuccess:
    def test_valid_pdf_returns_200(self, client, sample_pdf_bytes):
        response = upload(client, sample_pdf_bytes, "test.pdf")
        assert response.status_code == 200

    def test_valid_pdf_returns_json(self, client, sample_pdf_bytes):
        response = upload(client, sample_pdf_bytes, "test.pdf")
        assert response.content_type == "application/json"

    def test_valid_pdf_response_body(self, client, sample_pdf_bytes):
        response = upload(client, sample_pdf_bytes, "test.pdf")
        data = json.loads(response.data)
        assert "message" in data
        assert "filename" in data
        assert "chunks_processed" in data

    def test_chunks_processed_is_positive(self, client, sample_pdf_bytes):
        response = upload(client, sample_pdf_bytes, "test.pdf")
        data = json.loads(response.data)
        assert data["chunks_processed"] >= 1

    def test_filename_in_response(self, client, sample_pdf_bytes):
        response = upload(client, sample_pdf_bytes, "my_document.pdf")
        data = json.loads(response.data)
        # secure_filename may alter the name slightly; it should still end in .pdf
        assert data["filename"].endswith(".pdf")

    def test_file_saved_to_upload_folder(self, client, app, sample_pdf_bytes):
        upload(client, sample_pdf_bytes, "saved_check.pdf")
        upload_folder = app.config["UPLOAD_FOLDER"]
        saved_files = os.listdir(upload_folder)
        assert any(f.endswith(".pdf") for f in saved_files)

    def test_pdf_uppercase_extension_accepted(self, client, sample_pdf_bytes):
        """File extension check should be case-insensitive."""
        response = upload(client, sample_pdf_bytes, "REPORT.PDF")
        assert response.status_code == 200


# ── Validation failures ───────────────────────────────────────────────────────

class TestUploadValidation:
    def test_no_file_field_returns_400(self, client):
        """POST with no file field at all."""
        response = client.post("/api/upload", content_type="multipart/form-data")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_empty_filename_returns_400(self, client):
        """POST with an empty filename (no file selected)."""
        response = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_wrong_extension_returns_400(self, client):
        """A .txt file should be rejected."""
        response = upload(client, b"some text content", "notes.txt")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_docx_extension_returns_400(self, client):
        response = upload(client, b"PK\x03\x04fake docx", "report.docx")
        assert response.status_code == 400

    def test_fake_pdf_extension_returns_400(self, client, not_a_pdf_bytes):
        """A .pdf file whose bytes don't start with %PDF- should be rejected."""
        response = upload(client, not_a_pdf_bytes, "fake.pdf")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_error_response_is_json(self, client):
        """All error responses must be JSON, not HTML."""
        response = client.post("/api/upload", content_type="multipart/form-data")
        assert response.content_type == "application/json"

    def test_no_stack_trace_in_error(self, client, not_a_pdf_bytes):
        """Internal error details must not be exposed to the client."""
        response = upload(client, not_a_pdf_bytes, "fake.pdf")
        body = response.data.decode()
        assert "Traceback" not in body
        assert "pypdf" not in body


# ── Empty / scanned PDF ───────────────────────────────────────────────────────

class TestUploadEmptyPdf:
    def test_empty_pdf_returns_422(self, client, empty_pdf_bytes):
        """A PDF with no extractable text should return 422, not 500."""
        response = upload(client, empty_pdf_bytes, "scanned.pdf")
        assert response.status_code == 422

    def test_empty_pdf_error_message(self, client, empty_pdf_bytes):
        response = upload(client, empty_pdf_bytes, "scanned.pdf")
        data = json.loads(response.data)
        assert "error" in data
        # Message should mention text extraction, not reveal internal details
        error_msg = data["error"].lower()
        assert any(word in error_msg for word in ["text", "extractable", "scan"])


# ── Method validation ─────────────────────────────────────────────────────────

class TestUploadMethods:
    def test_get_not_allowed(self, client):
        response = client.get("/api/upload")
        assert response.status_code == 405
        assert response.content_type == "application/json"

    def test_put_not_allowed(self, client):
        response = client.put("/api/upload")
        assert response.status_code == 405

    def test_dot_only_filename_returns_400(self, client, sample_pdf_bytes):
        """'.pdf' with no base name is rejected by secure_filename at the route."""
        response = upload(client, sample_pdf_bytes, ".pdf")
        assert response.status_code == 400
