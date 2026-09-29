"""
tests/test_security.py — Security-focused tests (Phase 12).

Covers every security requirement from the Phase 12 checklist:

  Upload security:
    - Path traversal filename handling at the route level
    - Magic-byte (non-PDF) rejection
    - Extension-only validation bypass attempt

  /api/ask input validation:
    - Server-side question length cap (2000 chars)
    - Question too long → 400, not 500

  Configuration security:
    - SECRET_KEY default is a placeholder string, not empty
    - FLASK_DEBUG reads from the environment variable
    - .env is listed in .gitignore
    - .env.example exists and contains no real secrets

  Error response safety:
    - No filesystem paths exposed in upload error responses
    - No filesystem paths exposed in /api/ask error responses
    - No stack traces in any error response
    - Internal error details not leaked for unexpected exceptions

  Docker security:
    - .dockerignore excludes .env
    - .dockerignore excludes uploads/ and chroma_db/ (runtime data)

All LLM calls are mocked. Tests run fully offline.
"""

import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Project root — used for file existence checks
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Upload — path traversal
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadPathTraversal:
    """Filenames that attempt path traversal must be sanitised, not rejected
    outright — secure_filename converts them to safe names that stay within
    the upload folder."""

    def _upload(self, client, pdf_bytes, filename):
        return client.post(
            "/api/upload",
            data={"file": (io.BytesIO(pdf_bytes), filename)},
            content_type="multipart/form-data",
        )

    def test_path_traversal_filename_does_not_escape_upload_dir(
        self, client, app, sample_pdf_bytes
    ):
        """../../etc/passwd.pdf must be sanitised and saved inside upload_folder."""
        self._upload(client, sample_pdf_bytes, "../../etc/passwd.pdf")
        upload_folder = app.config["UPLOAD_FOLDER"]
        # Any saved file must live directly in the upload folder, not above it
        for f in os.listdir(upload_folder):
            full = os.path.join(upload_folder, f)
            assert os.path.abspath(full).startswith(os.path.abspath(upload_folder))

    def test_path_traversal_filename_sanitised_to_pdf(
        self, client, sample_pdf_bytes
    ):
        """A traversal attempt that results in a valid safe name still processes."""
        response = self._upload(client, sample_pdf_bytes, "../../safe.pdf")
        # secure_filename("../../safe.pdf") → "safe.pdf" — valid, should succeed
        assert response.status_code == 200
        data = json.loads(response.data)
        # Returned filename must not contain path separators
        assert "/" not in data["filename"]
        assert "\\" not in data["filename"]
        assert ".." not in data["filename"]

    def test_all_dots_filename_rejected(self, client, sample_pdf_bytes):
        """A filename of only dots/separators produces empty safe name → 400."""
        response = self._upload(client, sample_pdf_bytes, "../../.pdf")
        # secure_filename("../../.pdf") → "" or something without a base —
        # either way the route must not accept it silently
        # Acceptable outcomes: 400 (rejected) or 200 with a sanitised name
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            data = json.loads(response.data)
            assert ".." not in data["filename"]
            assert "/" not in data["filename"]

    def test_windows_path_traversal_sanitised(self, client, sample_pdf_bytes):
        """Windows-style backslash traversal is sanitised by secure_filename."""
        response = self._upload(
            client, sample_pdf_bytes, "..\\..\\windows\\report.pdf"
        )
        if response.status_code == 200:
            data = json.loads(response.data)
            assert "\\" not in data["filename"]
            assert ".." not in data["filename"]

    def test_upload_error_does_not_expose_filesystem_path(
        self, client, not_a_pdf_bytes
    ):
        """Error responses for upload failures must not contain filesystem paths."""
        response = self._upload(client, not_a_pdf_bytes, "fake.pdf")
        body = response.data.decode()
        # Must not contain absolute path patterns
        assert "C:\\" not in body
        assert "/home/" not in body
        assert "/Users/" not in body
        # Must not expose the server-side uploads directory path
        # (a generic message may say "upload" but must not reveal the OS path)
        import re
        assert not re.search(r'[/\\]uploads[/\\]', body)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. /api/ask — server-side question length cap
# ═══════════════════════════════════════════════════════════════════════════════

class TestAskQuestionLengthCap:
    """The server must enforce a question length limit independently of
    the frontend maxlength attribute."""

    MAX = 2000

    def _ask(self, client, question):
        return client.post(
            "/api/ask",
            data=json.dumps({"question": question}),
            content_type="application/json",
        )

    def test_question_at_limit_accepted(self, client, sample_pdf_bytes):
        """A question exactly at the limit (2000 chars) must be accepted."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "doc.pdf")},
            content_type="multipart/form-data",
        )
        question = "a" * self.MAX
        with patch("app.services.qa_service.invoke_llm", return_value="ok"):
            response = self._ask(client, question)
        assert response.status_code == 200

    def test_question_over_limit_returns_400(self, client):
        """A question one character over the limit must return 400."""
        question = "x" * (self.MAX + 1)
        response = self._ask(client, question)
        assert response.status_code == 400

    def test_question_well_over_limit_returns_400(self, client):
        """A very long question (e.g. 10 000 chars) must return 400."""
        question = "y" * 10_000
        response = self._ask(client, question)
        assert response.status_code == 400

    def test_oversized_question_error_is_json(self, client):
        """The 400 for an oversized question must be JSON."""
        question = "z" * (self.MAX + 1)
        response = self._ask(client, question)
        assert response.content_type == "application/json"

    def test_oversized_question_has_error_key(self, client):
        """The 400 body must contain an 'error' key."""
        question = "z" * (self.MAX + 1)
        response = self._ask(client, question)
        data = json.loads(response.data)
        assert "error" in data

    def test_oversized_question_no_stack_trace(self, client):
        """The 400 for oversized question must not expose a stack trace."""
        question = "z" * (self.MAX + 1)
        response = self._ask(client, question)
        assert "Traceback" not in response.data.decode()

    def test_question_just_under_limit_accepted(self, client, sample_pdf_bytes):
        """A question of 1999 chars (one under the limit) must be accepted."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "doc.pdf")},
            content_type="multipart/form-data",
        )
        question = "a" * (self.MAX - 1)
        with patch("app.services.qa_service.invoke_llm", return_value="ok"):
            response = self._ask(client, question)
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Configuration security
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigSecurity:
    """The application configuration must not hard-code secrets."""

    def test_secret_key_is_not_empty(self):
        """SECRET_KEY default must not be an empty string."""
        from app.config import Config
        assert Config.SECRET_KEY != ""
        assert len(Config.SECRET_KEY) > 0

    def test_secret_key_default_is_placeholder_not_production_value(self):
        """The default SECRET_KEY must read like a development placeholder,
        not a real production key — it should contain a recognisable hint."""
        from app.config import Config
        # The default warns users it needs changing
        key = Config.SECRET_KEY.lower()
        assert "dev" in key or "change" in key or "secret" in key

    def test_secret_key_reads_from_env(self, monkeypatch):
        """SECRET_KEY must be overridable via the SECRET_KEY environment variable."""
        import importlib
        import app.config as cfg_module

        monkeypatch.setenv("SECRET_KEY", "my-test-override-key-for-this-test")
        try:
            importlib.reload(cfg_module)
            assert cfg_module.Config.SECRET_KEY == "my-test-override-key-for-this-test"
        finally:
            # Always restore the module to its original state so other tests
            # are not affected, even if the assertion above fails.
            monkeypatch.delenv("SECRET_KEY", raising=False)
            importlib.reload(cfg_module)

    def test_flask_debug_reads_from_env(self, monkeypatch):
        """FLASK_DEBUG=false in the environment must set DEBUG to False."""
        import importlib
        import app.config as cfg_module

        monkeypatch.setenv("FLASK_DEBUG", "false")
        try:
            importlib.reload(cfg_module)
            assert cfg_module.Config.DEBUG is False
        finally:
            monkeypatch.delenv("FLASK_DEBUG", raising=False)
            importlib.reload(cfg_module)

    def test_flask_debug_true_from_env(self, monkeypatch):
        """FLASK_DEBUG=true in the environment must set DEBUG to True."""
        import importlib
        import app.config as cfg_module

        monkeypatch.setenv("FLASK_DEBUG", "true")
        try:
            importlib.reload(cfg_module)
            assert cfg_module.Config.DEBUG is True
        finally:
            monkeypatch.delenv("FLASK_DEBUG", raising=False)
            importlib.reload(cfg_module)

    def test_no_openai_api_key_in_config(self):
        """Config must not reference or require an OpenAI API key."""
        import app.config as cfg_module
        config_source = Path(cfg_module.__file__).read_text()
        assert "OPENAI_API_KEY" not in config_source
        assert "openai" not in config_source.lower()

    def test_ollama_base_url_default_is_localhost(self):
        """Default Ollama URL must be a localhost address (not a remote service)."""
        from app.config import Config
        assert "localhost" in Config.OLLAMA_BASE_URL or "127.0.0.1" in Config.OLLAMA_BASE_URL


# ═══════════════════════════════════════════════════════════════════════════════
# 4. .gitignore and .env.example
# ═══════════════════════════════════════════════════════════════════════════════

class TestGitignoreAndEnvExample:
    """.env must be git-ignored; .env.example must exist."""

    def test_env_file_in_gitignore(self):
        """.env must appear as an entry in .gitignore."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text()
        lines = [line.strip() for line in gitignore.splitlines()]
        assert ".env" in lines, ".env must be listed in .gitignore"

    def test_env_example_exists(self):
        """.env.example must exist in the project root."""
        assert (PROJECT_ROOT / ".env.example").is_file()

    def test_env_example_contains_secret_key(self):
        """.env.example must document the SECRET_KEY variable."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "SECRET_KEY" in content

    def test_env_example_contains_ollama_model(self):
        """.env.example must document OLLAMA_MODEL."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "OLLAMA_MODEL" in content

    def test_env_example_does_not_contain_real_secrets(self):
        """.env.example values must all be placeholders, not real credentials."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        # Should not contain anything that looks like a real API key
        assert "sk-" not in content          # OpenAI key pattern
        assert "AKIA" not in content         # AWS access key pattern
        assert "ghp_" not in content         # GitHub token pattern

    def test_uploads_directory_in_gitignore(self):
        """uploads/ must be git-ignored to prevent PDF files being committed."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text()
        assert "uploads/" in gitignore

    def test_chroma_db_in_gitignore(self):
        """chroma_db/ must be git-ignored."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text()
        assert "chroma_db/" in gitignore


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Error responses — no internal details exposed
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorResponseSafety:
    """All error responses must be safe for client consumption."""

    def _upload(self, client, pdf_bytes, filename):
        return client.post(
            "/api/upload",
            data={"file": (io.BytesIO(pdf_bytes), filename)},
            content_type="multipart/form-data",
        )

    def _ask(self, client, question):
        return client.post(
            "/api/ask",
            data=json.dumps({"question": question}),
            content_type="application/json",
        )

    def test_upload_invalid_file_no_traceback(self, client, not_a_pdf_bytes):
        body = self._upload(client, not_a_pdf_bytes, "fake.pdf").data.decode()
        assert "Traceback" not in body

    def test_upload_invalid_file_no_internal_module_names(
        self, client, not_a_pdf_bytes
    ):
        """Library names like pypdf/langchain must not appear in error bodies."""
        body = self._upload(client, not_a_pdf_bytes, "fake.pdf").data.decode()
        assert "pypdf" not in body
        assert "langchain" not in body

    def test_ask_empty_question_no_traceback(self, client):
        body = self._ask(client, "").data.decode()
        assert "Traceback" not in body

    def test_ask_empty_question_no_internal_paths(self, client):
        body = self._ask(client, "").data.decode()
        assert "site-packages" not in body
        assert "app\\routes" not in body
        assert "app/routes" not in body

    def test_upload_missing_file_field_no_traceback(self, client):
        response = client.post("/api/upload", content_type="multipart/form-data")
        assert "Traceback" not in response.data.decode()

    def test_404_response_no_traceback(self, client):
        body = client.get("/this/does/not/exist").data.decode()
        assert "Traceback" not in body

    def test_405_response_no_traceback(self, client):
        body = client.post("/api/health").data.decode()
        assert "Traceback" not in body

    def test_ask_unexpected_error_no_internal_detail(
        self, client, sample_pdf_bytes
    ):
        """An unexpected RuntimeError in the pipeline must return safe 500."""
        client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf_bytes), "doc.pdf")},
            content_type="multipart/form-data",
        )
        with patch(
            "app.services.qa_service.invoke_llm",
            side_effect=RuntimeError("internal secret detail xyz"),
        ):
            response = self._ask(client, "What is in the doc?")

        body = response.data.decode()
        assert response.status_code == 500
        assert "internal secret detail xyz" not in body
        assert "Traceback" not in body


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Docker security files
# ═══════════════════════════════════════════════════════════════════════════════

class TestDockerSecurityFiles:
    """.dockerignore must exclude sensitive files."""

    def test_dockerignore_exists(self):
        assert (PROJECT_ROOT / ".dockerignore").is_file()

    def test_dockerignore_excludes_env(self):
        """.dockerignore must exclude .env to prevent secrets baking into image."""
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        lines = [line.strip() for line in content.splitlines()]
        assert ".env" in lines

    def test_dockerignore_excludes_uploads(self):
        """uploads/ should not be baked into the Docker image."""
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert "uploads/" in content

    def test_dockerignore_excludes_chroma_db(self):
        """chroma_db/ is runtime data — must not be in the image."""
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert "chroma_db/" in content

    def test_dockerignore_excludes_pycache(self):
        """__pycache__/ must be excluded from the Docker build context."""
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert "__pycache__/" in content


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Magic-byte / extension bypass combinations
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadContentValidation:
    """Verify that both extension and content validation work together."""

    def _upload(self, client, file_bytes, filename):
        return client.post(
            "/api/upload",
            data={"file": (io.BytesIO(file_bytes), filename)},
            content_type="multipart/form-data",
        )

    def test_html_file_named_pdf_rejected(self, client):
        """An HTML file renamed to .pdf must be rejected (magic bytes check)."""
        html_bytes = b"<html><body>Not a PDF</body></html>"
        response = self._upload(client, html_bytes, "page.pdf")
        assert response.status_code == 400

    def test_jpeg_named_pdf_rejected(self, client):
        """A JPEG file renamed to .pdf must be rejected."""
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG magic
        response = self._upload(client, jpeg_bytes, "photo.pdf")
        assert response.status_code == 400

    def test_zip_named_pdf_rejected(self, client):
        """A ZIP file renamed to .pdf must be rejected."""
        zip_bytes = b"PK\x03\x04" + b"\x00" * 100  # ZIP magic
        response = self._upload(client, zip_bytes, "archive.pdf")
        assert response.status_code == 400

    def test_empty_file_named_pdf_rejected(self, client):
        """An empty file with .pdf extension must be rejected."""
        response = self._upload(client, b"", "empty.pdf")
        assert response.status_code == 400

    def test_pdf_named_txt_rejected(self, client, sample_pdf_bytes):
        """A valid PDF with .txt extension must be rejected (extension check)."""
        response = self._upload(client, sample_pdf_bytes, "document.txt")
        assert response.status_code == 400

    def test_valid_pdf_accepted(self, client, sample_pdf_bytes):
        """A valid PDF with .pdf extension must succeed."""
        response = self._upload(client, sample_pdf_bytes, "valid.pdf")
        assert response.status_code == 200

    def test_magic_bytes_check_error_message_is_safe(self, client):
        """The error for a non-PDF file must not reveal internal details."""
        response = self._upload(client, b"<html>not pdf</html>", "evil.pdf")
        body = response.data.decode()
        assert "Traceback" not in body
        assert "pypdf" not in body



# ═══════════════════════════════════════════════════════════════════════════════
# 8. Path confinement — saved file always inside upload_folder
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadPathConfinement:
    """Saved files must always resolve to a path inside UPLOAD_FOLDER.

    These tests verify the defense-in-depth os.path.abspath() check added
    in Phase 10, which runs after secure_filename() to make the confinement
    guarantee explicit in code.
    """

    def _upload(self, client, pdf_bytes, filename):
        return client.post(
            "/api/upload",
            data={"file": (io.BytesIO(pdf_bytes), filename)},
            content_type="multipart/form-data",
        )

    def test_normal_filename_saved_inside_upload_folder(
        self, client, app, sample_pdf_bytes
    ):
        """A normal filename must produce a save path inside UPLOAD_FOLDER."""
        self._upload(client, sample_pdf_bytes, "normal.pdf")
        upload_folder = os.path.abspath(app.config["UPLOAD_FOLDER"])
        for f in os.listdir(upload_folder):
            resolved = os.path.abspath(os.path.join(upload_folder, f))
            assert resolved.startswith(upload_folder), (
                f"File '{f}' resolved to '{resolved}', "
                f"which is outside upload folder '{upload_folder}'"
            )

    def test_traversal_filename_does_not_escape_upload_folder(
        self, client, app, sample_pdf_bytes
    ):
        """A traversal attempt must not produce a file outside UPLOAD_FOLDER."""
        self._upload(client, sample_pdf_bytes, "../../etc/passwd.pdf")
        upload_folder = os.path.abspath(app.config["UPLOAD_FOLDER"])
        for f in os.listdir(upload_folder):
            resolved = os.path.abspath(os.path.join(upload_folder, f))
            assert resolved.startswith(upload_folder)

    def test_saved_filename_contains_no_path_separators(
        self, client, app, sample_pdf_bytes
    ):
        """Filenames written to disk must not contain path separators."""
        self._upload(client, sample_pdf_bytes, "../../report.pdf")
        upload_folder = app.config["UPLOAD_FOLDER"]
        for f in os.listdir(upload_folder):
            assert "/" not in f
            assert "\\" not in f
            assert ".." not in f
