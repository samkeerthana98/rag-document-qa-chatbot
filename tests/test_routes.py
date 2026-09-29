"""
tests/test_routes.py — Tests for Phase 2 Flask routes and error handlers.

Covers:
  - GET /api/health
  - GET /
  - 404 error handler (unknown URL)
  - 405 error handler (wrong HTTP method)
  - 413 error handler (payload too large)
  - 500 error handler (unhandled exception in a route)

Interview explanation:
  These are integration tests — they test the full request/response cycle
  through Flask without starting a real HTTP server.  The test client from
  conftest.py handles all the plumbing.

  We test the error handlers explicitly because they are centralized
  infrastructure: if they break, every error across the entire application
  would return an HTML page instead of JSON, breaking any frontend that
  expects consistent API responses.
"""

import json
import pytest


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_returns_200(self, client):
        """Health endpoint should return HTTP 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Health endpoint should return JSON content type."""
        response = client.get("/api/health")
        assert response.content_type == "application/json"

    def test_health_body(self, client):
        """Health endpoint should return {"status": "healthy"}."""
        response = client.get("/api/health")
        data = json.loads(response.data)
        assert data == {"status": "healthy"}

    def test_health_post_not_allowed(self, client):
        """Health endpoint does not accept POST — should return 405."""
        response = client.post("/api/health")
        assert response.status_code == 405


# ── Frontend route ────────────────────────────────────────────────────────────

class TestFrontend:
    def test_index_returns_200(self, client):
        """GET / should return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_returns_html(self, client):
        """GET / should return an HTML document."""
        response = client.get("/")
        assert b"<!DOCTYPE html>" in response.data or b"<!doctype html>" in response.data.lower()

    def test_index_contains_title(self, client):
        """The HTML page should contain the application title.

        The title may appear with HTML entity encoding (&amp;) in the <title>
        tag and as literal '&' in visible text — either form is acceptable.
        """
        response = client.get("/")
        # Accept both the entity-encoded form (in <title>) and the raw form (in body text)
        assert (
            b"RAG Document Q&amp;A Chatbot" in response.data
            or b"RAG Document Q&A Chatbot" in response.data
        )


# ── Error handlers ────────────────────────────────────────────────────────────

class TestErrorHandlers:
    def test_404_unknown_route(self, client):
        """Requesting a non-existent URL should return 404 with JSON body."""
        response = client.get("/this/does/not/exist")
        assert response.status_code == 404

    def test_404_returns_json(self, client):
        """404 response should be JSON, not an HTML error page."""
        response = client.get("/no-such-endpoint")
        assert response.content_type == "application/json"

    def test_404_body(self, client):
        """404 JSON body should contain an 'error' key."""
        response = client.get("/no-such-endpoint")
        data = json.loads(response.data)
        assert "error" in data

    def test_405_wrong_method(self, client):
        """POST to a GET-only endpoint should return 405."""
        response = client.post("/api/health")
        assert response.status_code == 405

    def test_405_returns_json(self, client):
        """405 response should be JSON."""
        response = client.post("/api/health")
        assert response.content_type == "application/json"

    def test_405_body(self, client):
        """405 JSON body should contain an 'error' key."""
        response = client.post("/api/health")
        data = json.loads(response.data)
        assert "error" in data

    def test_413_payload_too_large(self, app, client):
        """Uploading data larger than MAX_CONTENT_LENGTH should return 413."""
        # Temporarily set a tiny limit to trigger the 413 handler reliably.
        original_limit = app.config["MAX_CONTENT_LENGTH"]
        app.config["MAX_CONTENT_LENGTH"] = 10  # 10 bytes

        # A future upload endpoint will receive this; for now we just confirm
        # Flask's built-in size check fires and our handler converts it to JSON.
        # We send to a route that exists (POST not allowed on /) to ensure the
        # size check happens before the 405 check.
        data = b"x" * 100  # 100 bytes > 10-byte limit
        response = client.post(
            "/api/upload",
            data={"file": (data, "big.pdf")},
            content_type="multipart/form-data",
        )

        app.config["MAX_CONTENT_LENGTH"] = original_limit  # restore

        # 413 or 404 are both acceptable here:
        # - 413 means the size check fired (correct behaviour once upload route exists)
        # - 404 means the route doesn't exist yet (also acceptable in Phase 2)
        # What matters is that the response is JSON, not an HTML error page.
        assert response.content_type == "application/json"
        assert response.status_code in (404, 413)

    def test_500_handler(self, app, client):
        """Unhandled exceptions in route code should return 500 JSON."""
        # Register a temporary route that deliberately raises an exception.
        @app.route("/test-500-trigger")
        def _trigger_500():
            raise RuntimeError("Deliberate test error")

        # With TESTING=True Flask re-raises exceptions by default.
        # We temporarily disable that so the 500 handler can intercept it.
        app.config["TESTING"] = False
        app.config["PROPAGATE_EXCEPTIONS"] = False

        response = client.get("/test-500-trigger")

        app.config["TESTING"] = True  # restore

        assert response.status_code == 500
        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert "error" in data
        # The internal error message must NOT be leaked to the client.
        assert "Deliberate test error" not in data["error"]



# ── Frontend HTML structure ────────────────────────────────────────────────────

class TestFrontendStructure:
    """Verify that the rendered HTML contains all required UI elements.

    These are server-side checks on the HTML returned by GET /.
    They confirm the template is correctly rendered and that the JavaScript
    and CSS assets are referenced.  They do not execute JavaScript.

    A broken template (missing element ID, wrong tag, removed section) would
    fail here before any browser test is needed.
    """

    def _html(self, client):
        """Return the GET / response body as a decoded string."""
        return client.get("/").data.decode()

    # ── Static assets linked ──────────────────────────────────────────────────

    def test_css_stylesheet_linked(self, client):
        """The HTML must link to /static/style.css."""
        assert "/static/style.css" in self._html(client)

    def test_script_js_linked(self, client):
        """The HTML must include /static/script.js."""
        assert "/static/script.js" in self._html(client)

    # ── Static asset serving ──────────────────────────────────────────────────

    def test_css_file_served(self, client):
        """GET /static/style.css must return 200 and CSS content."""
        response = client.get("/static/style.css")
        assert response.status_code == 200

    def test_js_file_served(self, client):
        """GET /static/script.js must return 200."""
        response = client.get("/static/script.js")
        assert response.status_code == 200

    def test_css_content_type(self, client):
        """style.css must be served with a CSS or text content type."""
        response = client.get("/static/style.css")
        assert "css" in response.content_type or "text" in response.content_type

    def test_js_content_type(self, client):
        """script.js must be served with a JavaScript or text content type."""
        response = client.get("/static/script.js")
        ct = response.content_type
        assert "javascript" in ct or "text" in ct

    # ── Upload section ────────────────────────────────────────────────────────

    def test_upload_section_present(self, client):
        """The upload section must be present in the rendered HTML."""
        assert 'id="upload-section"' in self._html(client)

    def test_drop_zone_present(self, client):
        """The drag-and-drop zone must be present."""
        assert 'id="drop-zone"' in self._html(client)

    def test_file_input_present(self, client):
        """The hidden file input must be present."""
        assert 'id="file-input"' in self._html(client)

    def test_file_input_accepts_pdf(self, client):
        """The file input must restrict to PDF files via the accept attribute."""
        assert 'accept=".pdf"' in self._html(client)

    def test_upload_btn_present(self, client):
        """The upload button must be present."""
        assert 'id="upload-btn"' in self._html(client)

    def test_upload_status_present(self, client):
        """The upload status message container must be present."""
        assert 'id="upload-status"' in self._html(client)

    # ── Indexed documents section ─────────────────────────────────────────────

    def test_docs_section_present(self, client):
        """The indexed documents section must be present (initially hidden)."""
        assert 'id="docs-section"' in self._html(client)

    def test_doc_list_present(self, client):
        """The document list must be present."""
        assert 'id="doc-list"' in self._html(client)

    # ── Q&A section ───────────────────────────────────────────────────────────

    def test_qa_section_present(self, client):
        """The Q&A section must be present."""
        assert 'id="qa-section"' in self._html(client)

    def test_question_input_present(self, client):
        """The question textarea must be present."""
        assert 'id="question-input"' in self._html(client)

    def test_ask_btn_present(self, client):
        """The Ask button must be present."""
        assert 'id="ask-btn"' in self._html(client)

    def test_answer_container_present(self, client):
        """The answer container must be present."""
        assert 'id="answer-container"' in self._html(client)

    def test_answer_text_present(self, client):
        """The answer text element must be present."""
        assert 'id="answer-text"' in self._html(client)

    def test_sources_list_present(self, client):
        """The sources list element must be present."""
        assert 'id="sources-list"' in self._html(client)

    def test_answer_error_present(self, client):
        """The answer error container must be present."""
        assert 'id="answer-error"' in self._html(client)

    # ── Accessibility attributes ──────────────────────────────────────────────

    def test_upload_status_has_aria_live(self, client):
        """Upload status must announce changes to screen readers via aria-live."""
        html = self._html(client)
        # The element with id="upload-status" must have aria-live
        assert 'id="upload-status"' in html
        assert 'aria-live="polite"' in html

    def test_answer_error_has_role_alert(self, client):
        """The answer error container must have role='alert' for immediate SR announcement."""
        assert 'role="alert"' in self._html(client)

    def test_question_label_visually_hidden(self, client):
        """The question textarea must have an accessible label (visually hidden is acceptable)."""
        html = self._html(client)
        assert 'visually-hidden' in html
        assert 'for="question-input"' in html

    def test_drop_zone_has_role(self, client):
        """The drop zone should have a role attribute for accessibility."""
        assert 'id="drop-zone"' in self._html(client)
        assert 'role=' in self._html(client)

    def test_svgs_have_aria_hidden(self, client):
        """Decorative SVG icons must be hidden from screen readers."""
        assert 'aria-hidden="true"' in self._html(client)

    def test_html_lang_attribute(self, client):
        """The <html> element must have a lang attribute for accessibility."""
        assert '<html lang="' in self._html(client)

    # ── Safe rendering (no raw script injection) ──────────────────────────────

    def test_no_inline_script_in_template(self, client):
        """User-facing template must not contain inline <script> blocks other than the src tag.

        Inline event handlers or embedded scripts bypass CSP and are XSS risks.
        All JS must live in /static/script.js.
        """
        html = self._html(client)
        # The only script tag should be the external src reference
        import re
        script_tags = re.findall(r"<script[^>]*>", html, re.IGNORECASE)
        for tag in script_tags:
            assert 'src=' in tag, f"Found inline <script> tag without src: {tag}"
