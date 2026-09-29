"""
tests/conftest.py — Shared pytest fixtures.

conftest.py is a special pytest file.  Any fixtures defined here are
automatically available to every test file in the tests/ directory without
needing to import them.

Interview explanation:
  A fixture is a reusable piece of setup/teardown code.  Instead of creating
  a Flask test client in every test function, we define it once here and
  pytest injects it wherever a test function declares it as a parameter.

  tmp_path is a pytest built-in fixture that provides a temporary directory
  unique to each test — automatically cleaned up after the test finishes.
  Using it (rather than /tmp or a hardcoded path) makes fixtures work
  identically on Windows, macOS, and Linux.
"""

import io
import os

import pytest

from app import create_app


# ── Application fixtures ───────────────────────────────────────────────────────

@pytest.fixture()
def tmp_upload_dir(tmp_path):
    """Return a temporary directory for uploaded files.

    Uses pytest's built-in tmp_path fixture, which is:
      - unique per test
      - automatically deleted after the test
      - cross-platform (works on Windows, macOS, Linux)
    """
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return str(upload_dir)


@pytest.fixture()
def app(tmp_upload_dir, tmp_path):
    """Create a Flask application instance configured for testing.

    TESTING=True tells Flask to:
      - Propagate exceptions instead of handling them (better error messages
        in test output).
      - Disable the interactive debugger.

    We point UPLOAD_FOLDER and CHROMA_DB_PATH at temporary directories so
    tests never write to the real project directories.
    """
    chroma_dir = str(tmp_path / "chroma")
    os.makedirs(chroma_dir, exist_ok=True)

    test_app = create_app()
    test_app.config.update(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "UPLOAD_FOLDER": tmp_upload_dir,
            "CHROMA_DB_PATH": chroma_dir,
        }
    )
    yield test_app


@pytest.fixture()
def client(app):
    """Return a Flask test client for the test application.

    The test client lets tests make HTTP requests to the application
    without starting a real server.  Requests stay in-process, so tests
    run fast and don't need a network.
    """
    return app.test_client()


@pytest.fixture()
def runner(app):
    """Return a Flask CLI test runner.

    Useful for testing custom Flask CLI commands (added in later phases).
    """
    return app.test_cli_runner()


# ── PDF fixtures ───────────────────────────────────────────────────────────────

# A minimal but structurally valid PDF containing one page with "Hello World".
# Hand-crafted so there's no dependency on any PDF-generation library.
# pypdf recovers it cleanly even though the xref offset is approximate.
MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000062 00000 n
0000000119 00000 n
0000000273 00000 n
0000000372 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""


@pytest.fixture()
def sample_pdf_path(tmp_path):
    """Write the minimal PDF to a temporary file and return its path.

    Use this fixture in document_processor unit tests that need a file path.
    """
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(MINIMAL_PDF_BYTES)
    return str(pdf_file)


@pytest.fixture()
def sample_pdf_bytes():
    """Return the minimal PDF as raw bytes.

    Use this fixture in upload route tests that send a multipart request.
    """
    return MINIMAL_PDF_BYTES


@pytest.fixture()
def sample_pdf_stream():
    """Return the minimal PDF as a BytesIO stream.

    Useful when a function under test accepts a file-like object.
    """
    return io.BytesIO(MINIMAL_PDF_BYTES)


@pytest.fixture()
def not_a_pdf_bytes():
    """Return bytes that look like a .pdf filename but aren't a PDF."""
    return b"This is definitely not a PDF file."


@pytest.fixture()
def empty_pdf_bytes():
    """Return a minimal PDF whose single page has no extractable text.

    The page exists but its content stream produces no text — simulates a
    scanned image page.
    """
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << >> >>
endobj
4 0 obj
<< /Length 0 >>
stream
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000062 00000 n
0000000119 00000 n
0000000232 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
289
%%EOF"""
