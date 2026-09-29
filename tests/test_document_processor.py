"""
tests/test_document_processor.py — Unit tests for app/rag/document_processor.py.

These tests exercise the document processing logic in complete isolation from
Flask.  They do not start a server and do not require any uploaded files in
the real uploads/ directory.

Fixtures used (all defined in conftest.py):
  sample_pdf_path   — path to a temporary minimal PDF with text "Hello World"
  sample_pdf_bytes  — same PDF as raw bytes
  not_a_pdf_bytes   — non-PDF bytes (fails magic byte check)
  empty_pdf_bytes   — PDF with no extractable text (simulates scanned image)
  tmp_path          — pytest built-in: unique temporary directory per test
"""

import os
import pytest

from app.rag.document_processor import (
    DocumentProcessingError,
    allowed_extension,
    extract_pages,
    load_and_split,
    split_into_chunks,
    validate_pdf_content,
)


# ── allowed_extension ─────────────────────────────────────────────────────────

class TestAllowedExtension:
    def test_pdf_lowercase_accepted(self):
        assert allowed_extension("report.pdf") is True

    def test_pdf_uppercase_accepted(self):
        assert allowed_extension("REPORT.PDF") is True

    def test_pdf_mixed_case_accepted(self):
        assert allowed_extension("Report.Pdf") is True

    def test_txt_rejected(self):
        assert allowed_extension("notes.txt") is False

    def test_docx_rejected(self):
        assert allowed_extension("document.docx") is False

    def test_no_extension_rejected(self):
        assert allowed_extension("nodotfile") is False

    def test_empty_string_rejected(self):
        assert allowed_extension("") is False

    def test_dot_only_has_valid_extension(self):
        # ".pdf" technically has extension "pdf" — allowed_extension returns True.
        # The empty-base-name case is caught later by secure_filename() in the route,
        # which returns "" for ".pdf" and the route rejects it with a 400.
        # allowed_extension's responsibility is only the extension, not the base name.
        assert allowed_extension(".pdf") is True

    def test_path_traversal_extension(self):
        # Extension check alone: "../etc/passwd.pdf" has a .pdf extension
        # The route uses secure_filename to strip path separators.
        assert allowed_extension("../etc/passwd.pdf") is True

    def test_double_extension_uses_last(self):
        # "file.pdf.exe" — last extension is .exe, should be rejected
        assert allowed_extension("file.pdf.exe") is False


# ── validate_pdf_content ──────────────────────────────────────────────────────

class TestValidatePdfContent:
    def test_valid_pdf_bytes_accepted(self, sample_pdf_bytes):
        # Should not raise
        validate_pdf_content(sample_pdf_bytes[:5])

    def test_non_pdf_raises(self, not_a_pdf_bytes):
        with pytest.raises(DocumentProcessingError, match="does not appear to be a valid PDF"):
            validate_pdf_content(not_a_pdf_bytes[:5])

    def test_empty_bytes_raises(self):
        with pytest.raises(DocumentProcessingError):
            validate_pdf_content(b"")

    def test_html_bytes_raises(self):
        with pytest.raises(DocumentProcessingError):
            validate_pdf_content(b"<html>")

    def test_magic_bytes_only_accepted(self):
        # Anything starting with %PDF- passes the magic byte check
        validate_pdf_content(b"%PDF-corrupted-rest-is-fine-for-this-check")


# ── extract_pages ─────────────────────────────────────────────────────────────

class TestExtractPages:
    def test_returns_documents(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        assert len(pages) >= 1

    def test_page_content_is_string(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        for page in pages:
            assert isinstance(page.page_content, str)
            assert len(page.page_content) > 0

    def test_extracts_expected_text(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        combined = " ".join(p.page_content for p in pages)
        assert "Hello World" in combined

    def test_metadata_source_field(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        for page in pages:
            assert page.metadata["source"] == "sample.pdf"

    def test_metadata_page_number_is_one_based(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        # First page should be page_number=1
        assert pages[0].metadata["page_number"] == 1

    def test_metadata_page_is_zero_based(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        # page (0-based index) of first page should be 0
        assert pages[0].metadata["page"] == 0

    def test_metadata_total_pages(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        assert pages[0].metadata["total_pages"] >= 1

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_pages(str(tmp_path / "nonexistent.pdf"), "nonexistent.pdf")

    def test_empty_pdf_raises_processing_error(self, empty_pdf_bytes, tmp_path):
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(empty_pdf_bytes)
        with pytest.raises(DocumentProcessingError, match="no extractable text"):
            extract_pages(str(empty_pdf), "empty.pdf")

    def test_custom_filename_in_metadata(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "my_document.pdf")
        assert all(p.metadata["source"] == "my_document.pdf" for p in pages)


# ── split_into_chunks ─────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_returns_list(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        chunks = split_into_chunks(pages, chunk_size=500, chunk_overlap=50)
        assert isinstance(chunks, list)

    def test_chunks_are_non_empty(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        chunks = split_into_chunks(pages, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) > 0

    def test_metadata_preserved_in_chunks(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        chunks = split_into_chunks(pages, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page_number" in chunk.metadata

    def test_small_chunk_size_produces_more_chunks(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        # "Hello World" is 11 chars — chunk_size=5 should split it
        chunks_small = split_into_chunks(pages, chunk_size=5, chunk_overlap=0)
        chunks_large = split_into_chunks(pages, chunk_size=1000, chunk_overlap=0)
        assert len(chunks_small) >= len(chunks_large)

    def test_invalid_chunk_size_raises(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        with pytest.raises(DocumentProcessingError, match="chunk_size must be a positive integer"):
            split_into_chunks(pages, chunk_size=0, chunk_overlap=0)

    def test_negative_chunk_size_raises(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        with pytest.raises(DocumentProcessingError):
            split_into_chunks(pages, chunk_size=-100, chunk_overlap=0)

    def test_negative_overlap_raises(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        with pytest.raises(DocumentProcessingError, match="chunk_overlap must be non-negative"):
            split_into_chunks(pages, chunk_size=500, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        with pytest.raises(DocumentProcessingError, match="chunk_overlap.*must be less than"):
            split_into_chunks(pages, chunk_size=100, chunk_overlap=100)

    def test_overlap_larger_than_chunk_size_raises(self, sample_pdf_path):
        pages = extract_pages(sample_pdf_path, "sample.pdf")
        with pytest.raises(DocumentProcessingError):
            split_into_chunks(pages, chunk_size=100, chunk_overlap=200)


# ── load_and_split (integration of extract + split) ──────────────────────────

class TestLoadAndSplit:
    def test_returns_chunks(self, sample_pdf_path):
        chunks = load_and_split(sample_pdf_path, "sample.pdf")
        assert len(chunks) >= 1

    def test_chunks_have_metadata(self, sample_pdf_path):
        chunks = load_and_split(sample_pdf_path, "sample.pdf")
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page_number" in chunk.metadata

    def test_custom_chunk_size(self, sample_pdf_path):
        chunks = load_and_split(
            sample_pdf_path, "sample.pdf", chunk_size=50, chunk_overlap=10
        )
        # Every chunk should respect the chunk_size limit (with some tolerance
        # because the splitter may slightly exceed it on word boundaries)
        for chunk in chunks:
            assert len(chunk.page_content) <= 100  # generous upper bound

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_split(str(tmp_path / "missing.pdf"), "missing.pdf")

    def test_empty_pdf_raises(self, empty_pdf_bytes, tmp_path):
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(empty_pdf_bytes)
        with pytest.raises(DocumentProcessingError):
            load_and_split(str(empty_pdf), "empty.pdf")
