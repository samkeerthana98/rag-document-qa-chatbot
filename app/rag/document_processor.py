"""
app/rag/document_processor.py — PDF loading, text extraction, and chunking.

This module is responsible for the first two steps of the RAG ingestion pipeline:

  PDF file → extracted text pages → text chunks

Each chunk carries metadata (source filename, page number) so that when the
retriever finds it later, the application can tell the user exactly which page
the answer came from.

Interview explanation:
  We split documents into chunks rather than passing the whole PDF to the LLM
  because:
  1. LLMs have a limited context window (token limit). A long PDF won't fit.
  2. Smaller, focused chunks produce better similarity search results.
  3. We can point the user to the exact page the answer came from.

  We use RecursiveCharacterTextSplitter from LangChain, which tries to split
  on natural boundaries (paragraphs → sentences → words) before falling back
  to hard character splits. This keeps chunks semantically coherent.

Public API:
  load_and_split(file_path, filename, chunk_size, chunk_overlap)
      → list[Document]   (each Document has .page_content and .metadata)

  validate_pdf_file(file_storage)
      → None  (raises ValueError with a user-friendly message on failure)
"""

import logging
import os
from pathlib import Path
from typing import List

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class DocumentProcessingError(Exception):
    """Raised when a PDF cannot be processed for a known, user-facing reason.

    Distinct from unexpected exceptions (IOError, etc.) so the upload route
    can return a 422 with a helpful message instead of a generic 500.
    """


# ── File validation ───────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"pdf"}
# PDF magic bytes — every valid PDF starts with "%PDF-"
PDF_MAGIC_BYTES = b"%PDF-"


def allowed_extension(filename: str) -> bool:
    """Return True if the filename has an allowed extension.

    Checks the part after the last dot, case-insensitively.
    A file named just ".pdf" (no base name) is rejected.

    Args:
        filename: The original filename from the upload.

    Returns:
        True if the extension is in ALLOWED_EXTENSIONS, False otherwise.
    """
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_pdf_content(file_bytes: bytes) -> None:
    """Check that the file's content is actually a PDF, not just named .pdf.

    A malicious or mistaken upload could rename any file as .pdf.  Checking
    the magic bytes catches this before we try to parse the file.

    Args:
        file_bytes: The first few bytes of the uploaded file.

    Raises:
        DocumentProcessingError: If the magic bytes don't match a PDF.
    """
    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        raise DocumentProcessingError(
            "The uploaded file does not appear to be a valid PDF. "
            "Please upload a PDF document."
        )


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_pages(file_path: str, filename: str) -> List[Document]:
    """Extract text from each page of a PDF file.

    Returns one LangChain Document per page.  Each Document carries metadata:
      - source:    original filename (e.g. "report.pdf")
      - page:      0-based page index from pypdf
      - page_number: 1-based page number for display to the user

    Why LangChain Documents?
      LangChain's splitter and retriever components expect Document objects,
      so using them here keeps the interface consistent throughout the pipeline.

    Args:
        file_path: Absolute path to the saved PDF file on disk.
        filename:  Original filename to store in metadata.

    Returns:
        List of Document objects, one per page that contains extractable text.

    Raises:
        DocumentProcessingError: If the PDF is password-protected, corrupt,
                                  or contains no extractable text at all.
        FileNotFoundError: If file_path does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    logger.info("Extracting text from '%s'", filename)

    try:
        reader = pypdf.PdfReader(file_path, strict=False)
    except pypdf.errors.PdfReadError as exc:
        raise DocumentProcessingError(
            f"Could not read PDF '{filename}'. The file may be corrupt."
        ) from exc

    # Check for password protection before trying to read pages.
    if reader.is_encrypted:
        raise DocumentProcessingError(
            f"'{filename}' is password-protected. "
            "Please upload an unencrypted PDF."
        )

    total_pages = len(reader.pages)
    logger.info("'%s' has %d page(s)", filename, total_pages)

    pages: List[Document] = []
    empty_pages: List[int] = []

    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # A single bad page should not abort the whole document.
            logger.warning(
                "Could not extract text from page %d of '%s': %s",
                page_index + 1,
                filename,
                exc,
            )
            text = ""

        text = text.strip()

        if not text:
            empty_pages.append(page_index + 1)
            logger.debug(
                "Page %d of '%s' has no extractable text (may be an image/scan)",
                page_index + 1,
                filename,
            )
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_index,           # 0-based (pypdf convention)
                    "page_number": page_index + 1,  # 1-based (display to user)
                    "total_pages": total_pages,
                },
            )
        )

    if not pages:
        if empty_pages:
            raise DocumentProcessingError(
                f"'{filename}' contains no extractable text. "
                "This usually means the PDF is a scanned image. "
                "Please upload a PDF with selectable text."
            )
        raise DocumentProcessingError(
            f"'{filename}' appears to be empty (0 pages)."
        )

    if empty_pages:
        logger.warning(
            "'%s': %d page(s) had no extractable text and were skipped: %s",
            filename,
            len(empty_pages),
            empty_pages,
        )

    logger.info(
        "Extracted text from %d/%d page(s) of '%s'",
        len(pages),
        total_pages,
        filename,
    )
    return pages


# ── Text chunking ─────────────────────────────────────────────────────────────

def split_into_chunks(
    pages: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split page-level Documents into smaller overlapping chunks.

    Why split at all?
      A full page of text is often too long for a useful similarity search
      (the signal gets diluted) and too long to fit in an LLM's context window
      alongside several other pages.  Smaller chunks give sharper matches.

    Why overlap?
      An important sentence that falls at the boundary between two chunks
      would be split in half without overlap.  Sharing characters between
      consecutive chunks prevents this.

    The splitter tries to break at paragraph boundaries first, then sentence
    boundaries, then word boundaries, and only hard-cuts characters as a last
    resort.  This keeps chunks semantically coherent.

    Metadata from the source page is copied to every chunk it produces, so
    each chunk still knows which page and filename it came from.

    Args:
        pages:         List of page-level Documents from extract_pages().
        chunk_size:    Target character count per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of chunk-level Documents with inherited metadata.

    Raises:
        DocumentProcessingError: If chunk_size/overlap values are invalid.
    """
    if chunk_size <= 0:
        raise DocumentProcessingError(
            f"chunk_size must be a positive integer, got {chunk_size}."
        )
    if chunk_overlap < 0:
        raise DocumentProcessingError(
            f"chunk_overlap must be non-negative, got {chunk_overlap}."
        )
    if chunk_overlap >= chunk_size:
        raise DocumentProcessingError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Separator priority: paragraph → newline → space → character
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    logger.info(
        "Split %d page(s) into %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d)",
        len(pages),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_split(
    file_path: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Load a PDF, extract text, and split into chunks.

    This is the single function the upload route calls.  It combines
    extract_pages() and split_into_chunks() in the correct order.

    Args:
        file_path:     Absolute path to the saved PDF file.
        filename:      Original filename for metadata.
        chunk_size:    Characters per chunk (from app config).
        chunk_overlap: Overlap between consecutive chunks (from app config).

    Returns:
        List of chunk-level Documents, each with source/page metadata.

    Raises:
        DocumentProcessingError: For any known, user-facing problem.
        FileNotFoundError:       If the file doesn't exist on disk.
    """
    pages = extract_pages(file_path, filename)
    chunks = split_into_chunks(pages, chunk_size, chunk_overlap)
    return chunks
