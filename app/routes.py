"""
app/routes.py — Flask route definitions.

All API endpoints live here.  The routes are grouped into a Blueprint so they
can be registered (and unregistered in tests) cleanly.

Implemented routes:
  GET /              — Serve the frontend HTML page
  GET /api/health    — Health check
  POST /api/upload   — Upload a PDF and extract chunks

Planned routes (added in later phases):
  POST /api/ask      — Phase 9: RAG question answering

Interview explanation:
  A Blueprint is Flask's way of grouping related routes.  It keeps route
  definitions separate from application setup and makes the codebase easier
  to navigate as it grows.

  Routes are kept intentionally thin — they handle HTTP concerns only
  (parsing request data, returning responses) and delegate all business logic
  to service and RAG modules.
"""

import logging
import os

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app.rag.document_processor import (
    DocumentProcessingError,
    allowed_extension,
    load_and_split,
    validate_pdf_content,
)
from app.rag.embeddings import get_embedding_model
from app.rag.llm import LLMUnavailableError, get_llm
from app.rag.vector_store import add_chunks, get_chroma_client, get_collection
from app.services.qa_service import answer_question

logger = logging.getLogger(__name__)

# All routes in this file are registered under this Blueprint.
main = Blueprint("main", __name__)


# ── Frontend ──────────────────────────────────────────────────────────────────

@main.route("/", methods=["GET"])
def index():
    """Serve the main frontend page.

    render_template loads templates/index.html and returns it as HTML.
    The frontend communicates with the backend via the /api/* endpoints
    using JavaScript fetch() calls.

    Returns:
        HTML: The main application page.
    """
    logger.debug("Frontend page requested")
    return render_template("index.html")


# ── Health check ──────────────────────────────────────────────────────────────

@main.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint.

    Used by load balancers, Docker health checks, and monitoring tools to
    verify the application is running.

    Returns:
        JSON: {"status": "healthy"} with HTTP 200.
    """
    logger.debug("Health check requested")
    return jsonify({"status": "healthy"}), 200


# ── PDF upload ────────────────────────────────────────────────────────────────

@main.route("/api/upload", methods=["POST"])
def upload_file():
    """Accept a PDF upload, validate it, extract text chunks.

    In Phase 3, this endpoint:
      1. Validates the uploaded file (presence, extension, magic bytes).
      2. Saves it to UPLOAD_FOLDER with a secure filename.
      3. Extracts text and splits into chunks via document_processor.
      4. Returns chunk count and filename.

    In Phase 6 (ChromaDB), the chunks returned here will also be stored
    in the vector database.  That wiring is added then, not now.

    Request:
        multipart/form-data with a 'file' field containing a PDF.

    Returns:
        200: {"message": "...", "filename": "...", "chunks_processed": N}
        400: {"error": "..."} — missing file, wrong type, empty PDF
        422: {"error": "..."} — PDF is valid but content cannot be extracted
        500: {"error": "..."} — unexpected error (logged server-side)

    Interview explanation:
        422 Unprocessable Entity is the right status when the request is
        well-formed (it is a PDF) but the content has a semantic problem
        (no extractable text).  400 is for structural/validation failures.
    """
    # ── 1. Check that a file was included in the request ─────────────────────
    if "file" not in request.files:
        logger.warning("Upload request received with no file field")
        return jsonify({"error": "No file provided. Include a 'file' field in the request."}), 400

    file = request.files["file"]

    # An empty <input type="file"> submits an empty filename.
    if not file.filename:
        logger.warning("Upload request received with empty filename")
        return jsonify({"error": "No file selected."}), 400

    # ── 2. Validate file extension ────────────────────────────────────────────
    original_filename: str = file.filename
    if not allowed_extension(original_filename):
        logger.warning("Rejected upload: invalid extension for '%s'", original_filename)
        return jsonify({"error": "Invalid file type. Only PDF files are accepted."}), 400

    # ── 3. Sanitise the filename ──────────────────────────────────────────────
    # secure_filename strips path separators, spaces, and other dangerous
    # characters.  It prevents directory traversal attacks like
    # "../../../etc/passwd.pdf".
    safe_filename = secure_filename(original_filename)
    if not safe_filename:
        # secure_filename can return "" for names that are all special chars.
        return jsonify({"error": "Invalid filename."}), 400

    # Re-validate the extension on the sanitised name.
    # e.g. ".pdf" → secure_filename → "pdf" which has no extension at all.
    if not allowed_extension(safe_filename):
        logger.warning(
            "Rejected upload: sanitised filename '%s' has no valid extension",
            safe_filename,
        )
        return jsonify({"error": "Invalid filename or file type."}), 400

    # ── 4. Validate PDF magic bytes ───────────────────────────────────────────
    # Read the first few bytes without consuming the stream, then seek back.
    header = file.read(5)
    file.seek(0)
    try:
        validate_pdf_content(header)
    except DocumentProcessingError as exc:
        logger.warning("Rejected upload: invalid PDF content for '%s'", safe_filename)
        return jsonify({"error": str(exc)}), 400

    # ── 5. Save the file to disk ──────────────────────────────────────────────
    upload_folder: str = current_app.config["UPLOAD_FOLDER"]
    save_path = os.path.join(upload_folder, safe_filename)

    # Defense-in-depth: confirm the resolved path stays inside upload_folder.
    # secure_filename() already strips path separators, but this assertion
    # makes the confinement guarantee explicit and catches any future edge case
    # where the sanitisation logic changes.
    if not os.path.abspath(save_path).startswith(os.path.abspath(upload_folder)):
        logger.error(
            "Path confinement violation: '%s' would escape upload folder '%s'",
            save_path,
            upload_folder,
        )
        return jsonify({"error": "Invalid filename."}), 400

    try:
        file.save(save_path)
        logger.info("Saved uploaded file to '%s'", save_path)
    except OSError as exc:
        logger.error("Failed to save uploaded file '%s': %s", safe_filename, exc)
        return jsonify({"error": "Failed to save the uploaded file."}), 500

    # ── 6. Extract and chunk the PDF text ─────────────────────────────────────
    chunk_size: int = current_app.config["CHUNK_SIZE"]
    chunk_overlap: int = current_app.config["CHUNK_OVERLAP"]

    try:
        chunks = load_and_split(
            file_path=save_path,
            filename=safe_filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except DocumentProcessingError as exc:
        # Known, user-facing error — clean message, no stack trace.
        logger.warning("Document processing failed for '%s': %s", safe_filename, exc)
        return jsonify({"error": str(exc)}), 422
    except FileNotFoundError as exc:
        logger.error("File missing after save for '%s': %s", safe_filename, exc)
        return jsonify({"error": "An internal error occurred processing the file."}), 500
    except Exception as exc:
        # Unexpected error — log the full detail server-side, generic message to client.
        logger.exception(
            "Unexpected error processing '%s': %s", safe_filename, exc
        )
        return jsonify({"error": "An error occurred while processing the document."}), 500

    logger.info(
        "Successfully processed '%s': %d chunks", safe_filename, len(chunks)
    )

    # ── 7. Embed chunks and store in ChromaDB ─────────────────────────────────
    try:
        embedding_model = get_embedding_model(current_app.config["EMBEDDING_MODEL"])
        chroma_client = get_chroma_client(current_app.config["CHROMA_DB_PATH"])
        collection = get_collection(chroma_client, current_app.config["CHROMA_COLLECTION_NAME"])
        stored = add_chunks(collection, chunks, embedding_model, safe_filename)
        logger.info("Stored %d chunks in ChromaDB for '%s'", stored, safe_filename)
    except Exception as exc:
        logger.exception("Failed to store chunks in ChromaDB for '%s': %s", safe_filename, exc)
        return jsonify({"error": "Document processed but failed to store embeddings."}), 500

    return (
        jsonify(
            {
                "message": "Document processed successfully",
                "filename": safe_filename,
                "chunks_processed": stored,
            }
        ),
        200,
    )



# ── Q&A ───────────────────────────────────────────────────────────────────────

@main.route("/api/ask", methods=["POST"])
def ask_question():
    """Accept a natural-language question and return a grounded answer.

    Full pipeline:
      1. Parse and validate the JSON request body.
      2. Open the ChromaDB collection.
      3. Retrieve relevant chunks via the embedding model.
      4. Build a grounded prompt and call the local Ollama LLM.
      5. Return the answer and source references.

    Request body (JSON):
        {"question": "What is AWS IAM?"}

    Returns:
        200: {"answer": "...", "sources": [{"filename": "...", "page": N}]}
        400: {"error": "..."} — missing or empty question
        503: {"error": "..."} — Ollama is not running
        500: {"error": "..."} — unexpected error (logged server-side)

    Interview explanation:
        503 Service Unavailable is correct when OUR service is up but a
        dependency (Ollama) is down.  It tells the client "try again later"
        rather than "you did something wrong" (400) or "we crashed" (500).
    """
    # ── 1. Parse request ──────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data:
        logger.warning("/api/ask received non-JSON or empty body")
        return jsonify({"error": "Request body must be JSON with a 'question' field."}), 400

    question = data.get("question", "")
    if not isinstance(question, str) or not question.strip():
        logger.warning("/api/ask received empty or missing question")
        return jsonify({"error": "The 'question' field is required and must not be empty."}), 400

    # Server-side length cap — the frontend enforces maxlength=1000 but any
    # HTTP client can bypass that.  Accepting an unbounded string would pass
    # it straight to the embedding model, which is a minor DoS vector.
    # 2000 chars gives comfortable headroom above the 1000-char UI limit.
    MAX_QUESTION_LENGTH = 2000
    if len(question) > MAX_QUESTION_LENGTH:
        logger.warning(
            "/api/ask received oversized question (%d chars, limit %d)",
            len(question),
            MAX_QUESTION_LENGTH,
        )
        return jsonify(
            {"error": f"Question is too long. Maximum length is {MAX_QUESTION_LENGTH} characters."}
        ), 400

    logger.info("/api/ask question: '%s'", question.strip()[:80])

    # ── 2. Open ChromaDB collection ───────────────────────────────────────────
    try:
        embedding_model = get_embedding_model(current_app.config["EMBEDDING_MODEL"])
        chroma_client = get_chroma_client(current_app.config["CHROMA_DB_PATH"])
        collection = get_collection(chroma_client, current_app.config["CHROMA_COLLECTION_NAME"])
    except Exception as exc:
        logger.exception("Failed to open ChromaDB: %s", exc)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

    # ── 3. Build LLM client ───────────────────────────────────────────────────
    llm = get_llm(
        model=current_app.config["OLLAMA_MODEL"],
        base_url=current_app.config["OLLAMA_BASE_URL"],
    )

    # ── 4. Run the RAG pipeline ───────────────────────────────────────────────
    try:
        result = answer_question(
            question=question,
            collection=collection,
            embedding_model=embedding_model,
            llm=llm,
            top_k=current_app.config["TOP_K"],
        )
    except ValueError as exc:
        # Empty question — already validated above, but belt-and-suspenders.
        return jsonify({"error": str(exc)}), 400
    except LLMUnavailableError as exc:
        logger.error("Ollama unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Unexpected error in /api/ask: %s", exc)
        return jsonify({"error": "An error occurred while generating the answer."}), 500

    return (
        jsonify(
            {
                "answer": result.answer,
                "sources": result.sources,
            }
        ),
        200,
    )
