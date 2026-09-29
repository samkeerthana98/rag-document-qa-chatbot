"""
app/__init__.py — Flask application factory.

Using a factory function (create_app) instead of a module-level app instance
gives us two benefits:
  1. Tests can create isolated app instances with different configs.
  2. Avoids circular import problems as the project grows.

Interview explanation:
  The app factory pattern is the standard way to structure a Flask application.
  It lets you call create_app() with different configurations for testing vs
  production without changing any other code.
"""

import logging
import os
import time
from flask import Flask, jsonify, g, request
from dotenv import load_dotenv

# Load .env file into the environment before reading Config.
# This is a no-op if .env does not exist (e.g. in production where variables
# are injected directly into the environment).
load_dotenv()


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        A fully configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder="../templates",  # templates/ at project root
        static_folder="../static",       # static/ at project root
    )

    # ── Load configuration ────────────────────────────────────────────────────
    from app.config import Config
    app.config.from_object(Config)

    # ── Ensure required directories exist ────────────────────────────────────
    # uploads/ and chroma_db/ are git-ignored but must exist at runtime.
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["CHROMA_DB_PATH"], exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    # Configure a simple console logger.  In production you would replace this
    # with a structured logger or ship logs to CloudWatch.
    log_level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("RAG Document Q&A Chatbot starting up")

    # ── Request logging ───────────────────────────────────────────────────────
    # Log every incoming request and its response time.
    # g is Flask's per-request context object — safe to store state on it.
    # We avoid logging static file requests to reduce noise.

    @app.before_request
    def _log_request_start() -> None:
        """Record the request start time before the handler runs."""
        g.start_time = time.monotonic()
        # Skip logging for static assets — they are high-volume and uninteresting.
        if not request.path.startswith("/static/"):
            logger.info("→ %s %s", request.method, request.path)

    @app.after_request
    def _log_request_end(response):
        """Log the response status and duration after the handler completes."""
        if not request.path.startswith("/static/"):
            duration_ms = (time.monotonic() - g.start_time) * 1000
            logger.info(
                "← %s %s %d (%.1f ms)",
                request.method,
                request.path,
                response.status_code,
                duration_ms,
            )
        return response

    # ── Centralized error handlers ────────────────────────────────────────────
    # Returning JSON for all errors keeps the API consistent — clients always
    # receive a predictable {"error": "..."} body regardless of the error type.
    # Internal details (stack traces, file paths) are never exposed to the caller.

    @app.errorhandler(400)
    def bad_request(error):
        """400 Bad Request — malformed input from the client."""
        logger.warning("400 Bad Request: %s %s", request.method, request.path)
        return jsonify({"error": "Bad request"}), 400

    @app.errorhandler(404)
    def not_found(error):
        """404 Not Found — the requested URL does not exist."""
        logger.warning("404 Not Found: %s %s", request.method, request.path)
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """405 Method Not Allowed — wrong HTTP verb for this endpoint."""
        logger.warning(
            "405 Method Not Allowed: %s %s", request.method, request.path
        )
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """413 Payload Too Large — upload exceeds MAX_CONTENT_LENGTH.

        Flask raises this automatically when the incoming body is larger than
        app.config['MAX_CONTENT_LENGTH'].  We catch it here to return JSON
        instead of Flask's default HTML error page.
        """
        max_mb = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
        logger.warning("413 Payload Too Large: %s %s", request.method, request.path)
        return (
            jsonify({"error": f"File too large. Maximum upload size is {max_mb} MB"}),
            413,
        )

    @app.errorhandler(500)
    def internal_server_error(error):
        """500 Internal Server Error — unhandled exception in application code.

        We log the full error here (server-side only) and return a safe,
        generic message to the client.  Never expose stack traces externally.
        """
        logger.error(
            "500 Internal Server Error: %s %s — %s",
            request.method,
            request.path,
            str(error),
        )
        return jsonify({"error": "An internal server error occurred"}), 500

    # ── Register blueprints / routes ──────────────────────────────────────────
    from app.routes import main
    app.register_blueprint(main)

    logger.info("Application configured successfully")
    return app
