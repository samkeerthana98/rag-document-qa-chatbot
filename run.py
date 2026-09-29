"""
run.py — Application entry point.

Run this file to start the Flask development server:
    python run.py

For production, use a WSGI server (e.g. gunicorn):
    gunicorn -w 4 "app:create_app()"
"""

import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Read host and port from environment so they can be overridden without
    # changing code.  Defaults are fine for local development.
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    print(f"Starting RAG Document Q&A Chatbot on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
