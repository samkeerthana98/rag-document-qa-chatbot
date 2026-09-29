"""
app/services/ — Application-level service layer.

Services sit between the Flask routes and the RAG pipeline components.
They orchestrate multiple RAG modules to fulfil a single user request.

  qa_service.py  — Orchestrates: retrieve chunks → build prompt → call LLM

Having a service layer keeps route handlers thin (they only deal with HTTP
parsing and response formatting) and keeps business logic out of the RAG
modules (which should remain reusable and framework-agnostic).

This module will be implemented in Phase 9.
"""
