"""
app/rag/llm.py — Local LLM integration via Ollama.

This module wraps the LangChain OllamaLLM client so the rest of the pipeline
does not depend on Ollama-specific code.  If we ever add a second LLM provider
(e.g. a cloud API), we add it here and change only this file.

Interview explanation:
  LangChain provides a common interface for different LLM providers.
  OllamaLLM talks to a locally running Ollama HTTP service (default port 11434).
  The model (llama3.2) runs entirely on your machine — no API key, no cost,
  no data sent to external servers.

  We keep the LLM wrapper separate from the prompt-building logic (in
  qa_service.py) so each piece is independently testable and replaceable.

Public API:
  get_llm(model, base_url) -> OllamaLLM
      Returns a configured LangChain OllamaLLM instance.

  LLMUnavailableError
      Raised when Ollama cannot be reached (not running, wrong port, etc.).
      The /api/ask route catches this and returns a clean 503 response.
"""

import logging

from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when the Ollama service cannot be reached.

    Distinct from unexpected errors so the route can return 503 Service
    Unavailable with a user-friendly message instead of a generic 500.
    """


def get_llm(model: str, base_url: str) -> OllamaLLM:
    """Create a configured OllamaLLM instance.

    This does NOT make a network connection — it just creates the client
    object.  The connection happens when .invoke() is first called.

    Args:
        model:    Ollama model name (e.g. "llama3.2").
                  Must be pulled locally: ollama pull llama3.2
        base_url: URL of the Ollama HTTP service.
                  Default: "http://localhost:11434"
                  Docker: "http://host.docker.internal:11434"

    Returns:
        A LangChain OllamaLLM instance ready to call.
    """
    logger.debug("Creating OllamaLLM client: model=%s, base_url=%s", model, base_url)
    return OllamaLLM(model=model, base_url=base_url)


def invoke_llm(llm: OllamaLLM, prompt: str) -> str:
    """Send a prompt to the LLM and return its text response.

    Wraps llm.invoke() to:
      - Convert connection errors into LLMUnavailableError (clean 503).
      - Log the prompt length and response length for debugging.
      - Strip leading/trailing whitespace from the response.

    Args:
        llm:    A configured OllamaLLM instance from get_llm().
        prompt: The fully assembled prompt string including context + question.

    Returns:
        The LLM's text response, stripped of leading/trailing whitespace.

    Raises:
        LLMUnavailableError: If Ollama is not running or unreachable.
        Exception:           Any unexpected error from the LLM is re-raised.
    """
    logger.info("Sending prompt to LLM (%d chars)", len(prompt))
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        # langchain-ollama raises various errors when the service is down.
        # We catch them all here and re-raise as LLMUnavailableError if the
        # message suggests a connection problem.
        err_str = str(exc).lower()
        if any(kw in err_str for kw in ("connection", "refused", "connect", "timeout", "unreachable")):
            logger.error("Ollama service is unavailable: %s", exc)
            raise LLMUnavailableError(
                "The Ollama LLM service is not available. "
                "Please ensure Ollama is running: ollama serve"
            ) from exc
        logger.exception("Unexpected error from LLM: %s", exc)
        raise

    answer = str(response).strip()
    logger.info("LLM response received (%d chars)", len(answer))
    return answer
