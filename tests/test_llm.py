"""
tests/test_llm.py — Unit tests for app/rag/llm.py.

These tests exercise the LLM module in isolation — no Ollama service is
required.  The OllamaLLM client and its .invoke() call are mocked so:
  - Tests run offline (no network, no external process).
  - Tests run fast (no model inference).
  - The logic of get_llm() and invoke_llm() is verified independently of
    the rest of the RAG pipeline.

Phase 8 requirements verified:
  - Ollama integration exists (OllamaLLM is imported and used).
  - llama3.2 model is configurable (get_llm accepts model parameter).
  - Ollama URL is configurable (get_llm accepts base_url parameter).
  - LLM implementation is isolated (all logic lives in app/rag/llm.py).
  - LLM errors are handled cleanly (connection errors → LLMUnavailableError).
  - Non-connection errors propagate unchanged.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.rag.llm import LLMUnavailableError, get_llm, invoke_llm


# ── LLMUnavailableError ───────────────────────────────────────────────────────

class TestLLMUnavailableError:
    def test_is_exception_subclass(self):
        """LLMUnavailableError must be a proper Exception subclass."""
        assert issubclass(LLMUnavailableError, Exception)

    def test_can_be_raised_and_caught(self):
        """Can be raised and caught by its own type."""
        with pytest.raises(LLMUnavailableError):
            raise LLMUnavailableError("Ollama is not running")

    def test_message_preserved(self):
        """The error message is accessible from the exception."""
        try:
            raise LLMUnavailableError("custom message")
        except LLMUnavailableError as exc:
            assert "custom message" in str(exc)

    def test_not_caught_as_value_error(self):
        """LLMUnavailableError must NOT be a ValueError (it's a service error)."""
        assert not issubclass(LLMUnavailableError, ValueError)


# ── get_llm ───────────────────────────────────────────────────────────────────

class TestGetLlm:
    def test_returns_ollama_llm_instance(self):
        """get_llm() must return a LangChain OllamaLLM object."""
        from langchain_ollama import OllamaLLM
        llm = get_llm(model="llama3.2", base_url="http://localhost:11434")
        assert isinstance(llm, OllamaLLM)

    def test_model_configured(self):
        """The returned instance must use the model name that was passed in."""
        llm = get_llm(model="llama3.2", base_url="http://localhost:11434")
        assert llm.model == "llama3.2"

    def test_base_url_configured(self):
        """The returned instance must use the base_url that was passed in."""
        llm = get_llm(model="llama3.2", base_url="http://localhost:11434")
        assert "11434" in llm.base_url

    def test_custom_model_accepted(self):
        """Any model name can be passed — the function does not hard-code llama3.2."""
        llm = get_llm(model="mistral", base_url="http://localhost:11434")
        assert llm.model == "mistral"

    def test_custom_base_url_accepted(self):
        """Custom Ollama URL (e.g. Docker) must be accepted."""
        llm = get_llm(
            model="llama3.2", base_url="http://host.docker.internal:11434"
        )
        assert "host.docker.internal" in llm.base_url

    def test_does_not_connect_on_creation(self):
        """Creating the LLM client must not make any network calls.

        OllamaLLM is lazy — it connects only when .invoke() is called.
        This ensures get_llm() is safe to call even when Ollama is offline.
        """
        # If this raises a connection error, the implementation is wrong.
        # Using a deliberately unreachable URL to prove no connection is made.
        llm = get_llm(model="llama3.2", base_url="http://127.0.0.1:1")
        # Still an OllamaLLM — no exception raised on construction
        from langchain_ollama import OllamaLLM
        assert isinstance(llm, OllamaLLM)


# ── invoke_llm ────────────────────────────────────────────────────────────────

class TestInvokeLlm:
    def test_returns_string(self):
        """invoke_llm() must return a string."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The answer is 42."
        result = invoke_llm(mock_llm, "What is the answer?")
        assert isinstance(result, str)

    def test_returns_llm_response_text(self):
        """invoke_llm() must return what the LLM produces."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Paris is the capital of France."
        result = invoke_llm(mock_llm, "What is the capital of France?")
        assert result == "Paris is the capital of France."

    def test_strips_leading_trailing_whitespace(self):
        """Responses with surrounding whitespace must be stripped."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "  \n  Answer here.  \n  "
        result = invoke_llm(mock_llm, "Q?")
        assert result == "Answer here."

    def test_passes_prompt_to_llm(self):
        """The exact prompt must be passed to llm.invoke()."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ok"
        prompt = "This is the full assembled prompt."
        invoke_llm(mock_llm, prompt)
        mock_llm.invoke.assert_called_once_with(prompt)

    def test_llm_invoked_exactly_once(self):
        """The LLM must be invoked exactly once per call."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "response"
        invoke_llm(mock_llm, "question")
        assert mock_llm.invoke.call_count == 1


# ── Connection error handling in invoke_llm ───────────────────────────────────

class TestInvokeLlmConnectionErrors:
    """Connection errors must be converted to LLMUnavailableError.

    langchain-ollama raises various exception types when Ollama is not running.
    All of them contain keywords like "connection", "refused", "connect",
    "timeout", or "unreachable".  We test each keyword to confirm the mapping.
    """

    def _failing_llm(self, error_message: str) -> MagicMock:
        """Return a mock LLM whose .invoke() raises an Exception."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception(error_message)
        return mock_llm

    def test_connection_refused_raises_unavailable(self):
        """'connection refused' → LLMUnavailableError."""
        mock_llm = self._failing_llm("Connection refused to http://localhost:11434")
        with pytest.raises(LLMUnavailableError):
            invoke_llm(mock_llm, "prompt")

    def test_connect_keyword_raises_unavailable(self):
        """'connect' (e.g. 'cannot connect') → LLMUnavailableError."""
        mock_llm = self._failing_llm("Failed to connect to server")
        with pytest.raises(LLMUnavailableError):
            invoke_llm(mock_llm, "prompt")

    def test_timeout_raises_unavailable(self):
        """'timeout' → LLMUnavailableError."""
        mock_llm = self._failing_llm("Request timeout connecting to Ollama")
        with pytest.raises(LLMUnavailableError):
            invoke_llm(mock_llm, "prompt")

    def test_unreachable_raises_unavailable(self):
        """'unreachable' → LLMUnavailableError."""
        mock_llm = self._failing_llm("Host unreachable: localhost:11434")
        with pytest.raises(LLMUnavailableError):
            invoke_llm(mock_llm, "prompt")

    def test_refused_keyword_raises_unavailable(self):
        """'refused' → LLMUnavailableError."""
        mock_llm = self._failing_llm("Connection was refused by the server")
        with pytest.raises(LLMUnavailableError):
            invoke_llm(mock_llm, "prompt")

    def test_llm_unavailable_error_message_is_user_friendly(self):
        """The LLMUnavailableError message must be suitable for end users."""
        mock_llm = self._failing_llm("connection refused")
        with pytest.raises(LLMUnavailableError) as exc_info:
            invoke_llm(mock_llm, "prompt")
        msg = str(exc_info.value).lower()
        # Must mention Ollama (so the user knows what to fix) and not expose
        # internal technical details like port numbers or stack traces.
        assert "ollama" in msg

    def test_connection_error_chained_from_original(self):
        """LLMUnavailableError must chain from the original exception (__cause__)."""
        original = Exception("connection refused")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = original
        with pytest.raises(LLMUnavailableError) as exc_info:
            invoke_llm(mock_llm, "prompt")
        assert exc_info.value.__cause__ is original

    def test_non_connection_error_reraises_unchanged(self):
        """Errors unrelated to connectivity must NOT be wrapped in LLMUnavailableError."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("model does not exist: notamodel")
        with pytest.raises(ValueError):
            invoke_llm(mock_llm, "prompt")

    def test_non_connection_error_not_wrapped(self):
        """A generic RuntimeError must propagate as-is, not become LLMUnavailableError."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("out of memory")
        with pytest.raises(RuntimeError):
            invoke_llm(mock_llm, "prompt")

    def test_non_connection_error_is_not_llm_unavailable(self):
        """A logic error from the model must not be mistaken for a service error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("invalid prompt format")
        with pytest.raises(Exception) as exc_info:
            invoke_llm(mock_llm, "prompt")
        # Must NOT be an LLMUnavailableError
        assert not isinstance(exc_info.value, LLMUnavailableError)


# ── Config integration ────────────────────────────────────────────────────────

class TestLlmConfigIntegration:
    """Verify that the app config feeds correctly into get_llm()."""

    def test_default_model_is_llama32(self):
        """The default OLLAMA_MODEL config value must be 'llama3.2'."""
        from app.config import Config
        assert Config.OLLAMA_MODEL == "llama3.2"

    def test_default_base_url_is_localhost(self):
        """The default OLLAMA_BASE_URL must point to localhost:11434."""
        from app.config import Config
        assert "localhost" in Config.OLLAMA_BASE_URL
        assert "11434" in Config.OLLAMA_BASE_URL

    def test_default_provider_is_ollama(self):
        """The default LLM_PROVIDER must be 'ollama'."""
        from app.config import Config
        assert Config.LLM_PROVIDER == "ollama"

    def test_get_llm_uses_config_defaults(self):
        """get_llm() with config defaults produces a valid OllamaLLM."""
        from app.config import Config
        from langchain_ollama import OllamaLLM
        llm = get_llm(model=Config.OLLAMA_MODEL, base_url=Config.OLLAMA_BASE_URL)
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama3.2"
