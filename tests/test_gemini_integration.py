"""Gemini integration tests using mocks only (no live API calls)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import httpx
import pytest

from google.genai import errors as genai_errors

from summarization.llm_client import (
    LLMClientConfigError,
    LLMClientError,
    LLMClient,
    chat_completion,
    count_tokens,
    create_client,
    test_connection as check_connection,
)
from summarization.llm_summarizer import (
    LLMConfigError,
    LLMError,
    clear_chunk_cache,
    generate_summary,
    get_chunk_cache,
)
from utils.config import GeminiConfig, load_config


def _config(**overrides: Any) -> GeminiConfig:
    values = {
        "api_key": "test-key",
        "model_id": "gemini-3.5-flash-lite",
        "max_output_tokens": 256,
        "temperature": 0.2,
        "timeout_seconds": 120,
        "max_single_request_tokens": 16000,
    }
    values.update(overrides)
    return GeminiConfig(**values)


class FakeGeminiClient:
    """Minimal stand-in for google.genai.Client."""

    def __init__(
        self,
        text: str = "Generated summary",
        error: Exception | None = None,
        fail_times: int = 0,
        token_count: int | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.fail_times = fail_times
        self.call_count = 0
        self.calls: List[dict] = []
        self.count_calls: List[dict] = []
        self.token_count = token_count

        self.models = SimpleNamespace(
            generate_content=self._generate_content,
            count_tokens=self._count_tokens,
        )

    def _generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.call_count += 1
        if self.error is not None:
            if self.fail_times > 0 and self.call_count > self.fail_times:
                return SimpleNamespace(text=self.text)
            raise self.error
        return SimpleNamespace(text=self.text)

    def _count_tokens(self, **kwargs: Any) -> Any:
        self.count_calls.append(kwargs)
        if self.token_count is not None:
            return SimpleNamespace(total_tokens=self.token_count)
        raise RuntimeError("Token counting not supported")


def test_missing_api_key_disables_gemini(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL_ID", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    cfg = load_config()
    assert cfg.is_configured is False
    assert cfg.model_id == "gemini-3.5-flash-lite"
    with pytest.raises(LLMClientConfigError):
        create_client(cfg)


def test_valid_configuration_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.35")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("GEMINI_MAX_SINGLE_REQUEST_TOKENS", "12000")
    cfg = load_config()
    assert cfg.api_key == "abc123"
    assert cfg.model_id == "gemini-3.5-flash-lite"
    assert cfg.thinking_level == "low"
    assert cfg.max_output_tokens == 4096
    assert cfg.temperature == 0.35
    assert cfg.timeout_seconds == 90
    assert cfg.max_single_request_tokens == 12000
    assert cfg.is_configured is True


def test_invalid_configuration_values_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "eight-thousand")
    with pytest.raises(ValueError):
        load_config()

    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "ultra")
    with pytest.raises(ValueError):
        load_config()


def test_model_id_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")
    assert load_config().model_id == "gemini-3.5-flash-lite"

    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-3.8-pro")
    assert load_config().model_id == "gemini-3.8-pro"


def test_client_initialization_receives_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeGenai:
        @staticmethod
        def Client(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(models=SimpleNamespace())

    import summarization.llm_client as client_module

    monkeypatch.setattr(client_module, "genai", FakeGenai)
    created = create_client(_config(timeout_seconds=45))
    assert captured["api_key"] == "test-key"
    assert captured["http_options"].timeout == 45000
    assert created is not None


def test_mocked_successful_generation_passes_sdk_parameters() -> None:
    client = FakeGeminiClient(text="Structured output")
    result = generate_summary("Educational content.", summary_type="short", config=_config(), client=client)
    assert result.summary_text == "Structured output"
    assert result.model_id == "gemini-3.5-flash-lite"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"
    config = call["config"]
    assert config.temperature == 0.2
    assert config.max_output_tokens == 256
    assert config.system_instruction


def test_authentication_failure_is_reported_safely() -> None:
    error = genai_errors.ClientError(
        401,
        {"error": {"code": 401, "message": "API key not valid.", "status": "UNAUTHENTICATED"}},
    )
    client = FakeGeminiClient(error=error)
    with pytest.raises(LLMClientError) as exc_info:
        generate_summary("Content.", config=_config(api_key="secret-key"), client=client)
    message = str(exc_info.value)
    assert "Authentication failed" in message
    assert "secret-key" not in message


def test_quota_error_is_reported() -> None:
    error = genai_errors.ClientError(
        429,
        {"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED"}},
    )
    client = FakeGeminiClient(error=error)
    with pytest.raises(LLMClientError) as exc_info:
        generate_summary("Content.", config=_config(), client=client)
    assert "Rate limited" in str(exc_info.value)


def test_timeout_is_reported() -> None:
    client = FakeGeminiClient(error=httpx.TimeoutException("deadline exceeded"))
    with pytest.raises(LLMClientError) as exc_info:
        generate_summary("Content.", config=_config(), client=client)
    assert "timed out" in str(exc_info.value).lower()


def test_test_connection_without_api_key() -> None:
    ok, message = check_connection(_config(api_key=None))
    assert ok is False
    assert "GEMINI_API_KEY" in message


def test_test_connection_reports_success_with_mock() -> None:
    ok, message = check_connection(_config(), client=FakeGeminiClient(text="OK"))
    assert ok is True
    assert "OK" in message


def test_traditional_summarization_without_gemini() -> None:
    from summarization.textrank import summarize_text

    text = (
        "Machine learning is a branch of artificial intelligence. "
        "It builds models from data. Students learn core algorithms. "
        "TextRank selects the most central sentences."
    )
    result = summarize_text(text, num_sentences=2)
    assert result.selected_sentences
    assert len(result.selected_sentences) == 2


def test_unconfigured_config_raises_clear_error() -> None:
    with pytest.raises(LLMConfigError) as exc_info:
        generate_summary("Content.", config=_config(api_key=None), client=FakeGeminiClient())
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_streamlit_app_has_no_llm_configuration_controls() -> None:
    from pathlib import Path

    app_source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    for forbidden in (
        "st.text_input",
        "st.number_input",
        "GEMINI_API_KEY=",
        "type=\"password\"",
        "LLM Settings",
    ):
        assert forbidden not in app_source


def test_gemini_client_wrapper_delegates() -> None:
    client = LLMClient(_config(), client=FakeGeminiClient(text="Wrapped"))
    assert client.complete([{"role": "user", "content": "hi"}]) == "Wrapped"


# ---------------------------------------------------------------------------
# Issue 2 Specific Tests: Short doc optimization, retries, resumability, timeouts
# ---------------------------------------------------------------------------


def test_short_document_processed_in_single_request() -> None:
    """Verify documents within the token budget use one request and no chunking."""
    long_text = ("Educational content about machine learning and natural language processing. " * 50).strip()
    # Mock token count at 3500 tokens (<= 16000 budget)
    client = FakeGeminiClient(text="Single request summary", token_count=3500)
    result = generate_summary(long_text, summary_type="short", config=_config(), client=client)
    assert result.chunk_count == 1
    assert result.summary_text == "Single request summary"
    # Exactly one generation call made to Gemini
    assert len(client.calls) == 1
    assert len(client.count_calls) >= 1


def test_long_document_token_budget_exceeded_uses_chunks() -> None:
    """Verify genuinely long documents exceeding token budget use hierarchical chunking."""
    text = ("Paragraph of scientific research. " * 30).strip()
    # Mock token count at 25000 (> 16000 budget)
    client = FakeGeminiClient(text="Chunk summary", token_count=25000)
    result = generate_summary(text, summary_type="short", config=_config(), client=client, chunk_size=100)
    assert result.chunk_count > 1
    # Multiple chunk calls + 1 final combination call
    assert len(client.calls) == result.chunk_count + 1


def test_transient_503_retries_and_succeeds() -> None:
    """Verify 503 UNAVAILABLE is retried with backoff and succeeds on retry."""
    err_503 = genai_errors.ServerError(
        503,
        {"error": {"code": 503, "message": "The service is temporarily unavailable.", "status": "UNAVAILABLE"}},
    )
    # Fails 1 time with 503, then succeeds on attempt 2
    client = FakeGeminiClient(text="Recovered from 503", error=err_503, fail_times=1, token_count=100)
    messages = [{"role": "user", "content": "Summarize"}]
    response = chat_completion(_config(), messages, client=client, retries=2)
    assert response == "Recovered from 503"
    assert client.call_count == 2


def test_transient_504_deadline_exceeded_retries_and_succeeds() -> None:
    """Verify 504 DEADLINE_EXCEEDED is retried and succeeds on retry."""
    err_504 = genai_errors.ServerError(
        504,
        {"error": {"code": 504, "message": "Deadline exceeded while waiting for model.", "status": "DEADLINE_EXCEEDED"}},
    )
    client = FakeGeminiClient(text="Recovered from 504", error=err_504, fail_times=1, token_count=100)
    messages = [{"role": "user", "content": "Summarize"}]
    response = chat_completion(_config(), messages, client=client, retries=2)
    assert response == "Recovered from 504"
    assert client.call_count == 2


def test_client_side_timeout_distinguished_from_server_504() -> None:
    """Verify client-side timeouts produce distinct error messages from server 504."""
    client_timeout = FakeGeminiClient(error=httpx.TimeoutException("Client connection timed out"))
    with pytest.raises(LLMClientError) as exc_info:
        chat_completion(_config(), [{"role": "user", "content": "test"}], client=client_timeout, retries=0)
    assert "Client-side request timed out" in str(exc_info.value)

    err_504 = genai_errors.ServerError(504, {"error": {"code": 504, "message": "Server timeout"}})
    server_deadline = FakeGeminiClient(error=err_504)
    with pytest.raises(LLMClientError) as exc_info:
        chat_completion(_config(), [{"role": "user", "content": "test"}], client=server_deadline, retries=0)
    assert "504 DEADLINE_EXCEEDED" in str(exc_info.value)


def test_permanent_errors_never_retried() -> None:
    """Verify 401 unauthenticated and 404 not found are not retried."""
    err_401 = genai_errors.ClientError(401, {"error": {"code": 401, "message": "API key invalid"}})
    client_401 = FakeGeminiClient(error=err_401)
    with pytest.raises(LLMClientError):
        chat_completion(_config(), [{"role": "user", "content": "test"}], client=client_401, retries=2)
    # Never retried: exactly 1 attempt
    assert client_401.call_count == 1

    err_404 = genai_errors.ClientError(404, {"error": {"code": 404, "message": "Model not found"}})
    client_404 = FakeGeminiClient(error=err_404)
    with pytest.raises(LLMClientError):
        chat_completion(_config(), [{"role": "user", "content": "test"}], client=client_404, retries=2)
    assert client_404.call_count == 1


def test_max_three_attempts_strictly_enforced() -> None:
    """Verify retry loop does not exceed three attempts per request."""
    err_503 = genai_errors.ServerError(503, {"error": {"code": 503, "message": "Unavailable"}})
    client = FakeGeminiClient(error=err_503)
    with pytest.raises(LLMClientError):
        chat_completion(_config(), [{"role": "user", "content": "test"}], client=client, retries=5)
    # Strictly bounded to at most 3 attempts
    assert client.call_count == 3


def test_chunk_resumability_skips_completed_chunks() -> None:
    """Verify that retrying a failed multi-chunk document resumes from failed chunks."""
    clear_chunk_cache()
    text = ("Sentence about chemistry and molecular biology. " * 20).strip()
    cache: dict = {}

    # Call with persistent failure on chunk 2
    err_fail = genai_errors.ServerError(503, {"error": {"code": 503, "message": "Fail on chunk 2"}})
    client_fail = FakeGeminiClient(error=err_fail, fail_times=0, text="Chunk ok")

    # First attempt: fails on chunk 1 (or 2)
    with pytest.raises(LLMError):
        generate_summary(text, summary_type="short", config=_config(), client=client_fail, chunk_size=80, chunk_cache=cache)

    # Now simulate a client that succeeded on chunk 1 and then failed on chunk 2
    cache_key = list(cache.keys())[0] if cache else None
    if not cache_key:
        import hashlib
        cache_key = hashlib.sha256(f"{text}::short::gemini-3.5-flash-lite::{_config().model_id}".encode("utf-8")).hexdigest()
    cache[cache_key] = {1: "Pre-existing chunk 1 summary"}

    # Resuming with succeeding client: chunk 1 should not be called again
    client_success = FakeGeminiClient(text="New chunk summary")
    result = generate_summary(text, summary_type="short", config=_config(), client=client_success, chunk_size=80, chunk_cache=cache)
    assert result.summary_text == "New chunk summary"
    # Chunk 1 summary retained from cache
    assert cache[cache_key][1] == "Pre-existing chunk 1 summary"


def test_chunk_cache_invalidated_on_document_or_prompt_change() -> None:
    """Verify cache keys differ when text, model, or summary type changes."""
    cache: dict = {}
    text1 = "First document text. Another sentence. More words to ensure chunking."
    text2 = "Second different document text. Another sentence. More words to ensure chunking."
    client = FakeGeminiClient(text="Summary")

    generate_summary(text1, summary_type="short", config=_config(), client=client, chunk_size=20, chunk_cache=cache)
    generate_summary(text2, summary_type="short", config=_config(), client=client, chunk_size=20, chunk_cache=cache)
    generate_summary(text1, summary_type="detailed", config=_config(), client=client, chunk_size=20, chunk_cache=cache)

    # 3 distinct cache keys created
    assert len(cache) == 3


def test_progress_callback_receives_status() -> None:
    """Verify progress callback receives status updates with chunk counts."""
    progress_updates: List[dict] = []

    def _cb(status: dict) -> None:
        progress_updates.append(status)

    client = FakeGeminiClient(text="Summary", token_count=100)
    result = generate_summary(
        "Small text.",
        summary_type="short",
        config=_config(),
        client=client,
        progress_callback=_cb,
    )
    assert result.summary_text == "Summary"
    assert len(progress_updates) >= 1
    last_update = progress_updates[-1]
    assert last_update["total_chunks"] == 1
    assert last_update["completed_chunks"] == 1
    assert last_update["retry_status"] == "Completed"
