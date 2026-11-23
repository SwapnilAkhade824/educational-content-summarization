"""Tests for LLM summarization (mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest

from google.genai import errors as genai_errors
from summarization.llm_summarizer import (
    LLMConfigError,
    LLMError,
    LLMSummaryResult,
    TopicInterpretation,
    chunk_text,
    generate_summary,
    interpret_topics,
)
from utils.config import LLMConfig


class FakeClient:
    def __init__(self, text: str = "Generated summary", fail_on_call: int = 0) -> None:
        self.text = text
        self.fail_on_call = fail_on_call
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail_on_call and len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated api failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))]
        )


CONFIGURED = LLMConfig(
    api_key="test-key",
    model_id="test-model",
    max_tokens=100,
    temperature=0.2,
    timeout_seconds=60,
)


def test_generate_summary_single_chunk() -> None:
    client = FakeClient(text="Short summary output")
    result = generate_summary("Small educational text.", summary_type="short", config=CONFIGURED, client=client)
    assert isinstance(result, LLMSummaryResult)
    assert result.summary_text == "Short summary output"
    assert result.generated_text == result.summary_text
    assert result.chunk_count == 1
    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 100


def test_generate_summary_each_type() -> None:
    for stype in ["short", "detailed", "bullets", "definitions", "revision"]:
        client = FakeClient(text=f"{stype} output")
        result = generate_summary("Content.", summary_type=stype, config=CONFIGURED, client=client)
        assert result.summary_type == stype
        assert result.summary_text == f"{stype} output"


def test_missing_config_raises() -> None:
    unconfigured = LLMConfig(base_url=None, model_id=None, max_tokens=100, temperature=0.2, timeout_seconds=60)
    client = FakeClient()
    with pytest.raises(LLMConfigError):
        generate_summary("Content.", config=unconfigured, client=client)


def test_empty_input_raises() -> None:
    client = FakeClient()
    with pytest.raises(LLMError):
        generate_summary("", config=CONFIGURED, client=client)
    with pytest.raises(LLMError):
        generate_summary("   ", config=CONFIGURED, client=client)


def test_malformed_response_raises() -> None:
    class BadClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[])))

    with pytest.raises(LLMError):
        generate_summary("Content.", config=CONFIGURED, client=BadClient())


def test_empty_response_raises() -> None:
    class EmptyClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="   "))])
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=EmptyClient())
    assert "Empty model response" in str(exc_info.value)


def test_api_exception_raises() -> None:
    client = FakeClient(fail_on_call=1)
    with pytest.raises(LLMError):
        generate_summary("Content.", config=CONFIGURED, client=client)


def test_chunking() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    assert "Sentence one." in joined
    assert "Sentence four." in joined


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_long_document_combines_chunks() -> None:
    text = ("This is a sentence about machine learning. " * 20).strip()
    client = FakeClient(text="chunk summary")
    result = generate_summary(text, summary_type="short", config=CONFIGURED, client=client, chunk_size=100, overlap=20)
    assert result.chunk_count > 1
    assert len(client.calls) == result.chunk_count + 1
    assert result.summary_text == "chunk summary"


def test_chunk_failure_raises() -> None:
    text = ("Sentence one. " * 20).strip()
    client = FakeClient(fail_on_call=2)
    with pytest.raises(LLMError):
        generate_summary(text, config=CONFIGURED, client=client, chunk_size=50, overlap=10)


def _genai_client_error(status_code: int, message: str, status: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(
        status_code,
        {"error": {"code": status_code, "message": message, "status": status}},
    )


def _genai_server_error(status_code: int, message: str, status: str) -> genai_errors.ServerError:
    return genai_errors.ServerError(
        status_code,
        {"error": {"code": status_code, "message": message, "status": status}},
    )


def test_client_error_mapping() -> None:
    class AccessDeniedClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        _genai_client_error(401, "no access", "UNAUTHENTICATED")
                    )
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=AccessDeniedClient())
    assert "Authentication failed" in str(exc_info.value)


def test_malformed_response_keyerror() -> None:
    class MalformedClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: {"foo": "bar"}))

    with pytest.raises(LLMError):
        generate_summary("Content.", config=CONFIGURED, client=MalformedClient())


def test_network_error_mapping() -> None:
    class NetClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(httpx.ConnectError("connection failed"))
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=NetClient())
    assert "Network or connection" in str(exc_info.value)


def test_rate_limit_mapping() -> None:
    class ThrottleClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        _genai_client_error(429, "slow down", "RESOURCE_EXHAUSTED")
                    )
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=ThrottleClient())
    assert "Rate limited" in str(exc_info.value)


def test_timeout_mapping() -> None:
    class TimeoutClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("timed out"))
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=TimeoutClient())
    assert "timed out" in str(exc_info.value).lower()


def test_invalid_model_mapping() -> None:
    class ModelClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        _genai_client_error(404, "missing model not found", "NOT_FOUND")
                    )
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=ModelClient())
    assert "Invalid model or request" in str(exc_info.value)


def test_server_error_mapping() -> None:
    class ServerClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        _genai_server_error(500, "server down", "INTERNAL")
                    )
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=ServerClient())
    assert "Service unavailable" in str(exc_info.value)


def test_token_limit_mapping() -> None:
    class TokenClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        _genai_client_error(400, "context length exceeded", "INVALID_ARGUMENT")
                    )
                )
            )

    with pytest.raises(LLMError) as exc_info:
        generate_summary("Content.", config=CONFIGURED, client=TokenClient())
    assert "Token limit exceeded" in str(exc_info.value)


def test_interpret_topics() -> None:
    class Topic:
        def __init__(self) -> None:
            self.topic_id = 0
            self.keywords = ["neural", "network"]
            self.weights = [0.4, 0.2]

    client = FakeClient(text="Title: Neural Networks\nExplanation: This topic concerns neural networks.")
    result = interpret_topics([Topic()], config=CONFIGURED, client=client)
    assert len(result) == 1
    assert isinstance(result[0], TopicInterpretation)
    assert result[0].title == "Neural Networks"
    assert "neural networks" in result[0].explanation.lower()
    assert result[0].llm_assisted is True
    assert result[0].keywords == ["neural", "network"]


def test_interpret_multiple_topics_single_batched_request_and_markdown_cleanup() -> None:
    topics = [
        SimpleNamespace(topic_id=0, keywords=["information", "sentiment", "extraction"], weights=[0.3, 0.2, 0.1]),
        SimpleNamespace(topic_id=1, keywords=["neural", "network", "layer"], weights=[0.4, 0.3, 0.2]),
        SimpleNamespace(topic_id=2, keywords=["summarization", "textrank", "graph"], weights=[0.5, 0.3, 0.2]),
    ]
    batched_response = (
        "### LLM-Assisted Topic Interpretation\n\n"
        "Topic 0:\n"
        "**Title:** Information and Sentiment Extraction\n"
        "**Explanation:** Focuses on extracting entities and sentiment polarity.\n\n"
        "Topic 1:\n"
        "Title: Deep Neural Network Architectures\n"
        "Explanation: Covers multilayer neural networks and hidden layer training.\n\n"
        "Topic 2:\n"
        "Title: Graph-Based Extractive Summarization\n"
        "Explanation: Uses TextRank graph centrality to rank sentences.\n"
    )
    client = FakeClient(text=batched_response)
    result = interpret_topics(topics, config=CONFIGURED, client=client)
    # Must make only 1 single API call for all 3 topics!
    assert len(client.calls) == 1
    assert len(result) == 3
    assert result[0].title == "Information and Sentiment Extraction"
    assert "LLM-Assisted" not in result[0].title
    assert "Title:" not in result[0].explanation
    assert result[1].title == "Deep Neural Network Architectures"
    assert result[2].title == "Graph-Based Extractive Summarization"


# ---------------------------------------------------------------------------
# Cloud Gemini Fallback Chain & Local-Only Ollama Fallback Tests
# ---------------------------------------------------------------------------


def test_primary_gemini_503_falls_back_to_secondary_gemini() -> None:
    """Primary Gemini 503 should seamlessly fall back to the next verified Gemini model."""
    models_called: List[str] = []

    class FallbackGeminiClient:
        def __init__(self) -> None:
            self.models = SimpleNamespace(
                generate_content=self._generate_content,
                count_tokens=lambda **kwargs: SimpleNamespace(total_tokens=50),
            )

        def _generate_content(self, **kwargs: Any) -> Any:
            model = kwargs["model"]
            models_called.append(model)
            if model == "gemini-3.8-flash":
                raise _genai_server_error(503, "High demand", "UNAVAILABLE")
            return SimpleNamespace(text="Fallback Gemini summary output")

    cfg = LLMConfig(
        api_key="test-key",
        model_id="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash", "gemini-3.6-flash"),
        ollama_enabled=False,
    )
    result = generate_summary("Educational content.", summary_type="short", config=cfg, client=FallbackGeminiClient())
    assert result.summary_text == "Fallback Gemini summary output"
    assert result.model_id == "gemini-3.5-flash (cloud fallback)"
    assert "gemini-3.8-flash" in models_called
    assert "gemini-3.5-flash" in models_called
    assert "gemini-3.6-flash" not in models_called


def test_all_gemini_fail_falls_back_to_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """When all Gemini models fail (503/429) and local Ollama is reachable, Ollama answers."""
    import summarization.llm_client as client_mod

    class AlwaysFailGeminiClient:
        def __init__(self) -> None:
            self.models = SimpleNamespace(
                generate_content=self._generate_content,
                count_tokens=lambda **kwargs: SimpleNamespace(total_tokens=40),
            )

        def _generate_content(self, **kwargs: Any) -> Any:
            if kwargs["model"] == "gemini-3.8-flash":
                raise _genai_server_error(503, "Service unavailable", "UNAVAILABLE")
            raise _genai_client_error(429, "Quota exceeded", "RESOURCE_EXHAUSTED")

    monkeypatch.setattr(
        client_mod,
        "check_ollama_health",
        lambda base_url, timeout=1.5: (True, ["qwen2.5:1.5b-instruct", "llama3.2:3b"]),
    )
    monkeypatch.setattr(
        client_mod,
        "ollama_chat_completion",
        lambda base_url, model, messages, temperature=0.2, timeout=120.0: (
            f"Local summary from {model}"
        ),
    )

    cfg = LLMConfig(
        api_key="test-key",
        model_id="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash",),
        ollama_enabled=True,
        ollama_model="qwen2.5:1.5b-instruct",
    )
    result = generate_summary("Educational content.", summary_type="short", config=cfg, client=AlwaysFailGeminiClient())
    assert result.summary_text == "Local summary from qwen2.5:1.5b-instruct"
    assert result.model_id == "ollama/qwen2.5:1.5b-instruct (local fallback)"


def test_all_gemini_fail_and_ollama_unreachable_raises_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit Cloud simulation: all Gemini models fail & localhost Ollama unreachable fails fast."""
    import summarization.llm_client as client_mod

    class AlwaysFailGeminiClient:
        def __init__(self) -> None:
            self.models = SimpleNamespace(
                generate_content=lambda **kwargs: (_ for _ in ()).throw(
                    _genai_server_error(503, "Overloaded", "UNAVAILABLE")
                ),
                count_tokens=lambda **kwargs: SimpleNamespace(total_tokens=40),
            )

    health_calls: List[tuple[str, float]] = []

    def _unreachable(base_url: str, timeout: float = 1.5):
        health_calls.append((base_url, timeout))
        return False, []

    monkeypatch.setattr(client_mod, "check_ollama_health", _unreachable)

    cfg = LLMConfig(
        api_key="test-key",
        model_id="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash",),
        ollama_enabled=True,
    )
    with pytest.raises(LLMError) as exc_info:
        generate_summary("Educational content.", summary_type="short", config=cfg, client=AlwaysFailGeminiClient())
    assert "503" in str(exc_info.value) or "unavailable" in str(exc_info.value).lower()
    assert len(health_calls) == 1
    assert health_calls[0][1] <= 1.5


def test_gemini_4_omits_temperature_and_uses_low_thinking() -> None:
    """Gemini 4 models must omit temperature and never pass minimal thinking."""
    captured: List[dict] = []

    class CaptureClient:
        def __init__(self) -> None:
            self.models = SimpleNamespace(
                generate_content=self._generate_content,
                count_tokens=lambda **kwargs: SimpleNamespace(total_tokens=30),
            )

        def _generate_content(self, **kwargs: Any) -> Any:
            captured.append(kwargs)
            return SimpleNamespace(text="Gemini 4 output")

    cfg = LLMConfig(
        api_key="test-key",
        model_id="gemini-4-hydrogen",
        thinking_level="minimal",
        temperature=0.2,
        ollama_enabled=False,
    )
    result = generate_summary("Educational content.", summary_type="short", config=cfg, client=CaptureClient())
    assert result.summary_text == "Gemini 4 output"
    gen_cfg = captured[0]["config"]
    assert gen_cfg.temperature is None
    assert str(gen_cfg.thinking_config.thinking_level).lower().endswith("low")


def test_select_ollama_model_preference_order() -> None:
    from summarization.llm_client import select_ollama_model

    assert select_ollama_model(["llama3:8b", "qwen2.5:1.5b-instruct"], "qwen2.5:1.5b-instruct") == "qwen2.5:1.5b-instruct"
    assert select_ollama_model(["llama3:8b", "deepseek:1.5b-instruct"], "qwen2.5:1.5b-instruct") == "deepseek:1.5b-instruct"
    assert select_ollama_model(["nomic-embed-text:latest", "qwen3.5:4b"], "qwen2.5:1.5b-instruct") == "qwen3.5:4b"
    assert select_ollama_model([], "qwen2.5:1.5b-instruct") is None


