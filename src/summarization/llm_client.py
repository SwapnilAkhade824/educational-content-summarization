"""Google Gemini LLM client using the official google-genai SDK.

The client has no Streamlit dependency. Configuration is provided through
``GeminiConfig`` (loaded centrally from environment variables).
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
except Exception:  # pragma: no cover - google-genai is a declared dependency
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]

# Optional openai imports retained for backward-compatible test doubles.
try:
    from openai import (
        APIConnectionError as _OpenAIAPIConnectionError,
        APIError as _OpenAIAPIError,
        APIResponseValidationError as _OpenAIAPIResponseValidationError,
        APIStatusError as _OpenAIAPIStatusError,
        APITimeoutError as _OpenAIAPItimeoutError,
        AuthenticationError as _OpenAIAuthenticationError,
        BadRequestError as _OpenAIBadRequestError,
        InternalServerError as _OpenAIInternalServerError,
        NotFoundError as _OpenAINotFoundError,
        PermissionDeniedError as _OpenAIPermissionDeniedError,
        RateLimitError as _OpenAIRateLimitError,
        UnprocessableEntityError as _OpenAIUnprocessableEntityError,
    )
except Exception:  # pragma: no cover - openai is optional
    _OpenAIAPIConnectionError = Exception  # type: ignore[assignment]
    _OpenAIAPIError = Exception  # type: ignore[assignment]
    _OpenAIAPIResponseValidationError = Exception  # type: ignore[assignment]
    _OpenAIAPIStatusError = Exception  # type: ignore[assignment]
    _OpenAIAPItimeoutError = Exception  # type: ignore[assignment]
    _OpenAIAuthenticationError = Exception  # type: ignore[assignment]
    _OpenAIBadRequestError = Exception  # type: ignore[assignment]
    _OpenAIInternalServerError = Exception  # type: ignore[assignment]
    _OpenAINotFoundError = Exception  # type: ignore[assignment]
    _OpenAIPermissionDeniedError = Exception  # type: ignore[assignment]
    _OpenAIRateLimitError = Exception  # type: ignore[assignment]
    _OpenAIUnprocessableEntityError = Exception  # type: ignore[assignment]

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


class LLMClientError(Exception):
    """Raised when an LLM request fails."""


class LLMClientConfigError(LLMClientError):
    """Raised when LLM client configuration is invalid."""


@dataclass
class URLValidationResult:
    ok: bool
    message: str


# Backward-compatible helpers retained (no-op-ish for Gemini, but harmless).
def default_base_url(provider: str) -> Optional[str]:
    return None


def normalize_base_url(url: str) -> str:
    return (url or "").strip()


def validate_base_url(url: str) -> URLValidationResult:
    return URLValidationResult(True, "Gemini does not use a base URL.")


def _import_genai():
    if genai is None:
        raise LLMClientError("google-genai is not installed.")
    return genai


def _thinking_level_value(level: str) -> Any:
    if genai_types is None:
        return level
    try:
        enum_cls = genai_types.ThinkingLevel
        return enum_cls[str(level).strip().upper()]
    except Exception:
        # Fall back to the raw string; SDK may accept it.
        return level


def create_client(config: Any, client: Any = None) -> Any:
    """Create a Gemini client from configuration."""
    if client is not None:
        return client

    genai_cls = _import_genai()

    api_key = getattr(config, "api_key", None)
    if not api_key:
        raise LLMClientConfigError("GEMINI_API_KEY is not configured.")

    timeout_seconds = getattr(config, "timeout_seconds", 120)
    try:
        timeout_ms = int(float(timeout_seconds) * 1000)
    except (TypeError, ValueError):
        timeout_ms = 120000

    http_options = None
    if genai_types is not None:
        try:
            http_options = genai_types.HttpOptions(timeout=timeout_ms)
        except Exception:
            http_options = None

    try:
        if http_options is not None:
            return genai_cls.Client(api_key=api_key, http_options=http_options)
        return genai_cls.Client(api_key=api_key)
    except Exception as exc:
        raise LLMClientError(f"Failed to create Gemini client: {exc}") from exc


def _extract_text_gemini(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fall back to candidates/parts.
    candidates = getattr(response, "candidates", None)
    if candidates:
        collected: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts is None and isinstance(content, dict):
                parts = content.get("parts")
            for part in parts or []:
                part_text = getattr(part, "text", None)
                if part_text is None and isinstance(part, dict):
                    part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    collected.append(part_text.strip())
        if collected:
            return "\n".join(collected).strip()

    # Last resort: dict-like response.
    if isinstance(response, dict):
        text = response.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    raise LLMClientError("Malformed LLM response: missing text.")


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract retry-after delay in seconds if provided by the response or error message."""
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            val = headers.get("retry-after") or headers.get("Retry-After")
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    msg = str(exc)
    match = re.search(r"retry\s+after\s+(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            pass
    return None


def _compute_backoff(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_after: Optional[float] = None,
) -> float:
    """Compute bounded exponential backoff with random jitter."""
    import random

    if retry_after is not None and retry_after > 0:
        return min(retry_after, max_delay)
    delay = min(max_delay, base_delay * (2 ** attempt))
    jitter = random.uniform(0.0, 0.25 * delay)
    return delay + jitter


def _map_exception(exc: Exception) -> LLMClientError:
    # Client-side timeouts must be distinguished from server-side deadline errors.
    if httpx is not None:
        timeout_exc = getattr(httpx, "TimeoutException", None)
        connect_exc = getattr(httpx, "ConnectError", None)
        if timeout_exc is not None and isinstance(exc, timeout_exc):
            return LLMClientError(f"Client-side request timed out: {exc}")
        if connect_exc is not None and isinstance(exc, connect_exc):
            return LLMClientError(f"Network or connection error: {exc}")

    if isinstance(exc, _OpenAIAPItimeoutError):
        return LLMClientError(f"Client-side request timed out: {exc}")

    # Gemini / google-genai errors.
    if genai_errors is not None:
        client_error_cls = getattr(genai_errors, "ClientError", None)
        server_error_cls = getattr(genai_errors, "ServerError", None)
        api_error_cls = getattr(genai_errors, "APIError", None)

        msg = str(exc)
        lower = msg.lower()
        code = getattr(exc, "code", None)

        # Distinguish 504 Deadline Exceeded and 503 Unavailable
        if code == 504 or "504" in msg or "deadline" in lower:
            return LLMClientError(f"Server deadline exceeded (504 DEADLINE_EXCEEDED): {exc}")
        if code == 503 or "503" in msg or "unavailable" in lower:
            return LLMClientError(f"Service unavailable (503 UNAVAILABLE): {exc}")

        if client_error_cls is not None and isinstance(exc, client_error_cls):
            if code in (401, 403) or "api key" in lower or "unauth" in lower or "permission" in lower:
                return LLMClientError(f"Authentication failed: {exc}")
            if code == 429 or "quota" in lower or "resource exhausted" in lower or "rate" in lower:
                return LLMClientError(f"Rate limited (quota exceeded): {exc}")
            if code == 404 or "not found" in lower or ("model" in lower and "invalid" in lower):
                return LLMClientError(f"Invalid model or request: model not found or invalid endpoint: {exc}")
            if "token" in lower or "context" in lower or "length" in lower:
                return LLMClientError(f"Token limit exceeded: {msg}")
            if "param" in lower or "unsupported" in lower or "invalid argument" in lower:
                return LLMClientError(f"Unsupported API parameter: {msg}")
            return LLMClientError(f"Invalid request: {msg}")
        if server_error_cls is not None and isinstance(exc, server_error_cls):
            return LLMClientError(f"Service unavailable: {exc}")
        if api_error_cls is not None and isinstance(exc, api_error_cls):
            status = getattr(exc, "code", None)
            return LLMClientError(f"LLM API error ({status}): {exc}")

    # OpenAI-compatible errors (for test doubles / legacy clients).
    if isinstance(exc, (_OpenAIAuthenticationError, _OpenAIPermissionDeniedError)):
        return LLMClientError(f"Authentication failed: {exc}")
    if isinstance(exc, _OpenAINotFoundError):
        return LLMClientError(f"Invalid model or request: model not found or invalid endpoint: {exc}")
    if isinstance(exc, (_OpenAIBadRequestError, _OpenAIUnprocessableEntityError)):
        message = str(exc)
        lower = message.lower()
        if "token" in lower or "context" in lower or "length" in lower:
            return LLMClientError(f"Token limit exceeded: {message}")
        if "param" in lower or "unsupported" in lower:
            return LLMClientError(f"Unsupported API parameter: {message}")
        return LLMClientError(f"Invalid request: {message}")
    if isinstance(exc, _OpenAIRateLimitError):
        return LLMClientError(f"Rate limited (quota exceeded): {exc}")
    if isinstance(exc, _OpenAIAPIConnectionError):
        return LLMClientError(f"Network or connection error: {exc}")
    if isinstance(exc, _OpenAIInternalServerError):
        return LLMClientError(f"Service unavailable: {exc}")
    if isinstance(exc, _OpenAIAPIResponseValidationError):
        return LLMClientError(f"Malformed LLM response: {exc}")
    if isinstance(exc, _OpenAIAPIStatusError):
        status = getattr(exc, "status_code", None)
        return LLMClientError(f"LLM API error ({status}): {exc}")
    if isinstance(exc, _OpenAIAPIError):
        status = getattr(exc, "status_code", None)
        return LLMClientError(f"LLM API error ({status}): {exc}")

    if isinstance(exc, LLMClientError):
        return exc
    return LLMClientError(f"Unexpected LLM error: {exc}")


def _is_transient(exc: Exception) -> bool:
    """Determine whether an error is transient and eligible for retry.

    Transient: 503 UNAVAILABLE, 504 DEADLINE_EXCEEDED, 429 Rate Limit, temporary network/timeout failures.
    Permanent (never retried): 401/403 Auth failures, 404 Model Not Found, 400 Bad Request / Token limits.
    """
    msg = str(exc)
    lower = msg.lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

    # Permanent failures must NEVER be retried
    if code in (401, 403) or "unauthenticated" in lower or "api key" in lower or "permission" in lower:
        return False
    if code == 404 or "not found" in lower or ("model" in lower and "invalid" in lower):
        return False
    if "token limit" in lower or "context length" in lower or "invalid argument" in lower or "unsupported" in lower:
        return False

    # Transient failures
    if code in (429, 500, 502, 503, 504):
        return True
    if "503" in msg or "unavailable" in lower:
        return True
    if "504" in msg or "deadline" in lower:
        return True
    if "rate" in lower or "quota" in lower or "resource exhausted" in lower:
        return True

    if genai_errors is not None:
        server_error_cls = getattr(genai_errors, "ServerError", None)
        if server_error_cls is not None and isinstance(exc, server_error_cls):
            return True

    transient_types = [
        _OpenAIRateLimitError,
        _OpenAIAPItimeoutError,
        _OpenAIAPIConnectionError,
        _OpenAIInternalServerError,
    ]
    if httpx is not None:
        for name in ("TimeoutException", "ConnectError", "NetworkError"):
            cls = getattr(httpx, name, None)
            if cls is not None:
                transient_types.append(cls)

    return isinstance(exc, tuple(transient_types))


def _complete_once(
    client: Any,
    config: Any,
    messages: List[dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    # Gemini path.
    if hasattr(client, "models") and hasattr(client.models, "generate_content"):
        system_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
        system_instruction = "\n\n".join(p for p in system_parts if p.strip()) or None

        contents: List[Any] = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                continue
            content = m.get("content", "")
            gemini_role = "model" if role in {"assistant", "model"} else "user"
            contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})

        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        temp = getattr(config, "temperature", 0.2) if temperature is None else temperature
        tokens = getattr(config, "max_output_tokens", getattr(config, "max_tokens", 8192)) if max_tokens is None else max_tokens

        gen_config_kwargs = {
            "temperature": temp,
            "max_output_tokens": tokens,
        }
        if system_instruction is not None:
            gen_config_kwargs["system_instruction"] = system_instruction

        thinking_level = getattr(config, "thinking_level", None)
        if thinking_level and genai_types is not None:
            try:
                gen_config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_level=_thinking_level_value(thinking_level)
                )
            except Exception:
                pass

        generate_content_config = None
        if genai_types is not None:
            try:
                generate_content_config = genai_types.GenerateContentConfig(**gen_config_kwargs)
            except Exception:
                generate_content_config = None

        if generate_content_config is None:
            response = client.models.generate_content(
                model=getattr(config, "model_id", None),
                contents=contents,
                **gen_config_kwargs,
            )
        else:
            response = client.models.generate_content(
                model=getattr(config, "model_id", None),
                contents=contents,
                config=generate_content_config,
            )
        return _extract_text_gemini(response)

    # OpenAI-compatible path.
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        create = client.chat.completions.create
        response = create(
            model=getattr(config, "model_id", None),
            messages=messages,
            temperature=getattr(config, "temperature", 0.2) if temperature is None else temperature,
            max_tokens=getattr(config, "max_output_tokens", getattr(config, "max_tokens", 1024)) if max_tokens is None else max_tokens,
            stream=False,
        )
        return _extract_text_openai(response)

    # Legacy converse path.
    if hasattr(client, "converse"):
        try:
            from .llm_client import _complete_once as _shared  # noqa: F401
        except Exception:
            pass
        response = client.converse(
            modelId=getattr(config, "model_id", None),
            messages=[{"role": "user", "content": [{"text": messages[-1].get("content", "") if messages else ""}]}],
            system=[{"text": messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""}],
            inferenceConfig={
                "maxTokens": getattr(config, "max_output_tokens", getattr(config, "max_tokens", 1024)),
                "temperature": getattr(config, "temperature", 0.2),
            },
        )
        try:
            content = response["output"]["message"]["content"][0]["text"]
            if not isinstance(content, str) or not content.strip():
                raise LLMClientError("Empty model response.")
            return content.strip()
        except (KeyError, TypeError, IndexError) as exc:
            raise LLMClientError("Malformed LLM response.") from exc

    raise LLMClientError("LLM client does not support chat completions.")


def _extract_text_openai(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise LLMClientError("Malformed LLM response: missing choices.")
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, str):
        raise LLMClientError("Malformed LLM response: missing text.")
    result = content.strip()
    if not result:
        raise LLMClientError("Empty model response.")
    return result


def count_tokens(config: Any, text: str, client: Any = None) -> Optional[int]:
    """Count tokens using the official Google GenAI SDK if supported.

    Returns the exact integer token count, or None if token counting is not
    supported by the client (e.g. non-Gemini client, mock client, or offline).
    """
    if not text or not text.strip():
        return 0
    try:
        llm_client = create_client(config, client)
        if hasattr(llm_client, "models") and hasattr(llm_client.models, "count_tokens"):
            model = getattr(config, "model_id", None) or "gemini-3.8-flash"
            res = llm_client.models.count_tokens(model=model, contents=text)
            total = getattr(res, "total_tokens", None)
            if total is not None:
                return int(total)
    except Exception:
        pass
    return None


def chat_completion(
    config: Any,
    messages: Sequence[dict],
    client: Any = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    retries: int = 2,
    retry_callback: Optional[Callable[[int, int, float, Exception], None]] = None,
) -> str:
    """Send a completion request and return the response text.

    Retries transient errors (503 UNAVAILABLE, 504 DEADLINE_EXCEEDED, 429 rate limits,
    temporary network disconnects) using bounded exponential backoff with jitter and
    respecting retry-after information when available. Bounded to at most three attempts.
    Permanent errors (401, 403, 404, bad requests) fail immediately without retry.
    """
    if not getattr(config, "model_id", None):
        raise LLMClientConfigError(
            "GEMINI_MODEL_ID is not configured. Set it in .env "
            "(default: gemini-3.8-flash)."
        )
    llm_client = create_client(config, client)
    logger = logging.getLogger("educational_summarizer")
    last_error: Optional[LLMClientError] = None
    message_list = [dict(message) for message in messages]

    # Strictly enforce maximum 3 attempts (retries <= 2)
    max_attempts = min(3, max(1, retries + 1))

    for attempt in range(max_attempts):
        try:
            return _complete_once(llm_client, config, message_list, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            mapped = _map_exception(exc) if not isinstance(exc, LLMClientError) else exc
            last_error = mapped
            is_last = (attempt >= max_attempts - 1)
            if not _is_transient(exc) or is_last:
                raise mapped

            retry_after = _extract_retry_after(exc)
            backoff = _compute_backoff(attempt, base_delay=0.5, max_delay=8.0, retry_after=retry_after)
            if retry_callback:
                try:
                    retry_callback(attempt + 1, max_attempts, backoff, mapped)
                except Exception:
                    pass
            logger.warning(
                "Transient LLM error on attempt %d/%d: %s. Retrying in %.2fs",
                attempt + 1,
                max_attempts,
                mapped,
                backoff,
            )
            time.sleep(backoff)

    raise last_error or LLMClientError("LLM request failed.")


def list_models(config: Any, client: Any = None) -> List[str]:
    """Discover models through the Gemini SDK."""
    llm_client = create_client(config, client)
    try:
        pager = llm_client.models.list()
    except Exception as exc:
        raise _map_exception(exc)

    models: List[str] = []
    for item in pager or []:
        name = getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("name")
        if name is not None:
            models.append(str(name))
    return sorted(set(models))


def test_connection(config: Any, client: Any = None) -> Tuple[bool, str]:
    """Send a tiny request to verify that the selected model works."""
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with exactly: OK"},
    ]
    try:
        result = chat_completion(config, messages, client=client, max_tokens=8, temperature=0.0, retries=0)
    except LLMClientError as exc:
        return False, str(exc)
    if not result:
        return False, "Empty model response."
    return True, f"Connected. Model responded: {result[:80]}"


def test_endpoint(config: Any, client: Any = None) -> Tuple[bool, str]:
    return test_connection(config, client=client)


class LLMClient:
    """Small wrapper around configuration and the Gemini client."""

    def __init__(self, config: Any, client: Any = None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self) -> Any:
        return create_client(self.config, self._client)

    def complete(self, messages: Sequence[dict], **kwargs: Any) -> str:
        return chat_completion(self.config, messages, client=self._client, **kwargs)

    def list_models(self) -> List[str]:
        return list_models(self.config, client=self._client)

    def test_connection(self) -> Tuple[bool, str]:
        return test_connection(self.config, client=self._client)


# Backward-compatible aliases.
LocalLLMClient = LLMClient
OpenAICompatibleClient = LLMClient
GroqLLMClient = LLMClient
LLM = LLMClient
GeminiClient = LLMClient