"""Google Gemini LLM client with verified cloud fallback chain and local Ollama fallback.

Tier 1 (Cloud): Primary Gemini model followed by configured verified fallback Gemini models,
with per-family generation settings (e.g. omitting temperature on Gemini 4 models).
Tier 2 (Local-only): Fast health check against localhost Ollama (/api/tags, 1.5s timeout).
When unreachable (e.g. Streamlit Community Cloud), Ollama is skipped immediately without hanging.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
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

    if isinstance(exc, LLMClientError):
        return exc
    return LLMClientError(f"Unexpected LLM error: {exc}")


def _is_auth_or_token_permanent(exc: Exception) -> bool:
    """Errors that should never trigger retries or cycling to another cloud model."""
    msg = str(exc)
    lower = msg.lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (401, 403) or "unauthenticated" in lower or "api key" in lower or "permission" in lower:
        return True
    if "token limit" in lower or "context length" in lower:
        return True
    return False


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

    transient_types: List[type] = []
    if httpx is not None:
        for name in ("TimeoutException", "ConnectError", "NetworkError"):
            cls = getattr(httpx, name, None)
            if cls is not None:
                transient_types.append(cls)

    return bool(transient_types) and isinstance(exc, tuple(transient_types))


def _should_try_next_model(exc: Exception) -> bool:
    """Return True if a failure on one Gemini model should advance to the next fallback."""
    if _is_auth_or_token_permanent(exc):
        return False
    if _is_transient(exc):
        return True
    msg = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 404 or "not found" in msg or "missing text" in msg:
        return True
    return False


# ---------------------------------------------------------------------------
# Tier 2: Local-only Ollama fallback (zero extra dependencies, Cloud-safe)
# ---------------------------------------------------------------------------

_OLLAMA_EMBED_MARKERS = ("embed", "bge-", "mxbai", "all-minilm", "snowflake-arctic")


def _ollama_url(base_url: str, path: str) -> str:
    """Build an Ollama endpoint URL, preferring IPv4 127.0.0.1 over Windows IPv6 localhost delay."""
    raw = base_url if isinstance(base_url, str) and base_url.strip() else "http://localhost:11434"
    base = raw.strip().rstrip("/")
    base = re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", base, flags=re.IGNORECASE)
    return f"{base}{path}"


def check_ollama_health(base_url: str = "http://localhost:11434", timeout: float = 1.5) -> Tuple[bool, List[str]]:
    """Fast local Ollama health check via GET {base_url}/api/tags.

    Returns (is_reachable, installed_model_names). Fails fast (<= 1.5s) on
    cloud hosts like Streamlit Community Cloud where localhost:11434 is absent.
    """
    url = _ollama_url(base_url, "/api/tags")
    try:
        if httpx is not None:
            resp = httpx.get(url, timeout=timeout)
            if resp.status_code != 200:
                return False, []
            payload = resp.json()
        else:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if getattr(resp, "status", 200) != 200:
                    return False, []
                payload = json.loads(resp.read().decode("utf-8"))
        models = payload.get("models") or []
        names = [str(m.get("name", "")).strip() for m in models if isinstance(m, dict) and m.get("name")]
        return True, names
    except Exception:
        return False, []


def select_ollama_model(
    installed_models: Sequence[str],
    preferred_model: str = "qwen2.5:1.5b-instruct",
) -> Optional[str]:
    """Select the preferred local Ollama model, else a 1.5b instruct tag, else first chat model."""
    cleaned = [str(m).strip() for m in (installed_models or []) if str(m).strip()]
    if not cleaned:
        return None

    pref_low = (preferred_model or "").strip().lower()
    if pref_low:
        for name in cleaned:
            low = name.lower()
            if low == pref_low or low == f"{pref_low}:latest":
                return name

    # Prefer any installed 1.5b instruct tag
    for name in cleaned:
        low = name.lower()
        if "1.5b" in low and "instruct" in low:
            return name
    for name in cleaned:
        if "1.5b" in name.lower():
            return name

    # Fall back to the first installed non-embedding chat model
    for name in cleaned:
        low = name.lower()
        if not any(marker in low for marker in _OLLAMA_EMBED_MARKERS):
            return name
    return None


def ollama_chat_completion(
    base_url: str,
    model: str,
    messages: Sequence[dict],
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """Send a non-streaming chat completion request to local Ollama (/api/chat)."""
    url = _ollama_url(base_url, "/api/chat")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ],
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_ctx": 4096,
        },
    }
    if "qwen3" in (model or "").lower():
        body["think"] = False

    try:
        if httpx is not None:
            resp = httpx.post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        else:
            raw = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=raw,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise LLMClientError(f"Local Ollama request failed ({model}): {exc}") from exc

    message = payload.get("message") if isinstance(payload, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LLMClientError(f"Malformed or empty local Ollama response ({model}).")

    # Strip optional reasoning tags (e.g. <think>...</think>) if a local reasoning model emits them.
    cleaned = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned or content.strip()


def _try_ollama_fallback(
    config: Any,
    messages: Sequence[dict],
    temperature: Optional[float] = None,
) -> Optional[Tuple[str, str]]:
    """Attempt Tier 2 local Ollama fallback if enabled and reachable; returns (text, label) or None."""
    if not getattr(config, "ollama_enabled", False):
        return None

    base_url = getattr(config, "ollama_base_url", "http://localhost:11434")
    preferred = getattr(config, "ollama_model", "qwen2.5:1.5b-instruct")
    reachable, installed = check_ollama_health(base_url, timeout=1.5)
    if not reachable:
        return None

    selected = select_ollama_model(installed, preferred_model=preferred)
    if not selected:
        return None

    temp = getattr(config, "temperature", 0.2) if temperature is None else temperature
    timeout = float(getattr(config, "timeout_seconds", 120))
    text = ollama_chat_completion(
        base_url=base_url,
        model=selected,
        messages=messages,
        temperature=temp,
        timeout=timeout,
    )
    return text, f"ollama/{selected} (local fallback)"


def _complete_once(
    client: Any,
    config: Any,
    messages: List[dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model_override: Optional[str] = None,
) -> str:
    target_model = model_override or getattr(config, "model_id", None)

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
        tokens = (
            getattr(config, "max_output_tokens", getattr(config, "max_tokens", 8192))
            if max_tokens is None
            else max_tokens
        )

        lower_model = str(target_model or "").lower()
        is_gemini_4 = lower_model.startswith("gemini-4")

        gen_config_kwargs: dict[str, Any] = {
            "max_output_tokens": tokens,
        }
        # Per-family rule: Gemini 4 rejects custom sampling parameters like temperature.
        if not is_gemini_4:
            gen_config_kwargs["temperature"] = temp

        if system_instruction is not None:
            gen_config_kwargs["system_instruction"] = system_instruction

        thinking_level = getattr(config, "thinking_level", None)
        if is_gemini_4:
            # Gemini 4 rejects 'minimal' thinking; enforce 'low' (or higher if configured).
            effective_level = "low" if (not thinking_level or str(thinking_level).lower() == "minimal") else thinking_level
        else:
            effective_level = thinking_level

        if effective_level and genai_types is not None:
            try:
                gen_config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_level=_thinking_level_value(effective_level)
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
                model=target_model,
                contents=contents,
                **gen_config_kwargs,
            )
        else:
            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=generate_content_config,
            )
        return _extract_text_gemini(response)

    # Duck-typed chat.completions path (for unit-test doubles).
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        create = client.chat.completions.create
        response = create(
            model=target_model,
            messages=messages,
            temperature=getattr(config, "temperature", 0.2) if temperature is None else temperature,
            max_tokens=getattr(config, "max_output_tokens", getattr(config, "max_tokens", 1024)) if max_tokens is None else max_tokens,
            stream=False,
        )
        return _extract_text_openai(response)

    # Legacy converse path.
    if hasattr(client, "converse"):
        response = client.converse(
            modelId=target_model,
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


def _build_model_chain(config: Any) -> List[str]:
    primary = getattr(config, "model_id", None)
    fallbacks = getattr(config, "fallback_models", ()) or ()
    chain: List[str] = []
    if primary:
        chain.append(str(primary).strip())
    for fb in fallbacks:
        m = str(fb).strip()
        if m and m not in chain:
            chain.append(m)
    return chain


def chat_completion_with_metadata(
    config: Any,
    messages: Sequence[dict],
    client: Any = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    retries: int = 2,
    retry_callback: Optional[Callable[[int, int, float, Exception], None]] = None,
) -> Tuple[str, str]:
    """Send a completion request across Tier 1 (Gemini chain) and Tier 2 (local Ollama).

    Returns:
        Tuple of ``(response_text, backend_model_label)``.
    """
    logger = logging.getLogger("educational_summarizer")
    message_list = [dict(message) for message in messages]
    api_key = getattr(config, "api_key", None)
    primary_model = getattr(config, "model_id", None)

    # If Gemini cloud is unconfigured, allow local-only Ollama fallback when enabled & reachable.
    if not primary_model or (client is None and not api_key):
        ollama_res = _try_ollama_fallback(config, message_list, temperature=temperature)
        if ollama_res is not None:
            logger.info("Using local Ollama fallback: %s", ollama_res[1])
            return ollama_res
        if not primary_model:
            raise LLMClientConfigError(
                "GEMINI_MODEL_ID is not configured. Set it in .env "
                "(default: gemini-3.8-flash)."
            )
        raise LLMClientConfigError("GEMINI_API_KEY is not configured.")

    llm_client = create_client(config, client)
    model_chain = _build_model_chain(config)
    has_fallbacks = len(model_chain) > 1

    # Keep UI snappy when cycling multiple cloud models (max 2 attempts per model);
    # preserve max 3 attempts when running a single model without fallbacks.
    max_attempts_per_model = (
        min(2, max(1, retries + 1)) if has_fallbacks else min(3, max(1, retries + 1))
    )
    base_delay = 0.35 if has_fallbacks else 0.5
    max_delay = 2.5 if has_fallbacks else 8.0

    last_error: Optional[LLMClientError] = None

    for model_idx, target_model in enumerate(model_chain):
        for attempt in range(max_attempts_per_model):
            try:
                text = _complete_once(
                    llm_client,
                    config,
                    message_list,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_override=target_model,
                )
                label = target_model if model_idx == 0 else f"{target_model} (cloud fallback)"
                if model_idx > 0:
                    logger.info("Gemini cloud fallback succeeded via %s", target_model)
                return text, label
            except Exception as exc:
                mapped = _map_exception(exc) if not isinstance(exc, LLMClientError) else exc
                last_error = mapped
                is_last_attempt = attempt >= max_attempts_per_model - 1

                if _is_auth_or_token_permanent(exc):
                    # Auth/token errors will fail identically on other Gemini models.
                    break

                if not _is_transient(exc):
                    break

                if not is_last_attempt:
                    retry_after = _extract_retry_after(exc)
                    backoff = _compute_backoff(
                        attempt,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        retry_after=retry_after,
                    )
                    if retry_callback:
                        try:
                            retry_callback(attempt + 1, max_attempts_per_model, backoff, mapped)
                        except Exception:
                            pass
                    logger.warning(
                        "Transient LLM error on %s (attempt %d/%d): %s. Retrying in %.2fs",
                        target_model,
                        attempt + 1,
                        max_attempts_per_model,
                        mapped,
                        backoff,
                    )
                    time.sleep(backoff)

        # Decide whether to advance to the next fallback cloud model
        if last_error is not None and not _should_try_next_model(last_error):
            break
        if model_idx < len(model_chain) - 1:
            logger.warning(
                "Gemini model '%s' failed (%s); falling back to '%s'.",
                target_model,
                last_error,
                model_chain[model_idx + 1],
            )

    # Tier 2: Local-only Ollama fallback (safe no-op if disabled or unreachable)
    try:
        ollama_res = _try_ollama_fallback(config, message_list, temperature=temperature)
        if ollama_res is not None:
            logger.info("All cloud Gemini models failed; succeeded via %s", ollama_res[1])
            return ollama_res
    except LLMClientError as ollama_exc:
        logger.warning("Local Ollama fallback also failed: %s", ollama_exc)

    raise last_error or LLMClientError("LLM request failed.")


def chat_completion(
    config: Any,
    messages: Sequence[dict],
    client: Any = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    retries: int = 2,
    retry_callback: Optional[Callable[[int, int, float, Exception], None]] = None,
) -> str:
    """Send a completion request and return the response text."""
    text, _ = chat_completion_with_metadata(
        config,
        messages,
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
        retry_callback=retry_callback,
    )
    return text


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
        result, used_model = chat_completion_with_metadata(
            config, messages, client=client, max_tokens=64, temperature=0.0, retries=0
        )
    except LLMClientError as exc:
        return False, str(exc)
    if not result:
        return False, "Empty model response."
    return True, f"Connected ({used_model}). Model responded: {result[:80]}"


def test_endpoint(config: Any, client: Any = None) -> Tuple[bool, str]:
    return test_connection(config, client=client)


class LLMClient:
    """Small wrapper around configuration and the Gemini/Ollama client."""

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
