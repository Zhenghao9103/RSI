import json
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from .config import get_settings


class ContextWindowExceededError(RuntimeError):
    pass


def is_context_window_error(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "") or "").lower()
    if code == "context_length_exceeded":
        return True
    if code in {"rate_limit_exceeded", "rate_limit_error", "insufficient_quota"}:
        return False
    if getattr(exc, "status_code", None) == 429:
        return False

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "too many tokens",
        )
    )


class ModelError(RuntimeError):
    """Base error for model configuration and provider failures."""


class ModelConfigurationError(ModelError):
    """Raised when the Agent model is not configured."""


class ModelClientError(ModelError):
    """Raised when the Agent provider returns an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_error",
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.attempts = attempts


@lru_cache(maxsize=1)
def _agent_client() -> OpenAI:
    """Build the independent answer/agent transport client.

    The cache is deliberately limited to the transport object. Responses are
    never cached because they are request-scoped and may contain private paper
    evidence.
    """

    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("AGENT_API_KEY", settings.agent_api_key),
            ("AGENT_BASE_URL", settings.agent_base_url),
            ("AGENT_MODEL", settings.agent_model),
        )
        if not value
    ]
    if missing:
        raise ModelConfigurationError(
            "Agent model is not configured: " + ", ".join(missing) + "."
        )

    try:
        return OpenAI(
            api_key=settings.agent_api_key,
            base_url=settings.agent_base_url,
            timeout=settings.agent_timeout_seconds,
            max_retries=0,
        )
    except Exception as exc:
        raise ModelClientError(_safe_model_error(exc)) from exc


def agent_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    allow_json_fence: bool = False,
) -> dict[str, Any]:
    """Call the Agent model for a required, bounded JSON object.

    Unlike the best-effort answer helper, structured control-plane calls fail
    explicitly so the orchestrator can distinguish provider and schema errors.
    """

    settings = get_settings()
    request: dict[str, Any] = {
        "model": settings.agent_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = None
    for attempt in range(1, 3):
        try:
            response = _agent_client().chat.completions.create(**request)
            break
        except ContextWindowExceededError:
            raise
        except ModelError:
            raise
        except Exception as exc:
            if is_context_window_error(exc):
                raise ContextWindowExceededError(_safe_model_error(exc)) from exc
            category = _provider_error_category(exc)
            if attempt == 1 and category in {
                "connection_error",
                "rate_limit",
                "server_error",
                "timeout",
            }:
                continue
            raise ModelClientError(
                _safe_model_error(exc),
                category=category,
                attempts=attempt,
            ) from exc

    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) or ""
    if not isinstance(content, str) or not content.strip():
        raise ModelClientError(
            "Agent returned an empty response.",
            category="empty_response",
        )
    if allow_json_fence:
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\n(.*?)\n\s*```", content.strip(), re.I | re.S
        )
        if fenced:
            content = fenced.group(1)
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelClientError(
            "Agent returned invalid JSON.",
            category="invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise ModelClientError(
            "Agent response must be a JSON object.",
            category="invalid_json",
        )
    return payload


def complex_text_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str | None:
    """Run a plain completion on the independent complex model."""

    return _text_completion(
        _agent_client,
        get_settings().resolved_agent_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _text_completion(
    client_factory: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
) -> str | None:
    if not model:
        return None
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    try:
        response = client_factory().chat.completions.create(**request)
    except ContextWindowExceededError:
        raise
    except ModelError:
        return None
    except Exception as exc:
        if is_context_window_error(exc):
            raise ContextWindowExceededError(_safe_model_error(exc)) from exc
        return None
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) if message else None
    return content.strip() if isinstance(content, str) and content.strip() else None


def _safe_model_error(exc: Exception) -> str:
    del exc
    return "Model provider request failed."


def _provider_error_category(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 504}:
        return "timeout"
    if status_code == 429:
        return "rate_limit"
    if isinstance(status_code, int) and status_code >= 500:
        return "server_error"
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "client_error"

    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "ratelimit" in name or "rate_limit" in name:
        return "rate_limit"
    if "connection" in name or "connect" in name:
        return "connection_error"
    if "server" in name or "serviceunavailable" in name:
        return "server_error"
    return "provider_error"
