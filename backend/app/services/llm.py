"""Required context-error utilities copied from Paper-Agent llm.py."""

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
