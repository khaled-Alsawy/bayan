from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_quota_fallback_used: ContextVar[bool] = ContextVar("local_quota_fallback_used", default=False)


def local_quota_fallback_used() -> bool:
    """Return request-scoped attribution for the current local completion."""

    return _quota_fallback_used.get()


@contextmanager
def local_completion_attribution(*, quota_fallback: bool) -> Iterator[None]:
    """Scope local completion attribution without mutating shared client state."""

    token = _quota_fallback_used.set(quota_fallback)
    try:
        yield
    finally:
        _quota_fallback_used.reset(token)
