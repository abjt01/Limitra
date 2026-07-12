"""Multi-key rate limit manager."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limitra._base import RateLimiter, RateLimitResult


class RateLimitManager:
    """Orchestrates per-key rate limiters using any :class:`RateLimiter` algorithm.

    Creates and caches a separate :class:`RateLimiter` instance for each
    unique key (e.g., user ID, IP address, API key).  Idle limiters can
    be cleaned up periodically via :meth:`cleanup`.

    This class does **not** extend :class:`RateLimiter` — it delegates to
    individual limiter instances instead.

    Args:
        algorithm: A :class:`RateLimiter` subclass to instantiate per key.
        **kwargs: Keyword arguments forwarded to ``algorithm(...)`` when
            creating new limiter instances.

    Example:
        >>> from limitra._sliding_window import SlidingWindow
        >>> mgr = RateLimitManager(SlidingWindow, limit=100, window=60.0)
        >>> result = mgr.allow("user-42")
        >>> result.allowed
        True
    """

    __slots__ = ("_algorithm", "_kwargs", "_last_access", "_limiters", "_lock")

    def __init__(self, algorithm: type[RateLimiter], **kwargs: Any) -> None:
        self._algorithm = algorithm
        self._kwargs = kwargs
        self._limiters: dict[str, RateLimiter] = {}
        self._last_access: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, key: str) -> RateLimiter:
        """Return the limiter for *key*, creating one if necessary.

        Must be called while ``self._lock`` is held.

        Args:
            key: The rate-limit key.

        Returns:
            The :class:`RateLimiter` instance for *key*.
        """
        if key not in self._limiters:
            self._limiters[key] = self._algorithm(**self._kwargs)
        return self._limiters[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self, key: str, cost: int = 1) -> RateLimitResult:
        """Consume ``cost`` units from the limiter associated with *key*.

        Args:
            key: The rate-limit key (e.g., user ID, IP address).
            cost: Number of units to consume.  Defaults to 1.

        Returns:
            A :class:`RateLimitResult` with the decision and metadata.
        """
        with self._lock:
            limiter = self._get_or_create(key)
            self._last_access[key] = time.monotonic()
        return limiter.allow(cost)

    def peek(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check whether a request for *key* would be allowed, without consuming.

        Args:
            key: The rate-limit key.
            cost: Number of units to check.  Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.
        """
        with self._lock:
            limiter = self._get_or_create(key)
        return limiter.peek(cost)

    def get(self, key: str) -> RateLimiter | None:
        """Return the limiter for *key*, or ``None`` if not tracked.

        Args:
            key: The rate-limit key.

        Returns:
            The :class:`RateLimiter` instance, or ``None``.
        """
        with self._lock:
            return self._limiters.get(key)

    def remove(self, key: str) -> bool:
        """Remove the limiter for *key*.

        Args:
            key: The rate-limit key to remove.

        Returns:
            ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        with self._lock:
            existed = key in self._limiters
            self._limiters.pop(key, None)
            self._last_access.pop(key, None)
            return existed

    def cleanup(self, max_idle: float = 300.0) -> int:
        """Remove limiters that have been idle for more than *max_idle* seconds.

        Args:
            max_idle: Maximum idle time in seconds before a limiter is
                removed.  Defaults to 300 (5 minutes).

        Returns:
            Number of limiters removed.
        """
        now = time.monotonic()
        with self._lock:
            stale_keys = [
                key
                for key, last in self._last_access.items()
                if now - last > max_idle
            ]
            for key in stale_keys:
                del self._limiters[key]
                del self._last_access[key]
            return len(stale_keys)

    def keys(self) -> list[str]:
        """Return a list of all currently tracked keys.

        Returns:
            List of rate-limit keys.
        """
        with self._lock:
            return list(self._limiters.keys())

    def __len__(self) -> int:
        """Return the number of tracked keys."""
        with self._lock:
            return len(self._limiters)

    def __contains__(self, key: object) -> bool:
        """Check whether *key* is currently tracked."""
        with self._lock:
            return key in self._limiters

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return (
            f"RateLimitManager("
            f"algorithm={self._algorithm.__name__}, "
            f"keys={len(self._limiters)})"
        )
