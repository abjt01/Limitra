"""Sliding window log rate limiter."""

from __future__ import annotations

import bisect

from limitra._base import RateLimiter, RateLimitResult


class SlidingLog(RateLimiter):
    """Rate limiter using the sliding window log algorithm.

    Maintains a sorted list of exact timestamps for every accepted
    request.  This provides perfectly accurate rate limiting at the cost
    of ``O(n)`` memory where *n* is the window limit.

    Args:
        limit: Maximum number of requests allowed per window.
            Must be at least 1.
        window: Window duration in seconds.  Must be positive.

    Raises:
        ValueError: If ``limit`` or ``window`` is out of range.

    Example:
        >>> limiter = SlidingLog(limit=5, window=10.0)
        >>> result = limiter.allow()
        >>> result.allowed
        True
    """

    __slots__ = ("_limit", "_log", "_window")

    def __init__(self, limit: int, window: float) -> None:
        super().__init__()
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        self._limit = limit
        self._window = float(window)
        self._log: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Remove timestamps that have fallen outside the current window.

        Args:
            now: Current monotonic timestamp.
        """
        cutoff = now - self._window
        idx = bisect.bisect_left(self._log, cutoff)
        self._log = self._log[idx:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self, cost: int = 1) -> RateLimitResult:
        """Attempt to consume ``cost`` units from the rate limiter.

        Args:
            cost: Number of units to consume.  Defaults to 1.

        Returns:
            A :class:`RateLimitResult` with the decision and metadata.

        Raises:
            ValueError: If ``cost`` is less than 1.
            TypeError: If ``cost`` is not an integer.
        """
        self._validate_cost(cost)
        with self._lock:
            now = self._now()
            self._prune(now)

            reset_after = (
                (self._log[0] + self._window - now) if self._log else 0.0
            )
            reset_after = max(0.0, reset_after)

            if len(self._log) + cost <= self._limit:
                self._log.extend([now] * cost)
                remaining = max(0, self._limit - len(self._log))
                # Recompute reset_after after insertion
                reset_after = (
                    max(0.0, self._log[0] + self._window - now)
                    if self._log
                    else 0.0
                )
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    limit=self._limit,
                    reset_after=reset_after,
                    retry_after=0.0,
                )

            # Denied — compute retry_after
            need = len(self._log) + cost - self._limit
            retry_after = max(0.0, self._log[need - 1] + self._window - now)

            return RateLimitResult(
                allowed=False,
                remaining=max(0, self._limit - len(self._log)),
                limit=self._limit,
                reset_after=reset_after,
                retry_after=retry_after,
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Check whether a request would be allowed without side effects.

        Must be called while ``self._lock`` is already held.  Pruning
        expired entries is safe — it only removes already-expired
        timestamps.

        Args:
            cost: Number of units to check.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.
        """
        self._validate_cost(cost)
        now = self._now()
        self._prune(now)

        reset_after = (
            max(0.0, self._log[0] + self._window - now) if self._log else 0.0
        )
        allowed = len(self._log) + cost <= self._limit

        if allowed:
            retry_after = 0.0
        else:
            need = len(self._log) + cost - self._limit
            retry_after = max(0.0, self._log[need - 1] + self._window - now)

        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, self._limit - len(self._log)),
            limit=self._limit,
            reset_after=reset_after,
            retry_after=retry_after,
        )

    def remaining(self) -> int:
        """Return the number of remaining requests allowed right now.

        Returns:
            Number of requests that would currently be allowed.
        """
        with self._lock:
            now = self._now()
            self._prune(now)
            return max(0, self._limit - len(self._log))

    def reset_after(self) -> float:
        """Return seconds until the oldest entry expires from the window.

        Returns:
            Seconds until full capacity is restored.  Returns ``0.0``
            if the log is empty.
        """
        with self._lock:
            now = self._now()
            self._prune(now)
            if self._log:
                return max(0.0, self._log[0] + self._window - now)
            return 0.0

    def reset(self) -> None:
        """Reset the limiter to its initial state."""
        with self._lock:
            self._log = []

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return f"SlidingLog(limit={self._limit}, window={self._window})"
