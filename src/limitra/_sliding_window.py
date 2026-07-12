"""Sliding window counter rate limiter."""

from __future__ import annotations

import math

from limitra._base import RateLimiter, RateLimitResult


class SlidingWindow(RateLimiter):
    """Rate limiter using the sliding window counter algorithm.

    Combines the current and previous fixed windows using a weighted
    overlap to approximate a true sliding window.  This gives more
    accurate rate limiting than a simple fixed window while keeping
    memory usage constant (only two counters are stored).

    Args:
        limit: Maximum number of requests allowed per window.
            Must be at least 1.
        window: Window duration in seconds.  Must be positive.

    Raises:
        ValueError: If ``limit`` or ``window`` is out of range.

    Example:
        >>> limiter = SlidingWindow(limit=100, window=60.0)
        >>> result = limiter.allow()
        >>> result.allowed
        True
    """

    __slots__ = ("_curr_counter", "_limit", "_prev_counter", "_window", "_window_start")

    def __init__(self, limit: int, window: float) -> None:
        super().__init__()
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        self._limit = limit
        self._window = float(window)
        self._window_start = self._now()
        self._curr_counter = 0
        self._prev_counter = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance(self, now: float) -> None:
        """Rotate window counters if the current window has elapsed.

        Args:
            now: Current monotonic timestamp.
        """
        if now >= self._window_start + self._window:
            elapsed = now - self._window_start
            windows_passed = int(elapsed / self._window)
            if windows_passed >= 2:
                self._prev_counter = 0
            else:
                self._prev_counter = self._curr_counter
            self._curr_counter = 0
            self._window_start += windows_passed * self._window

    def _weighted_count(self, now: float) -> float:
        """Compute the weighted request count across the sliding window.

        Args:
            now: Current monotonic timestamp.

        Returns:
            Weighted count combining the previous and current windows.
        """
        overlap_ratio = max(0.0, 1.0 - (now - self._window_start) / self._window)
        return self._prev_counter * overlap_ratio + self._curr_counter

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
            self._advance(now)
            weighted = self._weighted_count(now)

            reset_after = max(0.0, (self._window_start + self._window) - now)

            if weighted + cost <= self._limit:
                self._curr_counter += cost
                weighted = self._weighted_count(now)
                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, self._limit - math.ceil(weighted)),
                    limit=self._limit,
                    reset_after=reset_after,
                    retry_after=0.0,
                )

            return RateLimitResult(
                allowed=False,
                remaining=max(0, self._limit - math.ceil(weighted)),
                limit=self._limit,
                reset_after=reset_after,
                retry_after=reset_after,
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Check whether a request would be allowed without side effects.

        Must be called while ``self._lock`` is already held.

        Args:
            cost: Number of units to check.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.
        """
        self._validate_cost(cost)
        now = self._now()
        self._advance(now)
        weighted = self._weighted_count(now)

        reset_after = max(0.0, (self._window_start + self._window) - now)
        allowed = weighted + cost <= self._limit

        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, self._limit - math.ceil(weighted)),
            limit=self._limit,
            reset_after=reset_after,
            retry_after=0.0 if allowed else reset_after,
        )

    def remaining(self) -> int:
        """Return the number of remaining requests allowed right now.

        Returns:
            Number of requests that would currently be allowed.
        """
        with self._lock:
            now = self._now()
            self._advance(now)
            weighted = self._weighted_count(now)
            return max(0, self._limit - math.ceil(weighted))

    def reset_after(self) -> float:
        """Return seconds until the current window fully resets.

        Returns:
            Seconds until full capacity is restored.  Returns ``0.0``
            if the limiter is already at full capacity.
        """
        with self._lock:
            now = self._now()
            self._advance(now)
            return max(0.0, (self._window_start + self._window) - now)

    def reset(self) -> None:
        """Reset the limiter to its initial state."""
        with self._lock:
            self._curr_counter = 0
            self._prev_counter = 0
            self._window_start = self._now()

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return (
            f"SlidingWindow(limit={self._limit}, window={self._window})"
        )
