"""Sliding window log rate limiter."""

from __future__ import annotations

import bisect

from limitra._base import (
    RateLimiter,
    RateLimitResult,
    _check_size,
    _check_window,
    _satisfiable,
)


class SlidingLog(RateLimiter):
    """Rate limiter using the sliding window log algorithm.

    Maintains a sorted list of exact timestamps for every accepted
    request.  This provides perfectly accurate rate limiting at the cost
    of ``O(limit)`` memory per limiter.

    Args:
        limit: Maximum number of requests allowed per window.
            Must be at least 1.
        window: Window duration in seconds.  Must be positive.

    Raises:
        TypeError: If ``limit`` is not an integer or ``window`` is not a
            number.
        ValueError: If ``limit`` or ``window`` is out of range.

    Example:
        >>> from limitra import SlidingLog
        >>> limiter = SlidingLog(limit=5, window=10.0)
        >>> limiter.allow().allowed
        True
    """

    __slots__ = ("_limit", "_log", "_window")

    def __init__(self, limit: int, window: float) -> None:
        """Initialise the sliding window log.

        Args:
            limit: Maximum requests per window. Must be >= 1.
            window: Window duration in seconds. Must be > 0.

        Raises:
            TypeError: If ``limit`` is not an integer or ``window`` is not
                a number.
            ValueError: If ``limit`` or ``window`` is out of range.
        """
        super().__init__()
        self._limit = _check_size(limit, "limit")
        self._window = _check_window(window)
        self._log: list[float] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def limit(self) -> int:
        """Maximum number of requests allowed per window."""
        return self._limit

    @property
    def window(self) -> float:
        """Window duration in seconds."""
        return self._window

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Remove timestamps that have fallen outside the current window.

        Deletes the expired prefix in place rather than rebuilding the
        list, so pruning costs nothing when nothing has expired and is
        amortised O(1) per entry otherwise.

        ``bisect_right`` makes the window half-open — an entry whose age
        has reached exactly ``window`` is dropped — which is what lets a
        caller who waits precisely ``retry_after`` get in.

        Args:
            now: Current monotonic timestamp.
        """
        idx = bisect.bisect_right(self._log, now - self._window)
        if idx:
            del self._log[:idx]

    def _reset_after(self, now: float, newest: float | None) -> float:
        """Seconds until the whole log has expired.

        Args:
            now: Current monotonic timestamp.
            newest: The most recent timestamp in the log, or ``None`` if
                the log is empty.

        Returns:
            Seconds until nothing is counted against the limiter.
        """
        if newest is None:
            return 0.0
        return max(0.0, newest + self._window - now)

    def _retry_after(self, now: float, cost: int) -> float:
        """Seconds until ``cost`` slots have freed up.

        Args:
            now: Current monotonic timestamp.
            cost: Units the caller wants to spend.

        Returns:
            Seconds to wait before retrying.
        """
        # Slots that must expire before `cost` fits; the log is sorted, so
        # the one that frees the last needed slot is at index need - 1.
        need = len(self._log) + cost - self._limit
        return _satisfiable(self._log[need - 1] + self._window - now)

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
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        with self._lock:
            now = self._now()
            self._prune(now)

            if len(self._log) + cost <= self._limit:
                self._log.extend([now] * cost)
                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, self._limit - len(self._log)),
                    limit=self._limit,
                    reset_after=self._reset_after(now, self._log[-1]),
                    retry_after=0.0,
                )

            return RateLimitResult(
                allowed=False,
                remaining=max(0, self._limit - len(self._log)),
                limit=self._limit,
                reset_after=self._reset_after(
                    now, self._log[-1] if self._log else None
                ),
                retry_after=self._retry_after(now, cost),
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Report what :meth:`allow` would return, without consuming anything.

        Must be called while ``self._lock`` is already held.  Pruning
        expired entries is safe — it only removes already-expired
        timestamps.

        Args:
            cost: Number of units to check.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        now = self._now()
        self._prune(now)

        if len(self._log) + cost <= self._limit:
            # The hypothetical entries would land at `now`, making it the
            # newest timestamp in the log.
            return RateLimitResult(
                allowed=True,
                remaining=max(0, self._limit - (len(self._log) + cost)),
                limit=self._limit,
                reset_after=self._reset_after(now, now),
                retry_after=0.0,
            )

        return RateLimitResult(
            allowed=False,
            remaining=max(0, self._limit - len(self._log)),
            limit=self._limit,
            reset_after=self._reset_after(now, self._log[-1] if self._log else None),
            retry_after=self._retry_after(now, cost),
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
        """Return seconds until the limiter is back at full capacity.

        Returns:
            Seconds until the newest logged request expires, or ``0.0``
            if the log is empty.
        """
        with self._lock:
            now = self._now()
            self._prune(now)
            return self._reset_after(now, self._log[-1] if self._log else None)

    def reset(self) -> None:
        """Reset the limiter to its initial state."""
        with self._lock:
            self._log.clear()

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return f"SlidingLog(limit={self._limit}, window={self._window})"
