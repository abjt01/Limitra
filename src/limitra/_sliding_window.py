"""Sliding window counter rate limiter."""

from __future__ import annotations

import math

from limitra._base import (
    RateLimiter,
    RateLimitResult,
    _check_size,
    _check_window,
    _satisfiable,
)


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
        TypeError: If ``limit`` is not an integer or ``window`` is not a
            number.
        ValueError: If ``limit`` or ``window`` is out of range.

    Example:
        >>> from limitra import SlidingWindow
        >>> limiter = SlidingWindow(limit=100, window=60.0)
        >>> limiter.allow().allowed
        True
    """

    __slots__ = ("_curr_counter", "_limit", "_prev_counter", "_window", "_window_start")

    def __init__(self, limit: int, window: float) -> None:
        """Initialise the sliding window counter.

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
        self._window_start = self._now()
        self._curr_counter = 0
        self._prev_counter = 0

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

    def _advance(self, now: float) -> None:
        """Rotate window counters if the current window has elapsed.

        Args:
            now: Current monotonic timestamp.
        """
        if now >= self._window_start + self._window:
            elapsed = now - self._window_start
            # At least one window has passed by the guard above, but float
            # division can still floor to 0 when _window_start is large and
            # the window is short. Stepping by 0 would clear the previous
            # counter without moving the boundary, freeing the whole limit.
            windows_passed = max(1, int(elapsed / self._window))
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

    def _reset_after(self, now: float, prev: int, curr: int) -> float:
        """Seconds until nothing is counted against the limiter any more.

        The current window's requests only stop counting a full window
        after they become the *previous* window and decay away, so a fresh
        request holds capacity for up to two windows.

        Args:
            now: Current monotonic timestamp.
            prev: Previous-window counter to report against.
            curr: Current-window counter to report against.

        Returns:
            Seconds until the weighted count reaches zero.
        """
        if curr > 0:
            return max(0.0, (self._window_start + 2 * self._window) - now)
        if prev > 0:
            return max(0.0, (self._window_start + self._window) - now)
        return 0.0

    def _retry_after(self, now: float, cost: int) -> float:
        """Seconds until ``cost`` would actually fit.

        The weighted count decays continuously, so the answer is rarely the
        next window boundary: solving for the moment the previous window has
        decayed far enough is what makes a client that obeys ``retry_after``
        succeed on its first retry instead of being denied again.

        Args:
            now: Current monotonic timestamp.
            cost: Units the caller wants to spend.

        Returns:
            Seconds to wait before retrying.
        """
        # Room left in this window once the current counter is accounted for.
        target = self._limit - cost - self._curr_counter
        if target > 0 and self._prev_counter > target:
            # The previous window decays to `target` partway through this
            # one. Its counter is necessarily above `target`, or the request
            # would not have been denied; testing it also keeps the division
            # safe if that ever stops holding.
            deadline = self._window_start + self._window * (
                1.0 - target / self._prev_counter
            )
            return _satisfiable(deadline - now)

        # Not admissible before the boundary: the current counter alone is
        # already too large. After the rotation it becomes the previous one
        # and decays in turn, so solve again against that.
        boundary = self._window_start + self._window
        target = self._limit - cost
        if self._curr_counter <= target:
            return _satisfiable(boundary - now)
        deadline = boundary + self._window * (1.0 - target / self._curr_counter)
        return _satisfiable(deadline - now)

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
            self._advance(now)
            weighted = self._weighted_count(now)

            if weighted + cost <= self._limit:
                self._curr_counter += cost
                weighted = self._weighted_count(now)
                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, self._limit - math.ceil(weighted)),
                    limit=self._limit,
                    reset_after=self._reset_after(
                        now, self._prev_counter, self._curr_counter
                    ),
                    retry_after=0.0,
                )

            return RateLimitResult(
                allowed=False,
                remaining=max(0, self._limit - math.ceil(weighted)),
                limit=self._limit,
                reset_after=self._reset_after(
                    now, self._prev_counter, self._curr_counter
                ),
                retry_after=self._retry_after(now, cost),
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Report what :meth:`allow` would return, without consuming anything.

        Must be called while ``self._lock`` is already held.

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
        self._advance(now)
        weighted = self._weighted_count(now)

        if weighted + cost <= self._limit:
            return RateLimitResult(
                allowed=True,
                remaining=max(0, self._limit - math.ceil(weighted + cost)),
                limit=self._limit,
                reset_after=self._reset_after(
                    now, self._prev_counter, self._curr_counter + cost
                ),
                retry_after=0.0,
            )

        return RateLimitResult(
            allowed=False,
            remaining=max(0, self._limit - math.ceil(weighted)),
            limit=self._limit,
            reset_after=self._reset_after(now, self._prev_counter, self._curr_counter),
            retry_after=self._retry_after(now, cost),
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
        """Return seconds until the limiter is back at full capacity.

        Returns:
            Seconds until the weighted count reaches zero, or ``0.0`` if
            nothing is counted against the limiter.
        """
        with self._lock:
            now = self._now()
            self._advance(now)
            return self._reset_after(now, self._prev_counter, self._curr_counter)

    def reset(self) -> None:
        """Reset the limiter to its initial state."""
        with self._lock:
            self._curr_counter = 0
            self._prev_counter = 0
            self._window_start = self._now()

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return f"SlidingWindow(limit={self._limit}, window={self._window})"
