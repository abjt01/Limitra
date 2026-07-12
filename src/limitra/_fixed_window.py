"""Fixed Window Counter rate limiting algorithm."""

from __future__ import annotations

from limitra._base import RateLimiter, RateLimitResult


class FixedWindow(RateLimiter):
    """Fixed Window Counter rate limiter.

    Requests are counted within discrete time windows of duration
    ``window`` seconds.  Once the count reaches ``limit``, further
    requests are denied until the window rolls over.

    This is the simplest rate limiting strategy and works well when
    hard per-window caps are required.

    Args:
        limit: Maximum number of requests per window. Must be at least 1.
        window: Window duration in seconds. Must be positive.

    Raises:
        ValueError: If ``limit`` or ``window`` is out of range.

    Example:
        >>> limiter = FixedWindow(limit=100, window=60.0)
        >>> result = limiter.allow()
        >>> result.allowed
        True
    """

    __slots__ = ("_counter", "_limit", "_window", "_window_start")

    def __init__(self, limit: int, window: float) -> None:
        """Initialise the fixed window counter.

        Args:
            limit: Maximum requests per window. Must be >= 1.
            window: Window duration in seconds. Must be > 0.

        Raises:
            ValueError: If ``limit`` or ``window`` is out of range.
        """
        super().__init__()
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        self._limit: int = limit
        self._window: float = window
        self._counter: int = 0
        self._window_start: float = self._now()

    # -- internal helpers ------------------------------------------------- #

    def _advance_window(self) -> float:
        """Advance the window if the current one has expired.

        If the current monotonic time is past ``_window_start + _window``,
        the counter is reset and ``_window_start`` is moved forward.

        Returns:
            The current monotonic time.
        """
        now = self._now()
        if now >= self._window_start + self._window:
            self._counter = 0
            self._window_start = now
        return now

    # -- public API ------------------------------------------------------- #

    def allow(self, cost: int = 1) -> RateLimitResult:
        """Attempt to record ``cost`` requests in the current window.

        Args:
            cost: Number of requests to record. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` with the outcome.

        Raises:
            ValueError: If ``cost`` is less than 1.
            TypeError: If ``cost`` is not an integer.
        """
        self._validate_cost(cost)
        with self._lock:
            now = self._advance_window()
            if self._counter + cost <= self._limit:
                self._counter += cost
                reset = max(0.0, (self._window_start + self._window) - now)
                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, self._limit - self._counter),
                    limit=self._limit,
                    reset_after=reset,
                    retry_after=0.0,
                )
            reset = max(0.0, (self._window_start + self._window) - now)
            return RateLimitResult(
                allowed=False,
                remaining=max(0, self._limit - self._counter),
                limit=self._limit,
                reset_after=reset,
                retry_after=reset,
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Check whether ``cost`` requests would fit without recording them.

        This method does **not** acquire the lock — it is called from
        :meth:`~RateLimiter.peek` which already holds it.

        Args:
            cost: Number of requests to check. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.

        Raises:
            ValueError: If ``cost`` is less than 1.
            TypeError: If ``cost`` is not an integer.
        """
        self._validate_cost(cost)
        now = self._advance_window()
        if self._counter + cost <= self._limit:
            remaining_after = self._limit - (self._counter + cost)
            reset = max(0.0, (self._window_start + self._window) - now)
            return RateLimitResult(
                allowed=True,
                remaining=max(0, remaining_after),
                limit=self._limit,
                reset_after=reset,
                retry_after=0.0,
            )
        reset = max(0.0, (self._window_start + self._window) - now)
        return RateLimitResult(
            allowed=False,
            remaining=max(0, self._limit - self._counter),
            limit=self._limit,
            reset_after=reset,
            retry_after=reset,
        )

    def remaining(self) -> int:
        """Return the number of requests still allowed in this window.

        Returns:
            Remaining requests computed as ``limit - counter``.
        """
        with self._lock:
            self._advance_window()
            return max(0, self._limit - self._counter)

    def reset_after(self) -> float:
        """Return seconds until the current window expires.

        Returns:
            Seconds until the window rolls over. Returns ``0.0`` if the
            window has just started (effectively full capacity).
        """
        with self._lock:
            now = self._advance_window()
            return max(0.0, (self._window_start + self._window) - now)

    def reset(self) -> None:
        """Reset the counter and start a fresh window.

        After calling this method, the limiter behaves as if freshly
        constructed.
        """
        with self._lock:
            self._counter = 0
            self._window_start = self._now()

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return (
            f"FixedWindow(limit={self._limit}, window={self._window})"
        )
