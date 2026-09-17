"""Fixed Window Counter rate limiting algorithm."""

from __future__ import annotations

from limitra._base import RateLimiter, RateLimitResult, _check_size, _check_window


class FixedWindow(RateLimiter):
    """Fixed Window Counter rate limiter.

    Requests are counted within discrete time windows of duration
    ``window`` seconds.  Once the count reaches ``limit``, further
    requests are denied until the window rolls over.

    This is the simplest rate limiting strategy and works well when
    hard per-window caps are required.

    Boundary burst:
        Because the counter clears all at once, up to ``2 * limit``
        requests can land inside a single window-length interval that
        straddles a boundary — ``limit`` at the end of one window and
        ``limit`` at the start of the next.  Use :class:`SlidingWindow`
        or :class:`SlidingLog` when that matters.

    Args:
        limit: Maximum number of requests per window. Must be at least 1.
        window: Window duration in seconds. Must be positive.

    Raises:
        TypeError: If ``limit`` is not an integer or ``window`` is not a
            number.
        ValueError: If ``limit`` or ``window`` is out of range.

    Example:
        >>> from limitra import FixedWindow
        >>> limiter = FixedWindow(limit=100, window=60.0)
        >>> limiter.allow().allowed
        True
    """

    __slots__ = ("_counter", "_limit", "_window", "_window_start")

    def __init__(self, limit: int, window: float) -> None:
        """Initialise the fixed window counter.

        Args:
            limit: Maximum requests per window. Must be >= 1.
            window: Window duration in seconds. Must be > 0.

        Raises:
            TypeError: If ``limit`` is not an integer or ``window`` is not
                a number.
            ValueError: If ``limit`` or ``window`` is out of range.
        """
        super().__init__()
        self._limit: int = _check_size(limit, "limit")
        self._window: float = _check_window(window)
        self._counter: int = 0
        self._window_start: float = self._now()

    # -- configuration ---------------------------------------------------- #

    @property
    def limit(self) -> int:
        """Maximum number of requests allowed per window."""
        return self._limit

    @property
    def window(self) -> float:
        """Window duration in seconds."""
        return self._window

    # -- internal helpers ------------------------------------------------- #

    def _advance_window(self) -> float:
        """Advance the window if the current one has expired.

        Window boundaries step forward in whole ``window`` multiples from
        the limiter's construction time, so they depend only on elapsed
        time.  Re-anchoring to ``now`` instead would let a bystander call —
        ``remaining()`` from a metrics scrape, say — shift the boundary and
        change the ``Retry-After`` other callers are given.

        Returns:
            The current monotonic time.
        """
        now = self._now()
        elapsed = now - self._window_start
        if elapsed >= self._window:
            self._counter = 0
            self._window_start += (elapsed // self._window) * self._window
        return now

    def _reset_after(self, now: float, counter: int) -> float:
        """Seconds until the window rolls over, given a counter value.

        Args:
            now: Current monotonic time.
            counter: The counter to report against.

        Returns:
            ``0.0`` when nothing is counted against the window, otherwise
            the time until it clears.
        """
        if counter == 0:
            return 0.0
        return max(0.0, (self._window_start + self._window) - now)

    # -- public API ------------------------------------------------------- #

    def allow(self, cost: int = 1) -> RateLimitResult:
        """Attempt to record ``cost`` requests in the current window.

        Args:
            cost: Number of requests to record. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` with the outcome.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        with self._lock:
            now = self._advance_window()
            if self._counter + cost <= self._limit:
                self._counter += cost
                reset = self._reset_after(now, self._counter)
                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, self._limit - self._counter),
                    limit=self._limit,
                    reset_after=reset,
                    retry_after=0.0,
                )
            reset = self._reset_after(now, self._counter)
            return RateLimitResult(
                allowed=False,
                remaining=max(0, self._limit - self._counter),
                limit=self._limit,
                reset_after=reset,
                retry_after=reset,
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Report what :meth:`allow` would return, without recording anything.

        This method does **not** acquire the lock — it is called from
        :meth:`~RateLimiter.peek` which already holds it.

        Args:
            cost: Number of requests to check. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        now = self._advance_window()
        if self._counter + cost <= self._limit:
            counter_after = self._counter + cost
            reset = self._reset_after(now, counter_after)
            return RateLimitResult(
                allowed=True,
                remaining=max(0, self._limit - counter_after),
                limit=self._limit,
                reset_after=reset,
                retry_after=0.0,
            )
        reset = self._reset_after(now, self._counter)
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
        """Return seconds until the window clears.

        Returns:
            Seconds until the counter rolls over, or ``0.0`` if nothing is
            counted against the current window.
        """
        with self._lock:
            now = self._advance_window()
            return self._reset_after(now, self._counter)

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
        return f"FixedWindow(limit={self._limit}, window={self._window})"
