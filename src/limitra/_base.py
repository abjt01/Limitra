"""Abstract base class and shared types for rate limiting algorithms."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod


class RateLimitResult:
    """Result of a rate limit check.

    Provides complete context about the rate limit decision, suitable for
    populating HTTP headers like ``X-RateLimit-Remaining`` and ``Retry-After``.

    Attributes:
        allowed: Whether the request was permitted.
        remaining: Number of requests remaining in the current window.
        limit: Maximum number of requests allowed.
        reset_after: Seconds until the limiter fully resets or refills.
        retry_after: Seconds until a denied request could succeed. Always
            ``0.0`` when ``allowed`` is ``True``.
    """

    __slots__ = ("allowed", "limit", "remaining", "reset_after", "retry_after")

    def __init__(
        self,
        *,
        allowed: bool,
        remaining: int,
        limit: int,
        reset_after: float,
        retry_after: float,
    ) -> None:
        self.allowed = allowed
        self.remaining = remaining
        self.limit = limit
        self.reset_after = reset_after
        self.retry_after = retry_after

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return (
            f"RateLimitResult("
            f"allowed={self.allowed}, "
            f"remaining={self.remaining}, "
            f"limit={self.limit}, "
            f"reset_after={self.reset_after:.4f}, "
            f"retry_after={self.retry_after:.4f})"
        )

    def __bool__(self) -> bool:
        """Allow truthiness check.

        ``if result:`` is equivalent to ``if result.allowed:``.
        """
        return self.allowed

    def __eq__(self, other: object) -> bool:
        """Check equality based on all fields."""
        if not isinstance(other, RateLimitResult):
            return NotImplemented
        return (
            self.allowed == other.allowed
            and self.remaining == other.remaining
            and self.limit == other.limit
        )


class RateLimiter(ABC):
    """Abstract base class for all rate limiting algorithms.

    Every rate limiter implementation must subclass ``RateLimiter`` and
    implement the three abstract methods: :meth:`allow`, :meth:`remaining`,
    and :meth:`reset_after`.

    All implementations guarantee:
        - Thread safety via ``threading.Lock``
        - Clock-drift immunity via ``time.monotonic()``
        - Consistent API regardless of underlying algorithm

    Example:
        >>> limiter = TokenBucket(rate=10.0, capacity=100)
        >>> result = limiter.allow()
        >>> if result.allowed:
        ...     process_request()
    """

    __slots__ = ("_lock",)

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @abstractmethod
    def allow(self, cost: int = 1) -> RateLimitResult:
        """Attempt to consume ``cost`` units from the rate limiter.

        Args:
            cost: Number of units to consume. Defaults to 1. Must be a
                positive integer.

        Returns:
            A :class:`RateLimitResult` indicating whether the request was
            allowed and providing metadata about the current limiter state.

        Raises:
            ValueError: If ``cost`` is less than 1.
        """

    @abstractmethod
    def remaining(self) -> int:
        """Return the number of remaining requests allowed.

        This is a snapshot — the value may change immediately after reading
        in a multi-threaded environment.

        Returns:
            Number of requests that would be allowed right now.
        """

    @abstractmethod
    def reset_after(self) -> float:
        """Return seconds until the limiter fully resets or refills.

        Returns:
            Seconds until full capacity is restored. Returns ``0.0`` if
            the limiter is already at full capacity.
        """

    def peek(self, cost: int = 1) -> RateLimitResult:
        """Check whether a request would be allowed without consuming capacity.

        This is equivalent to calling :meth:`allow` but without any side
        effects on the limiter state.

        Args:
            cost: Number of units to check. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.
        """
        with self._lock:
            return self._peek_unlocked(cost)

    @abstractmethod
    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Peek implementation without acquiring the lock.

        Subclasses must implement this. It is always called while the lock
        is already held.

        Args:
            cost: Number of units to check.

        Returns:
            A :class:`RateLimitResult` representing what would happen.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the limiter to its initial state.

        After calling this method, the limiter behaves as if it were
        freshly constructed.
        """

    @staticmethod
    def _now() -> float:
        """Return the current monotonic time.

        Uses ``time.monotonic()`` to be immune to system clock adjustments
        (NTP, manual changes, etc.).

        Returns:
            Current monotonic timestamp in seconds.
        """
        return time.monotonic()

    @staticmethod
    def _validate_cost(cost: int) -> None:
        """Validate that cost is a positive integer.

        Args:
            cost: The cost value to validate.

        Raises:
            ValueError: If cost is less than 1.
            TypeError: If cost is not an integer.
        """
        if not isinstance(cost, int):
            raise TypeError(f"cost must be an integer, got {type(cost).__name__}")
        if cost < 1:
            raise ValueError(f"cost must be >= 1, got {cost}")
