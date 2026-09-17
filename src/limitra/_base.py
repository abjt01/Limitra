"""Abstract base class and shared types for rate limiting algorithms."""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Floor for :meth:`RateLimiter.wait` sleeps, so an optimistic ``retry_after``
#: of ``0.0`` cannot turn the wait loop into a busy spin.
_MIN_WAIT_SLEEP = 0.001


def _satisfiable(seconds: float) -> float:
    """Round a wait up by one ULP so sleeping exactly this long succeeds.

    ``retry_after`` inverts the same float arithmetic the limiter will redo
    on the next call, so the exact result can land one unit in the last
    place short and deny a caller who waited precisely as instructed.

    Args:
        seconds: The computed wait, which may be zero or negative.

    Returns:
        The wait, nudged just past the boundary. Never negative.
    """
    if seconds <= 0.0:
        return 0.0
    return math.nextafter(seconds, math.inf)


def _check_rate(rate: float) -> float:
    """Validate a per-second rate.

    ``rate <= 0`` alone is not enough: the comparison is ``False`` for NaN,
    which would leave the limiter fully open while reporting itself healthy.

    Args:
        rate: The rate to validate.

    Returns:
        The rate as a float.

    Raises:
        TypeError: If ``rate`` is not a real number.
        ValueError: If ``rate`` is not finite and positive.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise TypeError(f"rate must be a number, got {type(rate).__name__}")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"rate must be > 0 and finite, got {rate}")
    return float(rate)


def _check_window(window: float) -> float:
    """Validate a window duration in seconds.

    Args:
        window: The window to validate.

    Returns:
        The window as a float.

    Raises:
        TypeError: If ``window`` is not a real number.
        ValueError: If ``window`` is not finite and positive.
    """
    if isinstance(window, bool) or not isinstance(window, (int, float)):
        raise TypeError(f"window must be a number, got {type(window).__name__}")
    if not math.isfinite(window) or window <= 0:
        raise ValueError(f"window must be > 0 and finite, got {window}")
    return float(window)


def _check_size(value: int, name: str) -> int:
    """Validate a capacity or limit.

    These end up in :attr:`RateLimitResult.limit` and in HTTP headers, so a
    float would surface as ``X-RateLimit-Limit: 10.5``.

    Args:
        value: The capacity or limit to validate.
        name: The parameter name, for the error message.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` is less than 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RateLimitResult:
    """Immutable result of a rate limit check.

    Provides complete context about the rate limit decision, suitable for
    populating HTTP response headers (see :meth:`as_headers`).

    Instances are frozen, hashable, and compare by value across all five
    fields.

    Attributes:
        allowed: Whether the request was permitted.
        remaining: Units still available after this decision.
        limit: Maximum number of units allowed.
        reset_after: Seconds until the limiter is back at full capacity.
            ``0.0`` when it already is.
        retry_after: Seconds until a denied request could succeed. Always
            ``0.0`` when ``allowed`` is ``True``.
    """

    allowed: bool
    remaining: int
    limit: int
    reset_after: float
    retry_after: float

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

    def as_headers(self) -> dict[str, str]:
        """Render the result as rate limit HTTP response headers.

        Every time value is **seconds from now**, never a Unix timestamp,
        and is rounded up to whole seconds so a client that obeys the
        headers never comes back too early.

        Both the standard ``RateLimit-*`` names and the older
        ``X-RateLimit-*`` ones are emitted with identical values, because
        clients disagree about which to read. ``Retry-After`` appears only
        when the request was denied.

        Returns:
            A mapping of header name to header value.
        """
        limit = str(self.limit)
        remaining = str(self.remaining)
        reset = str(math.ceil(self.reset_after))
        headers = {
            "RateLimit-Limit": limit,
            "RateLimit-Remaining": remaining,
            "RateLimit-Reset": reset,
            "X-RateLimit-Limit": limit,
            "X-RateLimit-Remaining": remaining,
            "X-RateLimit-Reset": reset,
        }
        if not self.allowed:
            headers["Retry-After"] = str(math.ceil(self.retry_after))
        return headers


class RateLimiter(ABC):
    """Abstract base class for all rate limiting algorithms.

    Every rate limiter implementation must subclass ``RateLimiter`` and
    implement :meth:`allow`, :meth:`remaining`, :meth:`reset_after`,
    :meth:`reset`, :meth:`_peek_unlocked`, and the :attr:`limit` property.

    All implementations guarantee:
        - Thread safety via ``threading.Lock``
        - Clock-drift immunity via ``time.monotonic()``
        - Consistent API regardless of underlying algorithm

    State is held in memory and is local to a single process. Running
    several worker processes (gunicorn, ``uvicorn --workers``, multiple
    containers) gives each worker its own independent limiter, so the
    effective limit is multiplied by the worker count. Use a shared backend
    if you need a cluster-wide limit.

    Example:
        >>> from limitra import TokenBucket
        >>> limiter = TokenBucket(rate=10.0, capacity=100)
        >>> limiter.allow().allowed
        True
    """

    __slots__ = ("__weakref__", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    @abstractmethod
    def limit(self) -> int:
        """Maximum number of units the limiter will ever admit at once.

        This is ``capacity`` for the bucket algorithms and ``limit`` for the
        window algorithms. A ``cost`` greater than this value can never be
        satisfied, so :meth:`allow` rejects it with ``ValueError``.

        Returns:
            The limiter's maximum burst size.
        """

    @abstractmethod
    def allow(self, cost: int = 1) -> RateLimitResult:
        """Attempt to consume ``cost`` units from the rate limiter.

        Args:
            cost: Number of units to consume. Defaults to 1. Must be an
                integer in the range ``1 <= cost <= limit``.

        Returns:
            A :class:`RateLimitResult` indicating whether the request was
            allowed and providing metadata about the current limiter state.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """

    @abstractmethod
    def remaining(self) -> int:
        """Return the number of units still available.

        This is a snapshot — the value may change immediately after reading
        in a multi-threaded environment.

        Returns:
            Units that could be consumed right now.
        """

    @abstractmethod
    def reset_after(self) -> float:
        """Return seconds until the limiter is back at full capacity.

        Returns:
            Seconds until nothing is held against the limiter any more.
            ``0.0`` exactly when it is already at full capacity.
        """

    def peek(self, cost: int = 1) -> RateLimitResult:
        """Report what :meth:`allow` would return, without consuming anything.

        Time-based bookkeeping (refilling, draining, expiring old entries)
        still happens, since that only reflects elapsed time. The
        ``remaining`` of an allowed peek is what would be left *after* the
        hypothetical request, matching :meth:`allow`.

        Args:
            cost: Number of units to check. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
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

    def wait(self, cost: int = 1, timeout: float | None = None) -> RateLimitResult:
        """Block until ``cost`` units are available, then consume them.

        Sleeps for the ``retry_after`` reported by each denied attempt, so a
        caller pacing itself against a remote API does not busy-loop. The
        limiter's lock is never held while sleeping.

        In an asyncio program use :meth:`wait_async` instead — this call
        blocks the whole event loop.

        Args:
            cost: Number of units to consume. Defaults to 1.
            timeout: Maximum seconds to wait. ``None`` (the default) waits
                indefinitely. ``0`` makes this a single :meth:`allow` call.

        Returns:
            The successful :class:`RateLimitResult`, or the final denied
            result if ``timeout`` elapsed first. Check ``.allowed`` to tell
            the two apart.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range, or ``timeout`` is
                negative.
        """
        deadline = self._wait_deadline(timeout)
        while True:
            result = self.allow(cost)
            if result.allowed:
                return result
            delay = self._wait_delay(result, deadline)
            if delay is None:
                return result
            time.sleep(delay)

    async def wait_async(
        self, cost: int = 1, timeout: float | None = None
    ) -> RateLimitResult:
        """Await until ``cost`` units are available, then consume them.

        The asyncio counterpart to :meth:`wait`: identical semantics, but it
        yields to the event loop instead of blocking it.

        Args:
            cost: Number of units to consume. Defaults to 1.
            timeout: Maximum seconds to wait. ``None`` (the default) waits
                indefinitely. ``0`` makes this a single :meth:`allow` call.

        Returns:
            The successful :class:`RateLimitResult`, or the final denied
            result if ``timeout`` elapsed first.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range, or ``timeout`` is
                negative.
        """
        import asyncio

        deadline = self._wait_deadline(timeout)
        while True:
            result = self.allow(cost)
            if result.allowed:
                return result
            delay = self._wait_delay(result, deadline)
            if delay is None:
                return result
            await asyncio.sleep(delay)

    def _wait_deadline(self, timeout: float | None) -> float | None:
        """Turn a timeout into a monotonic deadline.

        Args:
            timeout: Seconds to wait, or ``None`` for no limit.

        Returns:
            The monotonic deadline, or ``None`` if there is no limit.

        Raises:
            ValueError: If ``timeout`` is negative.
        """
        if timeout is None:
            return None
        if timeout < 0:
            raise ValueError(f"timeout must be >= 0 or None, got {timeout}")
        return self._now() + timeout

    def _wait_delay(
        self, denied: RateLimitResult, deadline: float | None
    ) -> float | None:
        """Return how long to sleep before retrying, or ``None`` to give up.

        Args:
            denied: The denial that triggered this wait.
            deadline: Monotonic deadline, or ``None`` for no limit.

        Returns:
            Seconds to sleep, or ``None`` if the deadline has passed.
        """
        delay = max(denied.retry_after, _MIN_WAIT_SLEEP)
        if deadline is None:
            return delay
        left = deadline - self._now()
        if left <= 0:
            return None
        return min(delay, left)

    @staticmethod
    def _now() -> float:
        """Return the current monotonic time.

        Uses ``time.monotonic()`` to be immune to system clock adjustments
        (NTP, manual changes, etc.).

        Returns:
            Current monotonic timestamp in seconds.
        """
        return time.monotonic()

    def _validate_cost(self, cost: int) -> None:
        """Validate that ``cost`` is an integer the limiter could satisfy.

        Args:
            cost: The cost value to validate.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or exceeds :attr:`limit`.
        """
        if isinstance(cost, bool) or not isinstance(cost, int):
            raise TypeError(f"cost must be an integer, got {type(cost).__name__}")
        if cost < 1:
            raise ValueError(f"cost must be >= 1, got {cost}")
        limit = self.limit
        if cost > limit:
            raise ValueError(
                f"cost must be <= limit ({limit}), got {cost}; a request this "
                f"large can never be allowed by this limiter"
            )
