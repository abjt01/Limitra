"""Leaky Bucket rate limiting algorithm (water level model)."""

from __future__ import annotations

import math

from limitra._base import (
    RateLimiter,
    RateLimitResult,
    _check_rate,
    _check_size,
    _satisfiable,
)


class LeakyBucket(RateLimiter):
    """Leaky Bucket rate limiter using the water level model.

    The bucket starts empty and fills as requests arrive.  Water drains
    at a constant ``rate`` (units per second).  A request is denied when
    adding ``cost`` would cause the water level to exceed ``capacity``.

    Relationship to :class:`~limitra.TokenBucket`:
        As an admission control device this is the algebraic dual of the
        token bucket rather than a different policy: ``capacity - water``
        tracks the token count exactly, so for the same ``rate`` and
        ``capacity`` the two admit and deny precisely the same requests.
        The "smoothing" a leaky bucket is known for belongs to the
        *queueing* variant, which delays requests instead of refusing
        them — with allow/deny there is nothing to smooth.

        Pick whichever framing matches how you think about the budget:
        ``LeakyBucket`` if you reason about how full a queue is,
        ``TokenBucket`` if you reason about credit you spend and earn back.
        For genuinely paced output either set ``capacity=1``, so no burst
        is possible, or call :meth:`~limitra.RateLimiter.wait` and let the
        limiter space the calls out for you.

    Args:
        rate: Drain rate in units per second. Must be positive.
        capacity: Maximum water level the bucket can hold. This is also
            the largest burst it will admit. Must be at least 1.

    Raises:
        TypeError: If ``rate`` is not a number or ``capacity`` is not an
            integer.
        ValueError: If ``rate`` or ``capacity`` is out of range.

    Example:
        >>> from limitra import LeakyBucket
        >>> limiter = LeakyBucket(rate=10.0, capacity=100)
        >>> limiter.allow().allowed
        True
    """

    __slots__ = ("_capacity", "_last_drain", "_rate", "_water_level")

    def __init__(self, rate: float, capacity: int) -> None:
        """Initialise the leaky bucket.

        Args:
            rate: Drain rate in units per second. Must be positive.
            capacity: Maximum water level. Must be >= 1.

        Raises:
            TypeError: If ``rate`` is not a number or ``capacity`` is not an
                integer.
            ValueError: If ``rate`` or ``capacity`` is out of range.
        """
        super().__init__()
        self._rate: float = _check_rate(rate)
        self._capacity: int = _check_size(capacity, "capacity")
        self._water_level: float = 0.0
        self._last_drain: float = self._now()

    # -- configuration ---------------------------------------------------- #

    @property
    def rate(self) -> float:
        """Drain rate in units per second."""
        return self._rate

    @property
    def capacity(self) -> int:
        """Maximum water the bucket can hold."""
        return self._capacity

    @property
    def limit(self) -> int:
        """Maximum units admissible at once — an alias for :attr:`capacity`."""
        return self._capacity

    # -- internal helpers ------------------------------------------------- #

    def _drain(self) -> None:
        """Drain water that has leaked since the last drain.

        Water drained equals ``elapsed * rate``, and the level is clamped
        to a minimum of ``0.0``.  Updates ``_last_drain`` to the current
        time.
        """
        now = self._now()
        elapsed = now - self._last_drain
        self._water_level = max(0.0, self._water_level - elapsed * self._rate)
        self._last_drain = now

    # -- public API ------------------------------------------------------- #

    def allow(self, cost: int = 1) -> RateLimitResult:
        """Attempt to add ``cost`` units of water to the bucket.

        If the resulting water level would exceed ``capacity``, the
        request is denied.

        Args:
            cost: Number of units to add. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` with the outcome.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        with self._lock:
            self._drain()
            if self._water_level + cost <= self._capacity:
                self._water_level += cost
                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, self._capacity - math.ceil(self._water_level)),
                    limit=self._capacity,
                    reset_after=max(0.0, self._water_level / self._rate),
                    retry_after=0.0,
                )
            retry = (self._water_level + cost - self._capacity) / self._rate
            return RateLimitResult(
                allowed=False,
                remaining=max(0, self._capacity - math.ceil(self._water_level)),
                limit=self._capacity,
                reset_after=max(0.0, self._water_level / self._rate),
                retry_after=_satisfiable(retry, self._now()),
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Check whether ``cost`` units could be added without doing so.

        This method does **not** acquire the lock — it is called from
        :meth:`~RateLimiter.peek` which already holds it.

        Args:
            cost: Number of units to check. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        self._drain()
        if self._water_level + cost <= self._capacity:
            level_after = self._water_level + cost
            return RateLimitResult(
                allowed=True,
                remaining=max(0, self._capacity - math.ceil(level_after)),
                limit=self._capacity,
                reset_after=max(0.0, level_after / self._rate),
                retry_after=0.0,
            )
        retry = (self._water_level + cost - self._capacity) / self._rate
        return RateLimitResult(
            allowed=False,
            remaining=max(0, self._capacity - math.ceil(self._water_level)),
            limit=self._capacity,
            reset_after=max(0.0, self._water_level / self._rate),
            retry_after=_satisfiable(retry, self._now()),
        )

    def remaining(self) -> int:
        """Return the number of units that can still be added.

        Returns:
            Available capacity computed as ``capacity - ceil(water_level)``.
        """
        with self._lock:
            self._drain()
            return max(0, self._capacity - math.ceil(self._water_level))

    def reset_after(self) -> float:
        """Return seconds until the bucket is completely empty.

        Returns:
            Seconds until the water level reaches zero. Returns ``0.0``
            if already empty.
        """
        with self._lock:
            self._drain()
            return max(0.0, self._water_level / self._rate)

    def refund(self, cost: int = 1) -> None:
        """Remove ``cost`` units of water, never going below empty.

        Args:
            cost: Number of units to return. Defaults to 1.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range.
        """
        self._validate_cost(cost)
        with self._lock:
            self._drain()
            self._water_level = max(0.0, self._water_level - cost)

    def reset(self) -> None:
        """Reset the bucket to empty.

        After calling this method, the bucket behaves as if freshly
        constructed.
        """
        with self._lock:
            self._water_level = 0.0
            self._last_drain = self._now()

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return f"LeakyBucket(rate={self._rate}, capacity={self._capacity})"
