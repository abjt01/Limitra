"""Token Bucket rate limiting algorithm."""

from __future__ import annotations

from limitra._base import (
    RateLimiter,
    RateLimitResult,
    _check_rate,
    _check_size,
    _satisfiable,
)


class TokenBucket(RateLimiter):
    """Token Bucket rate limiter.

    Tokens are added to the bucket at a fixed ``rate`` (tokens per second),
    up to a maximum ``capacity``. Each :meth:`allow` call consumes tokens,
    and requests are denied when insufficient tokens are available.

    This algorithm is well suited for smoothing bursty traffic while still
    permitting short bursts up to ``capacity``.

    Args:
        rate: Tokens added per second. Must be positive.
        capacity: Maximum number of tokens the bucket can hold. Must be
            at least 1.

    Raises:
        ValueError: If ``rate`` or ``capacity`` is out of range.

    Example:
        >>> from limitra import TokenBucket
        >>> limiter = TokenBucket(rate=10.0, capacity=100)
        >>> limiter.allow().allowed
        True
    """

    __slots__ = ("_capacity", "_last_refill", "_rate", "_tokens")

    def __init__(self, rate: float, capacity: int) -> None:
        """Initialise the token bucket.

        Args:
            rate: Tokens added per second. Must be positive.
            capacity: Maximum tokens the bucket can hold. Must be >= 1.

        Raises:
            TypeError: If ``rate`` is not a number or ``capacity`` is not an
                integer.
            ValueError: If ``rate`` or ``capacity`` is out of range.
        """
        super().__init__()
        self._rate: float = _check_rate(rate)
        self._capacity: int = _check_size(capacity, "capacity")
        self._tokens: float = float(capacity)
        self._last_refill: float = self._now()

    # -- configuration ---------------------------------------------------- #

    @property
    def rate(self) -> float:
        """Tokens added per second."""
        return self._rate

    @property
    def capacity(self) -> int:
        """Maximum tokens the bucket can hold."""
        return self._capacity

    @property
    def limit(self) -> int:
        """Maximum units admissible at once — an alias for :attr:`capacity`."""
        return self._capacity

    # -- internal helpers ------------------------------------------------- #

    def _refill(self) -> None:
        """Add tokens accrued since the last refill.

        Tokens are calculated as ``elapsed * rate`` and capped at
        ``capacity``.  Updates ``_last_refill`` to the current time.
        """
        now = self._now()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    # -- public API ------------------------------------------------------- #

    def allow(self, cost: int = 1) -> RateLimitResult:
        """Attempt to consume ``cost`` tokens from the bucket.

        Args:
            cost: Number of tokens to consume. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` with the outcome.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, int(self._tokens)),
                    limit=self._capacity,
                    reset_after=max(0.0, (self._capacity - self._tokens) / self._rate),
                    retry_after=0.0,
                )
            return RateLimitResult(
                allowed=False,
                remaining=max(0, int(self._tokens)),
                limit=self._capacity,
                reset_after=max(0.0, (self._capacity - self._tokens) / self._rate),
                retry_after=_satisfiable(
                    (cost - self._tokens) / self._rate, self._now()
                ),
            )

    def _peek_unlocked(self, cost: int = 1) -> RateLimitResult:
        """Check whether ``cost`` tokens are available without consuming them.

        This method does **not** acquire the lock — it is called from
        :meth:`~RateLimiter.peek` which already holds it.

        Args:
            cost: Number of tokens to check. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is less than 1 or greater than
                :attr:`limit`.
        """
        self._validate_cost(cost)
        self._refill()
        if self._tokens >= cost:
            remaining_after = self._tokens - cost
            return RateLimitResult(
                allowed=True,
                remaining=max(0, int(remaining_after)),
                limit=self._capacity,
                reset_after=max(0.0, (self._capacity - remaining_after) / self._rate),
                retry_after=0.0,
            )
        return RateLimitResult(
            allowed=False,
            remaining=max(0, int(self._tokens)),
            limit=self._capacity,
            reset_after=max(0.0, (self._capacity - self._tokens) / self._rate),
            retry_after=_satisfiable((cost - self._tokens) / self._rate, self._now()),
        )

    def remaining(self) -> int:
        """Return the number of tokens currently available.

        Returns:
            Number of whole tokens available right now.
        """
        with self._lock:
            self._refill()
            return max(0, int(self._tokens))

    def reset_after(self) -> float:
        """Return seconds until the bucket is completely full.

        Returns:
            Seconds until the bucket reaches ``capacity``. Returns ``0.0``
            if already full.
        """
        with self._lock:
            self._refill()
            return max(0.0, (self._capacity - self._tokens) / self._rate)

    def refund(self, cost: int = 1) -> None:
        """Return ``cost`` tokens to the bucket, never exceeding capacity.

        Args:
            cost: Number of tokens to return. Defaults to 1.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range.
        """
        self._validate_cost(cost)
        with self._lock:
            self._refill()
            self._tokens = min(self._capacity, self._tokens + cost)

    def reset(self) -> None:
        """Reset the bucket to full capacity.

        After calling this method, the bucket behaves as if freshly
        constructed.
        """
        with self._lock:
            self._tokens = float(self._capacity)
            self._last_refill = self._now()

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return f"TokenBucket(rate={self._rate}, capacity={self._capacity})"
