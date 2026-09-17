"""Parametrized tests proving all 5 algorithms share an identical API.

This is a KEY test file — it's the proof of Limitra's unified design.
Every algorithm must pass every test identically, demonstrating that
swapping algorithms is a one-line change.
"""

from __future__ import annotations

import pytest

from limitra import (
    FixedWindow,
    LeakyBucket,
    RateLimiter,
    RateLimitResult,
    SlidingLog,
    SlidingWindow,
    TokenBucket,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_limiter(cls: type[RateLimiter]) -> RateLimiter:
    """Create any limiter with equivalent semantics (≈20 requests allowed)."""
    if cls in (TokenBucket, LeakyBucket):
        return cls(rate=10.0, capacity=20)
    return cls(limit=20, window=2.0)


ALL_ALGORITHMS = [TokenBucket, LeakyBucket, FixedWindow, SlidingWindow, SlidingLog]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
class TestUnifiedAPI:
    """Prove every algorithm honours the RateLimiter contract."""

    # -- type & hierarchy ---------------------------------------------------

    def test_is_rate_limiter_subclass(self, algorithm: type[RateLimiter]) -> None:
        """Every algorithm is a subclass of RateLimiter."""
        assert issubclass(algorithm, RateLimiter)

    # -- method existence ---------------------------------------------------

    def test_has_allow_method(self, algorithm: type[RateLimiter]) -> None:
        """Has allow() accepting cost kwarg."""
        limiter = make_limiter(algorithm)
        result = limiter.allow(cost=1)
        assert isinstance(result, RateLimitResult)

    def test_has_remaining_method(self, algorithm: type[RateLimiter]) -> None:
        """Has remaining() returning int."""
        limiter = make_limiter(algorithm)
        assert isinstance(limiter.remaining(), int)

    def test_has_reset_after_method(self, algorithm: type[RateLimiter]) -> None:
        """Has reset_after() returning float."""
        limiter = make_limiter(algorithm)
        assert isinstance(limiter.reset_after(), float)

    def test_has_peek_method(self, algorithm: type[RateLimiter]) -> None:
        """Has peek() returning RateLimitResult."""
        limiter = make_limiter(algorithm)
        result = limiter.peek()
        assert isinstance(result, RateLimitResult)

    def test_has_reset_method(self, algorithm: type[RateLimiter]) -> None:
        """Has reset() callable."""
        limiter = make_limiter(algorithm)
        assert limiter.reset() is None  # returns None, doesn't raise

    # -- return types -------------------------------------------------------

    def test_allow_returns_rate_limit_result(
        self, algorithm: type[RateLimiter]
    ) -> None:
        """allow() returns RateLimitResult."""
        limiter = make_limiter(algorithm)
        result = limiter.allow()
        assert isinstance(result, RateLimitResult)

    # -- basic behaviour ----------------------------------------------------

    def test_allow_fresh_limiter(self, algorithm: type[RateLimiter]) -> None:
        """A fresh limiter allows the first request."""
        limiter = make_limiter(algorithm)
        result = limiter.allow()
        assert result.allowed is True

    def test_remaining_non_negative(self, algorithm: type[RateLimiter]) -> None:
        """Remaining is always >= 0."""
        limiter = make_limiter(algorithm)
        assert limiter.remaining() >= 0
        # Even after exhausting capacity, remaining must stay non-negative
        for _ in range(25):
            limiter.allow()
        assert limiter.remaining() >= 0

    def test_reset_after_non_negative(self, algorithm: type[RateLimiter]) -> None:
        """reset_after is always >= 0."""
        limiter = make_limiter(algorithm)
        assert limiter.reset_after() >= 0.0
        limiter.allow()
        assert limiter.reset_after() >= 0.0

    def test_peek_returns_same_allowed_state(
        self, algorithm: type[RateLimiter]
    ) -> None:
        """Peek matches allow's allowed state without consuming capacity."""
        limiter = make_limiter(algorithm)
        peek_result = limiter.peek()
        remaining_before = limiter.remaining()

        # peek should not change remaining
        peek_result_2 = limiter.peek()
        remaining_after = limiter.remaining()

        assert peek_result.allowed == peek_result_2.allowed
        assert remaining_before == remaining_after

    # -- result fields ------------------------------------------------------

    def test_result_has_all_fields(self, algorithm: type[RateLimiter]) -> None:
        """Result has allowed, remaining, limit, reset_after, retry_after."""
        limiter = make_limiter(algorithm)
        result = limiter.allow()

        assert hasattr(result, "allowed")
        assert hasattr(result, "remaining")
        assert hasattr(result, "limit")
        assert hasattr(result, "reset_after")
        assert hasattr(result, "retry_after")

        assert isinstance(result.allowed, bool)
        assert isinstance(result.remaining, int)
        assert isinstance(result.limit, int)
        assert isinstance(result.reset_after, float)
        assert isinstance(result.retry_after, float)

    # -- validation ---------------------------------------------------------

    def test_cost_validation_value_error(self, algorithm: type[RateLimiter]) -> None:
        """cost=0 raises ValueError."""
        limiter = make_limiter(algorithm)
        with pytest.raises(ValueError):
            limiter.allow(cost=0)

    def test_cost_validation_type_error(self, algorithm: type[RateLimiter]) -> None:
        """cost='x' raises TypeError."""
        limiter = make_limiter(algorithm)
        with pytest.raises(TypeError):
            limiter.allow(cost="x")  # type: ignore[arg-type]

    # -- reset --------------------------------------------------------------

    def test_reset_restores_capacity(self, algorithm: type[RateLimiter]) -> None:
        """After reset, remaining returns to max."""
        limiter = make_limiter(algorithm)
        initial_remaining = limiter.remaining()

        # Consume some capacity
        for _ in range(5):
            limiter.allow()

        assert limiter.remaining() < initial_remaining

        # Reset should restore capacity
        limiter.reset()
        assert limiter.remaining() == initial_remaining

    # -- repr ---------------------------------------------------------------

    def test_repr_is_string(self, algorithm: type[RateLimiter]) -> None:
        """Repr returns a non-empty string."""
        limiter = make_limiter(algorithm)
        r = repr(limiter)
        assert isinstance(r, str)
        assert len(r) > 0

    # -- algorithm swap (the one-line swap promise) -------------------------

    def test_algorithm_swap(self, algorithm: type[RateLimiter]) -> None:
        """Same code works with any algorithm — prove the one-line swap promise.

        This test runs identical application logic against every algorithm
        to demonstrate that switching algorithms requires zero code changes.
        """
        limiter = make_limiter(algorithm)

        # Application code that works identically with any algorithm:
        result = limiter.allow(cost=1)
        assert isinstance(result, RateLimitResult)
        assert isinstance(result.allowed, bool)

        _ = limiter.remaining()
        _ = limiter.reset_after()

        peek = limiter.peek(cost=1)
        assert isinstance(peek, RateLimitResult)

        limiter.reset()
        assert limiter.remaining() > 0


# ---------------------------------------------------------------------------
# Configuration is readable back off the limiter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_limit_property_reports_max_burst(algorithm: type[RateLimiter]) -> None:
    """Every algorithm exposes the largest cost it could ever admit."""
    limiter = make_limiter(algorithm)
    assert limiter.limit == 20
    assert limiter.allow(cost=20).allowed is True


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_limit_matches_the_result_limit(algorithm: type[RateLimiter]) -> None:
    """The limiter's limit and the result's limit agree."""
    limiter = make_limiter(algorithm)
    assert limiter.allow().limit == limiter.limit


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_cost_above_limit_is_rejected(algorithm: type[RateLimiter]) -> None:
    """A cost no configuration could satisfy raises instead of denying.

    Returning a denial would hand the caller a ``retry_after`` that will
    never come true, so every algorithm refuses up front.
    """
    limiter = make_limiter(algorithm)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        limiter.allow(cost=limiter.limit + 1)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        limiter.peek(cost=limiter.limit + 1)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_bool_cost_is_rejected(algorithm: type[RateLimiter]) -> None:
    """``True`` is an int in Python, but passing it as a cost is a mistake."""
    limiter = make_limiter(algorithm)
    with pytest.raises(TypeError, match="cost must be an integer, got bool"):
        limiter.allow(cost=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("algorithm", [TokenBucket, LeakyBucket])
def test_bucket_config_accessors(algorithm: type[RateLimiter]) -> None:
    """Bucket algorithms expose the rate and capacity they were built with."""
    limiter = algorithm(rate=12.5, capacity=30)
    assert limiter.rate == 12.5
    assert limiter.capacity == 30
    assert limiter.limit == 30


@pytest.mark.parametrize("algorithm", [FixedWindow, SlidingWindow, SlidingLog])
def test_window_config_accessors(algorithm: type[RateLimiter]) -> None:
    """Window algorithms expose the limit and window they were built with."""
    limiter = algorithm(limit=30, window=12.5)
    assert limiter.limit == 30
    assert limiter.window == 12.5


# ---------------------------------------------------------------------------
# Peek on an exhausted limiter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_peek_reports_denial_once_exhausted(algorithm: type[RateLimiter]) -> None:
    """Peek agrees with allow after capacity is spent, and stays read-only."""
    limiter = make_limiter(algorithm)
    limiter.allow(cost=20)

    peeked = limiter.peek()
    assert peeked.allowed is False
    assert peeked.remaining == 0
    assert peeked.retry_after > 0.0, "a denial must say when to come back"

    # Peeking repeatedly must not move the limiter.
    assert limiter.peek() == peeked or limiter.peek().allowed is False
    assert limiter.allow().allowed is False, "peek must not have freed capacity"


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_peek_agrees_with_allow(algorithm: type[RateLimiter]) -> None:
    """Whatever peek predicts, the very next allow delivers."""
    limiter = make_limiter(algorithm)
    for _ in range(25):
        predicted = limiter.peek().allowed
        assert limiter.allow().allowed is predicted
