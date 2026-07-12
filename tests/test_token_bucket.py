"""Comprehensive tests for the TokenBucket rate limiter."""

import time

import pytest

from limitra import RateLimitResult, TokenBucket

# ── Initialization ──────────────────────────────────────────────────────── #


def test_init_valid():
    """TokenBucket can be created with valid rate and capacity."""
    limiter = TokenBucket(rate=10.0, capacity=100)
    assert limiter is not None


def test_init_invalid_rate():
    """Rate <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="rate must be > 0"):
        TokenBucket(rate=0, capacity=10)
    with pytest.raises(ValueError, match="rate must be > 0"):
        TokenBucket(rate=-5, capacity=10)


def test_init_invalid_capacity():
    """Capacity < 1 raises ValueError."""
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        TokenBucket(rate=10.0, capacity=0)
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        TokenBucket(rate=10.0, capacity=-1)


# ── Basic allow / deny ──────────────────────────────────────────────────── #


def test_allow_single():
    """A fresh limiter allows a single request."""
    limiter = TokenBucket(rate=10.0, capacity=10)
    result = limiter.allow()
    assert result.allowed is True


def test_allow_returns_result():
    """allow() returns a RateLimitResult with all expected fields."""
    limiter = TokenBucket(rate=10.0, capacity=10)
    result = limiter.allow()
    assert isinstance(result, RateLimitResult)
    assert hasattr(result, "allowed")
    assert hasattr(result, "remaining")
    assert hasattr(result, "limit")
    assert hasattr(result, "reset_after")
    assert hasattr(result, "retry_after")


def test_exhaust_capacity():
    """Allowing capacity times exhausts the bucket, then denies."""
    capacity = 5
    limiter = TokenBucket(rate=1.0, capacity=capacity)
    for _ in range(capacity):
        result = limiter.allow()
        assert result.allowed is True, "should be allowed while tokens remain"
    denied = limiter.allow()
    assert denied.allowed is False, "should be denied after exhausting capacity"


# ── Remaining ────────────────────────────────────────────────────────────── #


def test_remaining_decreases():
    """Remaining goes down with each allow call."""
    limiter = TokenBucket(rate=1.0, capacity=5)
    assert limiter.remaining() == 5
    limiter.allow()
    assert limiter.remaining() == 4
    limiter.allow()
    assert limiter.remaining() == 3


def test_remaining_full():
    """Remaining equals capacity on a fresh limiter."""
    limiter = TokenBucket(rate=10.0, capacity=20)
    assert limiter.remaining() == 20


# ── Refill over time ────────────────────────────────────────────────────── #


def test_refill_over_time():
    """Tokens refill after waiting (rate=100, sleep 0.15s → ~15 tokens)."""
    limiter = TokenBucket(rate=100.0, capacity=20)
    # Exhaust all tokens
    for _ in range(20):
        limiter.allow()
    assert limiter.remaining() == 0, "bucket should be empty"
    time.sleep(0.15)
    refilled = limiter.remaining()
    assert refilled > 0, "tokens should have refilled after sleeping"
    assert refilled <= 20, "remaining should not exceed capacity"


# ── Cost parameter ──────────────────────────────────────────────────────── #


def test_cost_parameter():
    """allow(cost=5) consumes 5 tokens at once."""
    limiter = TokenBucket(rate=1.0, capacity=10)
    result = limiter.allow(cost=5)
    assert result.allowed is True
    assert limiter.remaining() == 5


def test_cost_exceeds_capacity():
    """A cost greater than capacity is denied (but is a valid cost value)."""
    limiter = TokenBucket(rate=1.0, capacity=5)
    result = limiter.allow(cost=6)
    assert result.allowed is False


def test_cost_validation():
    """cost=0 and cost=-1 raise ValueError; cost='a' raises TypeError."""
    limiter = TokenBucket(rate=10.0, capacity=10)
    with pytest.raises(ValueError):
        limiter.allow(cost=0)
    with pytest.raises(ValueError):
        limiter.allow(cost=-1)
    with pytest.raises(TypeError):
        limiter.allow(cost="a")  # type: ignore[arg-type]


# ── Reset ────────────────────────────────────────────────────────────────── #


def test_reset():
    """reset() restores the bucket to full capacity."""
    limiter = TokenBucket(rate=1.0, capacity=10)
    for _ in range(10):
        limiter.allow()
    assert limiter.remaining() == 0
    limiter.reset()
    assert limiter.remaining() == 10


# ── Peek ─────────────────────────────────────────────────────────────────── #


def test_peek_no_side_effect():
    """peek() does not consume tokens."""
    limiter = TokenBucket(rate=1.0, capacity=10)
    before = limiter.remaining()
    peek_result = limiter.peek()
    after = limiter.remaining()
    assert before == after, "peek should not change remaining tokens"
    assert peek_result.allowed is True


# ── reset_after ──────────────────────────────────────────────────────────── #


def test_reset_after_returns_time_to_full():
    """reset_after() returns time to full capacity."""
    limiter = TokenBucket(rate=10.0, capacity=10)
    # Fresh limiter is already full
    assert limiter.reset_after() == pytest.approx(0.0, abs=0.05)
    # Consume some tokens
    limiter.allow(cost=5)
    ra = limiter.reset_after()
    assert ra > 0, "reset_after should be positive after consuming tokens"


# ── __repr__ ─────────────────────────────────────────────────────────────── #


def test_repr():
    """Repr contains 'TokenBucket'."""
    limiter = TokenBucket(rate=10.0, capacity=100)
    assert "TokenBucket" in repr(limiter)


# ── Result truthiness ────────────────────────────────────────────────────── #


def test_result_bool():
    """RateLimitResult is truthy when allowed, falsy when denied."""
    limiter = TokenBucket(rate=1.0, capacity=1)
    allowed = limiter.allow()
    assert bool(allowed) is True
    denied = limiter.allow()
    assert bool(denied) is False


# ── retry_after ──────────────────────────────────────────────────────────── #


def test_retry_after_zero_when_allowed():
    """retry_after is 0.0 when the request is allowed."""
    limiter = TokenBucket(rate=10.0, capacity=10)
    result = limiter.allow()
    assert result.retry_after == 0.0


def test_retry_after_positive_when_denied():
    """retry_after > 0 when the request is denied."""
    limiter = TokenBucket(rate=1.0, capacity=1)
    limiter.allow()  # exhaust
    denied = limiter.allow()
    assert denied.retry_after > 0, "retry_after should be positive when denied"
