"""Comprehensive tests for the LeakyBucket rate limiter."""

import time

import pytest

from limitra import LeakyBucket, RateLimitResult

# ── Initialization ──────────────────────────────────────────────────────── #


def test_init_valid():
    """LeakyBucket can be created with valid rate and capacity."""
    limiter = LeakyBucket(rate=10.0, capacity=100)
    assert limiter is not None


def test_init_invalid_rate():
    """Rate <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="rate must be > 0"):
        LeakyBucket(rate=0, capacity=10)
    with pytest.raises(ValueError, match="rate must be > 0"):
        LeakyBucket(rate=-5, capacity=10)


def test_init_invalid_capacity():
    """Capacity < 1 raises ValueError."""
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        LeakyBucket(rate=10.0, capacity=0)
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        LeakyBucket(rate=10.0, capacity=-1)


# ── Basic allow / deny ──────────────────────────────────────────────────── #


def test_allow_single():
    """A fresh (empty) limiter allows a single request."""
    limiter = LeakyBucket(rate=10.0, capacity=10)
    result = limiter.allow()
    assert result.allowed is True


def test_allow_returns_result():
    """allow() returns a RateLimitResult with all expected fields."""
    limiter = LeakyBucket(rate=10.0, capacity=10)
    result = limiter.allow()
    assert isinstance(result, RateLimitResult)
    assert hasattr(result, "allowed")
    assert hasattr(result, "remaining")
    assert hasattr(result, "limit")
    assert hasattr(result, "reset_after")
    assert hasattr(result, "retry_after")


def test_exhaust_capacity():
    """Filling to capacity then requesting one more should deny."""
    capacity = 5
    limiter = LeakyBucket(rate=1.0, capacity=capacity)
    for _ in range(capacity):
        result = limiter.allow()
        assert result.allowed is True, "should be allowed while bucket has room"
    denied = limiter.allow()
    assert denied.allowed is False, "should be denied once bucket is full"


# ── Remaining ────────────────────────────────────────────────────────────── #


def test_remaining_decreases():
    """Remaining goes down as water fills the bucket."""
    limiter = LeakyBucket(rate=1.0, capacity=5)
    assert limiter.remaining() == 5
    limiter.allow()
    assert limiter.remaining() == 4
    limiter.allow()
    assert limiter.remaining() == 3


def test_remaining_full_when_empty():
    """Remaining equals capacity when bucket is empty (fresh)."""
    limiter = LeakyBucket(rate=10.0, capacity=20)
    assert limiter.remaining() == 20


# ── Drain over time ─────────────────────────────────────────────────────── #


def test_drain_over_time():
    """Water drains after waiting (rate=100, sleep 0.15s → ~15 units drain)."""
    limiter = LeakyBucket(rate=100.0, capacity=20)
    # Fill the bucket completely
    for _ in range(20):
        limiter.allow()
    assert limiter.remaining() == 0, "bucket should be full (no remaining)"
    time.sleep(0.15)
    drained = limiter.remaining()
    assert drained > 0, "some water should have drained after sleeping"
    assert drained <= 20, "remaining should not exceed capacity"


# ── Cost parameter ──────────────────────────────────────────────────────── #


def test_cost_parameter():
    """allow(cost=5) adds 5 units of water to the bucket."""
    limiter = LeakyBucket(rate=1.0, capacity=10)
    result = limiter.allow(cost=5)
    assert result.allowed is True
    assert limiter.remaining() == 5


def test_cost_exceeds_capacity():
    """A cost greater than capacity is denied."""
    limiter = LeakyBucket(rate=1.0, capacity=5)
    result = limiter.allow(cost=6)
    assert result.allowed is False


def test_cost_validation():
    """cost=0 and cost=-1 raise ValueError."""
    limiter = LeakyBucket(rate=10.0, capacity=10)
    with pytest.raises(ValueError):
        limiter.allow(cost=0)
    with pytest.raises(ValueError):
        limiter.allow(cost=-1)


# ── Reset ────────────────────────────────────────────────────────────────── #


def test_reset():
    """reset() empties the bucket (restores full remaining capacity)."""
    limiter = LeakyBucket(rate=1.0, capacity=10)
    for _ in range(10):
        limiter.allow()
    assert limiter.remaining() == 0
    limiter.reset()
    assert limiter.remaining() == 10


# ── Peek ─────────────────────────────────────────────────────────────────── #


def test_peek_no_side_effect():
    """peek() does not change the water level."""
    limiter = LeakyBucket(rate=1.0, capacity=10)
    limiter.allow(cost=3)
    before = limiter.remaining()
    peek_result = limiter.peek()
    after = limiter.remaining()
    assert before == after, "peek should not change remaining capacity"
    assert peek_result.allowed is True


# ── reset_after ──────────────────────────────────────────────────────────── #


def test_reset_after_positive_when_filled():
    """reset_after > 0 when water is present in the bucket."""
    limiter = LeakyBucket(rate=10.0, capacity=10)
    limiter.allow(cost=5)
    ra = limiter.reset_after()
    assert ra > 0, "reset_after should be positive when bucket has water"


# ── __repr__ ─────────────────────────────────────────────────────────────── #


def test_repr():
    """Repr contains 'LeakyBucket'."""
    limiter = LeakyBucket(rate=10.0, capacity=100)
    assert "LeakyBucket" in repr(limiter)


# ── Result truthiness ────────────────────────────────────────────────────── #


def test_result_bool():
    """RateLimitResult truthiness matches allowed."""
    limiter = LeakyBucket(rate=1.0, capacity=1)
    allowed = limiter.allow()
    assert bool(allowed) is True
    denied = limiter.allow()
    assert bool(denied) is False


# ── retry_after ──────────────────────────────────────────────────────────── #


def test_retry_after_zero_when_allowed():
    """retry_after is 0.0 when the request is allowed."""
    limiter = LeakyBucket(rate=10.0, capacity=10)
    result = limiter.allow()
    assert result.retry_after == 0.0


def test_retry_after_positive_when_denied():
    """retry_after > 0 when the request is denied."""
    limiter = LeakyBucket(rate=1.0, capacity=1)
    limiter.allow()  # fill the bucket
    denied = limiter.allow()
    assert denied.retry_after > 0, "retry_after should be positive when denied"
