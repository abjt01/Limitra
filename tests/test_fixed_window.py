"""Comprehensive tests for the FixedWindow rate limiter."""

import time

import pytest

from limitra import FixedWindow, RateLimitResult

# ── Initialization ──────────────────────────────────────────────────────── #


def test_init_valid():
    """FixedWindow can be created with valid limit and window."""
    limiter = FixedWindow(limit=100, window=60.0)
    assert limiter is not None


def test_init_invalid_limit():
    """Limit < 1 raises ValueError."""
    with pytest.raises(ValueError, match="limit must be >= 1"):
        FixedWindow(limit=0, window=60.0)
    with pytest.raises(ValueError, match="limit must be >= 1"):
        FixedWindow(limit=-1, window=60.0)


def test_init_invalid_window():
    """Window <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="window must be > 0"):
        FixedWindow(limit=10, window=0)
    with pytest.raises(ValueError, match="window must be > 0"):
        FixedWindow(limit=10, window=-1.0)


# ── Basic allow / deny ──────────────────────────────────────────────────── #


def test_allow_single():
    """A fresh window allows a single request."""
    limiter = FixedWindow(limit=10, window=60.0)
    result = limiter.allow()
    assert result.allowed is True


def test_allow_returns_result():
    """allow() returns a RateLimitResult with all expected fields."""
    limiter = FixedWindow(limit=10, window=60.0)
    result = limiter.allow()
    assert isinstance(result, RateLimitResult)
    assert hasattr(result, "allowed")
    assert hasattr(result, "remaining")
    assert hasattr(result, "limit")
    assert hasattr(result, "reset_after")
    assert hasattr(result, "retry_after")


def test_exhaust_limit():
    """Allowing limit times exhausts the window, then denies."""
    limit = 5
    limiter = FixedWindow(limit=limit, window=60.0)
    for _ in range(limit):
        result = limiter.allow()
        assert result.allowed is True, "should be allowed while under limit"
    denied = limiter.allow()
    assert denied.allowed is False, "should be denied after reaching limit"


# ── Remaining ────────────────────────────────────────────────────────────── #


def test_remaining_decreases():
    """Remaining goes down with each allow call."""
    limiter = FixedWindow(limit=5, window=60.0)
    assert limiter.remaining() == 5
    limiter.allow()
    assert limiter.remaining() == 4
    limiter.allow()
    assert limiter.remaining() == 3


# ── Window reset ─────────────────────────────────────────────────────────── #


def test_window_reset():
    """After sleeping past the window duration, the counter resets."""
    limiter = FixedWindow(limit=5, window=0.1)
    # Exhaust the limit
    for _ in range(5):
        limiter.allow()
    assert limiter.remaining() == 0, "should be exhausted"
    time.sleep(0.15)
    result = limiter.allow()
    assert result.allowed is True, "should be allowed after window reset"
    assert limiter.remaining() == 4, "should have 4 remaining after one use"


# ── Cost parameter ──────────────────────────────────────────────────────── #


def test_cost_parameter():
    """allow(cost=3) consumes 3 of the limit at once."""
    limiter = FixedWindow(limit=10, window=60.0)
    result = limiter.allow(cost=3)
    assert result.allowed is True
    assert limiter.remaining() == 7


def test_cost_validation():
    """Invalid costs raise appropriate errors."""
    limiter = FixedWindow(limit=10, window=60.0)
    with pytest.raises(ValueError):
        limiter.allow(cost=0)
    with pytest.raises(ValueError):
        limiter.allow(cost=-1)
    with pytest.raises(TypeError):
        limiter.allow(cost="a")  # type: ignore[arg-type]


# ── Reset ────────────────────────────────────────────────────────────────── #


def test_reset():
    """reset() clears the counter and starts a fresh window."""
    limiter = FixedWindow(limit=5, window=60.0)
    for _ in range(5):
        limiter.allow()
    assert limiter.remaining() == 0
    limiter.reset()
    assert limiter.remaining() == 5


# ── Peek ─────────────────────────────────────────────────────────────────── #


def test_peek_no_side_effect():
    """peek() does not increment the counter."""
    limiter = FixedWindow(limit=10, window=60.0)
    limiter.allow(cost=3)
    before = limiter.remaining()
    peek_result = limiter.peek()
    after = limiter.remaining()
    assert before == after, "peek should not change remaining count"
    assert peek_result.allowed is True


# ── reset_after ──────────────────────────────────────────────────────────── #


def test_reset_after_positive():
    """reset_after > 0 within an active window."""
    limiter = FixedWindow(limit=10, window=60.0)
    limiter.allow()
    ra = limiter.reset_after()
    assert ra > 0, "reset_after should be positive within a window"


# ── __repr__ ─────────────────────────────────────────────────────────────── #


def test_repr():
    """Repr contains 'FixedWindow'."""
    limiter = FixedWindow(limit=10, window=60.0)
    assert "FixedWindow" in repr(limiter)


# ── retry_after == reset_after when denied ────────────────────────────────── #


def test_retry_after_equals_reset_after_when_denied():
    """For FixedWindow, retry_after equals reset_after when denied."""
    limiter = FixedWindow(limit=1, window=60.0)
    limiter.allow()  # exhaust
    denied = limiter.allow()
    assert denied.allowed is False
    assert denied.retry_after == denied.reset_after, (
        "for fixed window, retry_after should equal reset_after when denied"
    )
    assert denied.retry_after > 0
