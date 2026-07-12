"""Tests for the SlidingLog rate limiter."""

from __future__ import annotations

import time

import pytest

from limitra import RateLimitResult, SlidingLog

# ------------------------------------------------------------------ #
# Initialisation
# ------------------------------------------------------------------ #


def test_init_valid() -> None:
    """SlidingLog can be created with valid parameters."""
    sl = SlidingLog(limit=10, window=1.0)
    assert sl._limit == 10
    assert sl._window == 1.0


def test_init_invalid_limit() -> None:
    """Limit < 1 raises ValueError."""
    with pytest.raises(ValueError, match="limit must be >= 1"):
        SlidingLog(limit=0, window=1.0)
    with pytest.raises(ValueError, match="limit must be >= 1"):
        SlidingLog(limit=-3, window=1.0)


def test_init_invalid_window() -> None:
    """Window <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="window must be > 0"):
        SlidingLog(limit=10, window=0)
    with pytest.raises(ValueError, match="window must be > 0"):
        SlidingLog(limit=10, window=-1.0)


# ------------------------------------------------------------------ #
# allow() basics
# ------------------------------------------------------------------ #


def test_allow_single() -> None:
    """A single request in a fresh window is allowed."""
    sl = SlidingLog(limit=5, window=1.0)
    result = sl.allow()
    assert result.allowed is True


def test_allow_returns_result() -> None:
    """allow() returns a RateLimitResult with all expected fields."""
    sl = SlidingLog(limit=10, window=1.0)
    result = sl.allow()

    assert isinstance(result, RateLimitResult)
    assert result.allowed is True
    assert isinstance(result.remaining, int)
    assert result.limit == 10
    assert isinstance(result.reset_after, float)
    assert result.retry_after == 0.0


def test_exhaust_limit() -> None:
    """After exhausting the limit, further requests are denied."""
    sl = SlidingLog(limit=3, window=10.0)
    for _ in range(3):
        result = sl.allow()
        assert result.allowed is True

    denied = sl.allow()
    assert denied.allowed is False
    assert denied.remaining == 0
    assert denied.retry_after > 0.0


def test_remaining_decreases() -> None:
    """Remaining goes down with each allowed request."""
    sl = SlidingLog(limit=5, window=10.0)
    assert sl.remaining() == 5

    for i in range(5):
        result = sl.allow()
        expected_remaining = 5 - (i + 1)
        assert result.remaining == expected_remaining

    assert sl.remaining() == 0


# ------------------------------------------------------------------ #
# Window expiry
# ------------------------------------------------------------------ #


def test_window_expiry() -> None:
    """After sleeping past the window, old entries expire and capacity is restored."""
    sl = SlidingLog(limit=3, window=0.1)

    # Exhaust the limit
    for _ in range(3):
        sl.allow()
    assert sl.remaining() == 0

    # Sleep past the window
    time.sleep(0.15)

    # All old timestamps should have expired
    assert sl.remaining() == 3
    result = sl.allow()
    assert result.allowed is True


# ------------------------------------------------------------------ #
# Cost parameter
# ------------------------------------------------------------------ #


def test_cost_parameter() -> None:
    """allow(cost=2) adds 2 entries to the log."""
    sl = SlidingLog(limit=5, window=10.0)
    result = sl.allow(cost=2)
    assert result.allowed is True
    assert result.remaining == 3  # 5 - 2

    # Verify the log has 2 entries
    assert len(sl._log) == 2

    # Use cost=2 again (total 4), then cost=2 should be denied (would be 6 > 5)
    sl.allow(cost=2)
    denied = sl.allow(cost=2)
    assert denied.allowed is False


def test_cost_validation() -> None:
    """Invalid cost values raise appropriate errors."""
    sl = SlidingLog(limit=5, window=1.0)

    with pytest.raises(ValueError, match="cost must be >= 1"):
        sl.allow(cost=0)

    with pytest.raises(ValueError, match="cost must be >= 1"):
        sl.allow(cost=-1)

    with pytest.raises(TypeError, match="cost must be an integer"):
        sl.allow(cost=2.5)  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# Reset
# ------------------------------------------------------------------ #


def test_reset() -> None:
    """reset() clears the log and restores full capacity."""
    sl = SlidingLog(limit=3, window=10.0)

    # Exhaust the limit
    for _ in range(3):
        sl.allow()
    assert sl.remaining() == 0

    sl.reset()
    assert sl.remaining() == 3
    assert len(sl._log) == 0

    result = sl.allow()
    assert result.allowed is True


# ------------------------------------------------------------------ #
# Peek
# ------------------------------------------------------------------ #


def test_peek_no_side_effect() -> None:
    """peek() does not add timestamps to the log."""
    sl = SlidingLog(limit=3, window=10.0)

    # Consume 2 of 3
    sl.allow()
    sl.allow()
    log_len_before = len(sl._log)
    remaining_before = sl.remaining()

    # Peek should report status without modifying the log
    peek_result = sl.peek()
    assert peek_result.allowed is True
    assert len(sl._log) == log_len_before
    assert sl.remaining() == remaining_before

    # Multiple peeks should not change state
    for _ in range(10):
        sl.peek()
    assert len(sl._log) == log_len_before
    assert sl.remaining() == remaining_before


# ------------------------------------------------------------------ #
# Repr
# ------------------------------------------------------------------ #


def test_repr() -> None:
    """Repr contains 'SlidingLog'."""
    sl = SlidingLog(limit=5, window=10.0)
    r = repr(sl)
    assert "SlidingLog" in r
    assert "5" in r
    assert "10.0" in r


# ------------------------------------------------------------------ #
# Retry-after accuracy
# ------------------------------------------------------------------ #


def test_retry_after_accuracy() -> None:
    """When denied, retry_after indicates when oldest entries expire."""
    sl = SlidingLog(limit=2, window=0.2)

    # Fill the log
    sl.allow()
    sl.allow()

    # Next request is denied
    denied = sl.allow()
    assert denied.allowed is False

    # retry_after should be > 0 and <= window
    assert 0.0 < denied.retry_after <= 0.2

    # Sleep past retry_after plus a small margin
    time.sleep(denied.retry_after + 0.05)

    # Now a request should be allowed (oldest entry expired)
    result = sl.allow()
    assert result.allowed is True
