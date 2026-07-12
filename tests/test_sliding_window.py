"""Tests for the SlidingWindow rate limiter."""

from __future__ import annotations

import time

import pytest

from limitra import RateLimitResult, SlidingWindow

# ------------------------------------------------------------------ #
# Initialisation
# ------------------------------------------------------------------ #


def test_init_valid() -> None:
    """SlidingWindow can be created with valid parameters."""
    sw = SlidingWindow(limit=10, window=1.0)
    assert sw._limit == 10
    assert sw._window == 1.0


def test_init_invalid_limit() -> None:
    """Limit < 1 raises ValueError."""
    with pytest.raises(ValueError, match="limit must be >= 1"):
        SlidingWindow(limit=0, window=1.0)
    with pytest.raises(ValueError, match="limit must be >= 1"):
        SlidingWindow(limit=-5, window=1.0)


def test_init_invalid_window() -> None:
    """Window <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="window must be > 0"):
        SlidingWindow(limit=10, window=0)
    with pytest.raises(ValueError, match="window must be > 0"):
        SlidingWindow(limit=10, window=-1.0)


# ------------------------------------------------------------------ #
# allow() basics
# ------------------------------------------------------------------ #


def test_allow_single() -> None:
    """A single request in a fresh window is allowed."""
    sw = SlidingWindow(limit=5, window=1.0)
    result = sw.allow()
    assert result.allowed is True


def test_allow_returns_result() -> None:
    """allow() returns a RateLimitResult with all expected fields."""
    sw = SlidingWindow(limit=10, window=1.0)
    result = sw.allow()

    assert isinstance(result, RateLimitResult)
    assert result.allowed is True
    assert isinstance(result.remaining, int)
    assert result.limit == 10
    assert isinstance(result.reset_after, float)
    assert result.retry_after == 0.0


def test_exhaust_limit() -> None:
    """After exhausting the limit, further requests are denied."""
    sw = SlidingWindow(limit=3, window=10.0)
    for _ in range(3):
        result = sw.allow()
        assert result.allowed is True

    denied = sw.allow()
    assert denied.allowed is False
    assert denied.remaining == 0
    assert denied.retry_after > 0.0


def test_remaining_decreases() -> None:
    """Remaining goes down with each allowed request."""
    sw = SlidingWindow(limit=5, window=10.0)
    previous_remaining = sw.remaining()
    assert previous_remaining == 5

    for i in range(5):
        result = sw.allow()
        assert result.remaining < previous_remaining or i == 4
        previous_remaining = result.remaining

    assert sw.remaining() == 0


# ------------------------------------------------------------------ #
# Window rotation
# ------------------------------------------------------------------ #


def test_window_rotation() -> None:
    """After sleeping past the window, the counter rotates and capacity is restored."""
    sw = SlidingWindow(limit=5, window=0.1)

    # Exhaust the limit
    for _ in range(5):
        sw.allow()
    assert sw.remaining() == 0

    # Sleep past the window
    time.sleep(0.15)

    # Capacity should be restored (previous counter may still carry weight,
    # but after 1.5x the window, the previous window itself is the one we
    # just exhausted, and the overlap ratio is small).
    result = sw.allow()
    assert result.allowed is True


# ------------------------------------------------------------------ #
# Weighted count
# ------------------------------------------------------------------ #


def test_weighted_count() -> None:
    """Requests from the previous window carry weight into the current window."""
    sw = SlidingWindow(limit=5, window=0.1)

    # Use 4 of 5 in the first window
    for _ in range(4):
        sw.allow()

    # Sleep just past the window boundary so counters rotate
    time.sleep(0.11)

    # Now in a new window: prev_counter=4, curr_counter=0
    # The overlap ratio is close to 1.0 right at the boundary, so weighted
    # count ≈ 4 * overlap + 0. We should still have some capacity but not full 5.
    remaining = sw.remaining()
    assert remaining < 5  # Previous window's weight reduces capacity
    assert remaining >= 0


# ------------------------------------------------------------------ #
# Cost parameter
# ------------------------------------------------------------------ #


def test_cost_parameter() -> None:
    """allow(cost=3) consumes 3 units at once."""
    sw = SlidingWindow(limit=5, window=10.0)
    result = sw.allow(cost=3)
    assert result.allowed is True
    assert result.remaining == 2

    # Only 2 remaining, cost=3 should be denied
    denied = sw.allow(cost=3)
    assert denied.allowed is False


def test_cost_validation() -> None:
    """Invalid cost values raise appropriate errors."""
    sw = SlidingWindow(limit=5, window=1.0)

    with pytest.raises(ValueError, match="cost must be >= 1"):
        sw.allow(cost=0)

    with pytest.raises(ValueError, match="cost must be >= 1"):
        sw.allow(cost=-1)

    with pytest.raises(TypeError, match="cost must be an integer"):
        sw.allow(cost=1.5)  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# Reset
# ------------------------------------------------------------------ #


def test_reset() -> None:
    """reset() clears both counters and restores full capacity."""
    sw = SlidingWindow(limit=5, window=10.0)

    # Exhaust the limit
    for _ in range(5):
        sw.allow()
    assert sw.remaining() == 0

    sw.reset()
    assert sw.remaining() == 5

    result = sw.allow()
    assert result.allowed is True


# ------------------------------------------------------------------ #
# Peek
# ------------------------------------------------------------------ #


def test_peek_no_side_effect() -> None:
    """peek() does not modify the counters."""
    sw = SlidingWindow(limit=3, window=10.0)

    # Consume 2 of 3
    sw.allow()
    sw.allow()
    remaining_before = sw.remaining()

    # Peek should report status without changing state
    peek_result = sw.peek()
    assert peek_result.allowed is True
    assert sw.remaining() == remaining_before

    # Multiple peeks should not change state
    for _ in range(10):
        sw.peek()
    assert sw.remaining() == remaining_before


# ------------------------------------------------------------------ #
# Repr
# ------------------------------------------------------------------ #


def test_repr() -> None:
    """Repr contains 'SlidingWindow'."""
    sw = SlidingWindow(limit=10, window=60.0)
    r = repr(sw)
    assert "SlidingWindow" in r
    assert "10" in r
    assert "60.0" in r
