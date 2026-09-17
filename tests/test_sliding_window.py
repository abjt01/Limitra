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


def test_previous_window_is_dropped_after_a_long_idle_gap(clock) -> None:
    """Skipping two or more windows clears the carried-over count entirely.

    Only the immediately preceding window contributes to the weighted
    count, so a limiter that sat idle must come back at full capacity
    rather than dragging a stale counter forward.
    """
    limiter = SlidingWindow(limit=5, window=10.0)
    for _ in range(5):
        limiter.allow()
    assert limiter.allow().allowed is False

    clock.advance(25.0)  # more than two whole windows

    assert limiter.remaining() == 5, "a stale window must not be carried forward"
    for _ in range(5):
        assert limiter.allow().allowed is True


def test_previous_window_is_weighted_after_a_single_window_gap(clock) -> None:
    """Crossing exactly one boundary carries the previous count, decayed.

    This is what separates a sliding window from a fixed one: capacity
    returns gradually across the boundary instead of all at once.
    """
    limiter = SlidingWindow(limit=10, window=10.0)
    for _ in range(10):
        limiter.allow()

    # A quarter past the boundary: 75% of the previous window still counts,
    # and remaining rounds usage up, so 10 - ceil(7.5).
    clock.advance(12.5)
    assert limiter.remaining() == 2

    # Halfway through: half of it counts.
    clock.advance(2.5)
    assert limiter.remaining() == 5

    # And at the next boundary it is gone entirely.
    clock.advance(5.0)
    assert limiter.remaining() == 10


# ---------------------------------------------------------------------------
# Denials raised after the window has rotated
# ---------------------------------------------------------------------------


def test_retry_after_solves_the_decay_within_the_current_window(clock) -> None:
    """A denial carried by the *previous* window resolves mid-window.

    The previous window's contribution shrinks continuously, so the moment
    one more request fits is generally well before the next boundary.
    Returning the boundary instead is what used to halve the throughput.
    """
    limiter = SlidingWindow(limit=10, window=10.0)
    for _ in range(10):
        limiter.allow()

    clock.advance(10.0)  # rotate: the 10 requests become the previous window

    denied = limiter.allow()
    assert denied.allowed is False
    # Weighted count is 10; one unit frees up once it decays to 9, a tenth
    # of the way into this window.
    assert denied.retry_after == pytest.approx(1.0, abs=0.01)
    assert denied.retry_after < limiter.window

    clock.advance(denied.retry_after)
    assert limiter.allow().allowed is True


def test_retry_after_waits_for_the_boundary_when_this_window_is_full(
    clock,
) -> None:
    """When the current window alone blocks the request, wait for rotation."""
    limiter = SlidingWindow(limit=10, window=10.0)
    limiter.allow(cost=5)

    clock.advance(15.0)  # rotate, then halfway in: previous weighs 2.5
    assert limiter.allow(cost=7).allowed is True

    denied = limiter.allow(cost=3)
    assert denied.allowed is False
    assert denied.retry_after == pytest.approx(5.0, abs=0.01), "the boundary"

    clock.advance(denied.retry_after)
    assert limiter.allow(cost=3).allowed is True


def test_reset_after_spans_the_previous_window_only(clock) -> None:
    """With nothing spent this window, a full reset is one window away."""
    limiter = SlidingWindow(limit=10, window=10.0)
    limiter.allow(cost=4)

    clock.advance(10.0)  # rotate: prev=4, curr=0

    assert limiter.reset_after() == pytest.approx(10.0, abs=0.01)

    clock.advance(limiter.reset_after())
    assert limiter.remaining() == 10
    assert limiter.reset_after() == 0.0


def test_reset_after_spans_two_windows_while_this_one_is_in_use(clock) -> None:
    """A request made now holds capacity until it has decayed as previous."""
    limiter = SlidingWindow(limit=10, window=10.0)
    limiter.allow()

    assert limiter.reset_after() == pytest.approx(20.0, abs=0.01)
