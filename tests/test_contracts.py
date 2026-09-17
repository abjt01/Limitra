"""Cross-algorithm contract tests.

Everything the ``RateLimiter`` base class promises, checked against every
implementation on a frozen clock so the timing maths is exact rather than
approximate.
"""

from __future__ import annotations

import math

import pytest

from limitra import (
    FixedWindow,
    LeakyBucket,
    RateLimiter,
    SlidingLog,
    SlidingWindow,
    TokenBucket,
)
from limitra._base import _satisfiable

from .conftest import FakeClock

ALL_ALGORITHMS = [TokenBucket, LeakyBucket, FixedWindow, SlidingWindow, SlidingLog]
BUCKETS = [TokenBucket, LeakyBucket]
WINDOWS = [FixedWindow, SlidingWindow, SlidingLog]

PERIOD = 10.0
LIMIT = 10


def make_limiter(cls: type[RateLimiter], limit: int = LIMIT) -> RateLimiter:
    """Build any algorithm so ``limit`` units fit per ``PERIOD`` seconds."""
    if cls in BUCKETS:
        return cls(rate=limit / PERIOD, capacity=limit)
    return cls(limit=limit, window=PERIOD)


# ---------------------------------------------------------------------------
# Construction is validated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", BUCKETS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_rate_must_be_finite_and_positive(
    algorithm: type[RateLimiter], bad: float
) -> None:
    """A NaN rate used to leave the bucket permanently open.

    ``rate <= 0`` is ``False`` for NaN, so the guard let it through and the
    limiter silently admitted everything while reporting itself healthy —
    the worst possible failure direction for a rate limiter.
    """
    with pytest.raises(ValueError, match="rate must be > 0"):
        algorithm(rate=bad, capacity=10)


@pytest.mark.parametrize("algorithm", WINDOWS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_window_must_be_finite_and_positive(
    algorithm: type[RateLimiter], bad: float
) -> None:
    """A non-finite window makes every reported time meaningless."""
    with pytest.raises(ValueError, match="window must be > 0"):
        algorithm(limit=10, window=bad)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_size_must_be_a_whole_number(algorithm: type[RateLimiter]) -> None:
    """A float capacity would surface as ``X-RateLimit-Limit: 10.5``."""
    name = "capacity" if algorithm in BUCKETS else "limit"
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        if algorithm in BUCKETS:
            algorithm(rate=1.0, capacity=10.5)
        else:
            algorithm(limit=10.5, window=PERIOD)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_rejecting_a_bad_config_does_not_produce_a_usable_limiter(
    algorithm: type[RateLimiter],
) -> None:
    """A misconfigured limiter never exists, so it can never fail open."""
    with pytest.raises((TypeError, ValueError)):
        if algorithm in BUCKETS:
            algorithm(rate=float("nan"), capacity=10)
        else:
            algorithm(limit=10, window=float("nan"))


# ---------------------------------------------------------------------------
# reset_after
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_reset_after_is_zero_when_untouched(algorithm: type[RateLimiter]) -> None:
    """An idle limiter is already at full capacity, so nothing to wait for."""
    assert make_limiter(algorithm).reset_after() == 0.0


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_reset_after_is_positive_once_something_is_spent(
    algorithm: type[RateLimiter],
) -> None:
    """Spending anything means capacity is outstanding."""
    limiter = make_limiter(algorithm)
    limiter.allow()
    assert limiter.reset_after() > 0.0


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_reset_after_actually_restores_full_capacity(
    algorithm: type[RateLimiter], clock: FakeClock
) -> None:
    """Wait out reset_after and the limiter really is back to full.

    This is the contract the base class states, and it is what makes
    ``X-RateLimit-Reset`` mean something to a client.
    """
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT):
        assert limiter.allow().allowed is True
    assert limiter.remaining() == 0

    clock.advance(limiter.reset_after())

    assert limiter.remaining() == LIMIT
    assert limiter.reset_after() == 0.0


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_reset_after_is_zero_again_after_reset(
    algorithm: type[RateLimiter],
) -> None:
    """reset() puts the limiter back to its freshly built state."""
    limiter = make_limiter(algorithm)
    limiter.allow(cost=LIMIT)
    limiter.reset()

    assert limiter.remaining() == LIMIT
    assert limiter.reset_after() == 0.0


# ---------------------------------------------------------------------------
# retry_after
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_retry_after_is_zero_when_allowed(algorithm: type[RateLimiter]) -> None:
    """An allowed request has nothing to retry."""
    limiter = make_limiter(algorithm)
    assert limiter.allow().retry_after == 0.0
    assert limiter.peek().retry_after == 0.0


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
@pytest.mark.parametrize("cost", [1, 3, LIMIT])
def test_waiting_exactly_retry_after_succeeds(
    algorithm: type[RateLimiter], cost: int, clock: FakeClock
) -> None:
    """Obey retry_after to the microsecond and the retry must get through.

    A retry_after that is even slightly short means a well-behaved client
    is denied twice, which is how SlidingWindow used to give out roughly
    half the configured rate.
    """
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT):
        limiter.allow()

    denied = limiter.allow(cost)
    assert denied.allowed is False
    assert denied.retry_after > 0.0

    clock.advance(denied.retry_after)

    assert limiter.allow(cost).allowed is True, (
        f"{algorithm.__name__} advertised {denied.retry_after}s but was still "
        f"denied after waiting exactly that long"
    )


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_retry_after_is_not_wildly_pessimistic(
    algorithm: type[RateLimiter], clock: FakeClock
) -> None:
    """Retrying a hair earlier than advertised must genuinely be too early.

    Together with the test above this pins retry_after to the real
    boundary, rather than to some conservative over-estimate.
    """
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT):
        limiter.allow()

    denied = limiter.allow()
    clock.advance(denied.retry_after * 0.5)

    assert limiter.peek().allowed is False


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_retry_after_never_exceeds_reset_after(
    algorithm: type[RateLimiter],
) -> None:
    """Coming back for one slot cannot take longer than a full reset.

    Emitting ``Retry-After`` larger than ``X-RateLimit-Reset`` would be a
    contradiction in the same response.
    """
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT):
        limiter.allow()

    denied = limiter.allow()
    assert denied.retry_after <= denied.reset_after + 1e-9


# ---------------------------------------------------------------------------
# remaining
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_remaining_counts_down_to_zero_exactly(
    algorithm: type[RateLimiter],
) -> None:
    """remaining() tracks what allow() will actually grant."""
    limiter = make_limiter(algorithm)
    for expected in range(LIMIT - 1, -1, -1):
        assert limiter.allow().remaining == expected
        assert limiter.remaining() == expected


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_remaining_predicts_how_many_more_will_be_admitted(
    algorithm: type[RateLimiter],
) -> None:
    """Whatever remaining() says is left, that many requests get through."""
    limiter = make_limiter(algorithm)
    limiter.allow(cost=4)

    predicted = limiter.remaining()
    admitted = sum(1 for _ in range(LIMIT * 2) if limiter.allow().allowed)

    assert admitted == predicted


# ---------------------------------------------------------------------------
# Headers stay consistent with the result they came from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_headers_never_under_report_the_wait(
    algorithm: type[RateLimiter],
) -> None:
    """Rounded headers must never send a client back early."""
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT):
        limiter.allow()
    denied = limiter.allow()

    headers = denied.as_headers()
    assert int(headers["Retry-After"]) >= denied.retry_after
    assert int(headers["X-RateLimit-Reset"]) >= denied.reset_after
    assert int(headers["X-RateLimit-Remaining"]) == denied.remaining


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_reported_times_are_finite(algorithm: type[RateLimiter]) -> None:
    """No result ever carries NaN or infinity."""
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT + 3):
        result = limiter.allow()
        assert math.isfinite(result.reset_after)
        assert math.isfinite(result.retry_after)


# ---------------------------------------------------------------------------
# Argument types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", BUCKETS)
def test_rate_must_be_a_number(algorithm: type[RateLimiter]) -> None:
    """A string rate is a config mistake, not a value to coerce."""
    with pytest.raises(TypeError, match="rate must be a number"):
        algorithm(rate="fast", capacity=10)  # type: ignore[arg-type]


@pytest.mark.parametrize("algorithm", WINDOWS)
def test_window_must_be_a_number(algorithm: type[RateLimiter]) -> None:
    """A string window is a config mistake, not a value to coerce."""
    with pytest.raises(TypeError, match="window must be a number"):
        algorithm(limit=10, window="1m")  # type: ignore[arg-type]


@pytest.mark.parametrize("algorithm", BUCKETS)
def test_integer_rate_is_accepted(algorithm: type[RateLimiter]) -> None:
    """``rate=10`` is as valid as ``rate=10.0``."""
    limiter = algorithm(rate=10, capacity=10)
    assert limiter.rate == 10.0


@pytest.mark.parametrize("algorithm", WINDOWS)
def test_integer_window_is_accepted(algorithm: type[RateLimiter]) -> None:
    """``window=60`` is as valid as ``window=60.0``."""
    limiter = algorithm(limit=10, window=60)
    assert limiter.window == 60.0


# ---------------------------------------------------------------------------
# The retry nudge
# ---------------------------------------------------------------------------


def test_satisfiable_nudges_a_wait_past_the_boundary() -> None:
    """retry_after must land just after the moment capacity returns.

    It is derived by inverting the same float arithmetic the limiter redoes
    on the next call, so the exact answer can be one ULP short and deny a
    caller who waited precisely as told.
    """
    assert _satisfiable(1.0) > 1.0
    assert _satisfiable(1.0) == math.nextafter(1.0, math.inf)


@pytest.mark.parametrize("value", [0.0, -0.0, -1.0])
def test_satisfiable_never_returns_a_negative_wait(value: float) -> None:
    """Nothing is ever asked to wait a negative amount of time."""
    assert _satisfiable(value) == 0.0
