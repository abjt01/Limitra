"""Cross-algorithm contract tests.

Everything the ``RateLimiter`` base class promises, checked against every
implementation on a frozen clock so the timing maths is exact rather than
approximate.
"""

from __future__ import annotations

import copy
import math
import pickle
import weakref

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
    assert int(headers["RateLimit-Reset"]) >= denied.reset_after
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


def test_satisfiable_clears_the_boundary_at_realistic_clock_values() -> None:
    """The nudge must scale to the clock the caller lands on.

    One ULP of a short duration is far smaller than one ULP of a monotonic
    reading taken days into a machine's uptime, so nudging the duration
    left the caller landing fractionally short.
    """
    for now in (0.0, 1e3, 1e6, 1e9):
        for seconds in (1e-3, 0.25, 60.0):
            wait = _satisfiable(seconds, now)
            assert wait >= seconds
            assert now + wait > now + seconds, f"nudge vanished at now={now}"


@pytest.mark.parametrize("value", [0.0, -0.0, -1.0])
def test_satisfiable_never_advertises_an_immediate_retry(value: float) -> None:
    """A denial saying "retry in 0.0s" sends an obedient client into a spin."""
    assert _satisfiable(value, 1e6) > 0.0


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_every_denial_advertises_a_positive_wait(
    algorithm: type[RateLimiter], clock: FakeClock
) -> None:
    """``allowed is False`` must never come with ``retry_after == 0.0``.

    A caller looping on ``sleep(result.retry_after)`` would otherwise burn
    a core, and the HTTP header would read ``Retry-After: 0``.
    """
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT):
        limiter.allow()

    for step in (0.0, PERIOD / 3, PERIOD / 2, PERIOD * 0.9):
        clock.advance(step)
        for cost in (1, LIMIT):
            for result in (limiter.peek(cost), limiter.allow(cost)):
                if not result.allowed:
                    assert result.retry_after > 0.0, (
                        f"{algorithm.__name__} denied cost={cost} but said "
                        f"retry_after={result.retry_after}"
                    )


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_a_request_that_exactly_fits_is_admitted(
    algorithm: type[RateLimiter], clock: FakeClock
) -> None:
    """Float cancellation must not refuse a request that exactly fits."""
    limiter = make_limiter(algorithm)
    limiter.allow(cost=LIMIT // 2)

    free = limiter.remaining()
    assert free > 0
    assert limiter.allow(cost=free).allowed is True, "an exact fit was refused"
    assert limiter.remaining() == 0


def test_sliding_window_admits_an_exact_fit_against_a_decayed_window(
    clock: FakeClock,
) -> None:
    """The decayed count is where cancellation bites hardest.

    Combining two counters across a decaying overlap cancels digits, so a
    weighted count that is mathematically a whole number can come out a few
    ULPs high — refusing a request that fits exactly, and then advertising
    a zero-second wait for it.
    """
    for spent in range(1, LIMIT + 1):
        limiter = SlidingWindow(limit=LIMIT, window=PERIOD)
        limiter.allow(cost=spent)
        clock.advance(PERIOD * 1.5)  # half the previous window still counts

        free = limiter.remaining()
        result = limiter.allow(cost=free)
        assert result.allowed is True, f"exact fit of {free} refused after {spent}"
        assert result.retry_after == 0.0


# ---------------------------------------------------------------------------
# Copying a subclass
# ---------------------------------------------------------------------------


class _TaggedBucket(TokenBucket):
    """A subclass with an instance dict, the ordinary way people extend."""

    def __init__(self, tag: str, **kwargs: object) -> None:
        """Store an arbitrary attribute alongside the limiter state."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.tag = tag


def test_copying_keeps_subclass_attributes() -> None:
    """A subclass without __slots__ has a __dict__, which must be carried.

    Walking __slots__ alone silently dropped everything a subclass stored,
    so the copy came back missing attributes rather than failing loudly.
    """
    original = _TaggedBucket("primary", rate=1.0, capacity=10)
    original.allow(cost=3)

    for clone in (
        copy.copy(original),
        copy.deepcopy(original),
        pickle.loads(pickle.dumps(original)),
    ):
        assert clone.tag == "primary"
        assert clone.remaining() == 7
        assert clone._lock is not original._lock


# ---------------------------------------------------------------------------
# refund
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_refund_returns_spent_capacity(algorithm: type[RateLimiter]) -> None:
    """What refund() gives back, the next allow() can spend."""
    limiter = make_limiter(algorithm)
    limiter.allow(cost=4)
    assert limiter.remaining() == LIMIT - 4

    limiter.refund(4)

    assert limiter.remaining() == LIMIT


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_refund_unblocks_an_exhausted_limiter(
    algorithm: type[RateLimiter],
) -> None:
    """Crediting back one unit admits exactly one more request."""
    limiter = make_limiter(algorithm)
    for _ in range(LIMIT):
        limiter.allow()
    assert limiter.allow().allowed is False

    limiter.refund()

    assert limiter.allow().allowed is True
    assert limiter.allow().allowed is False, "only the refunded unit came back"


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_refund_is_clamped_not_an_error(algorithm: type[RateLimiter]) -> None:
    """Over-refunding cannot push a limiter past full capacity.

    A caller cannot easily know how much is still outstanding, so handing
    back too much is clamped rather than rejected.
    """
    limiter = make_limiter(algorithm)
    limiter.allow()

    limiter.refund(LIMIT)

    assert limiter.remaining() == LIMIT
    assert sum(1 for _ in range(LIMIT * 2) if limiter.allow().allowed) == LIMIT


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_refund_validates_cost(algorithm: type[RateLimiter]) -> None:
    """refund() applies the same cost rules as allow()."""
    limiter = make_limiter(algorithm)
    with pytest.raises(ValueError, match="cost must be >= 1"):
        limiter.refund(0)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        limiter.refund(LIMIT + 1)
    with pytest.raises(TypeError):
        limiter.refund(True)  # type: ignore[arg-type]


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_refund_on_an_untouched_limiter_is_a_no_op(
    algorithm: type[RateLimiter],
) -> None:
    """Crediting a limiter that owes nothing changes nothing."""
    limiter = make_limiter(algorithm)
    limiter.refund(LIMIT)
    assert limiter.remaining() == LIMIT
    assert limiter.reset_after() == 0.0


def test_refund_makes_two_tier_limits_composable() -> None:
    """The case refund() exists for: a burst tier and a sustained tier.

    Without it the burst tier is charged for requests the sustained tier
    rejected, so a request that was never served still erodes the budget.
    """
    burst = TokenBucket(rate=10.0, capacity=10)
    sustained = TokenBucket(rate=1.0, capacity=3)

    served = 0
    for _ in range(6):
        if not burst.allow().allowed:
            continue
        if sustained.allow().allowed:
            served += 1
        else:
            burst.refund()

    assert served == 3
    assert burst.remaining() == 10 - served, "burst tier charged only for served"


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_copy_gets_its_own_lock_and_state(algorithm: type[RateLimiter]) -> None:
    """A copy must be independent, not a second handle on one lock.

    Slot-copying used to carry the ``threading.Lock`` across, leaving two
    limiters with separate budgets guarded by a single lock.
    """
    original = make_limiter(algorithm)
    original.allow(cost=4)

    duplicate = copy.copy(original)

    assert duplicate._lock is not original._lock
    assert duplicate.remaining() == original.remaining()

    duplicate.allow(cost=LIMIT - 4)
    assert duplicate.remaining() == 0
    assert original.remaining() == LIMIT - 4, "the copy must not spend the original"


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_deepcopy_and_pickle_round_trip(algorithm: type[RateLimiter]) -> None:
    """Both used to fail with an opaque ``cannot pickle '_thread.lock'``."""
    original = make_limiter(algorithm)
    original.allow(cost=3)

    for clone in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert clone.remaining() == LIMIT - 3
        assert clone.limit == original.limit
        assert clone._lock is not original._lock
        assert clone.allow().allowed is True


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_copies_are_weak_referenceable(algorithm: type[RateLimiter]) -> None:
    """Limiters can live in a WeakValueDictionary."""
    limiter = make_limiter(algorithm)
    assert weakref.ref(limiter)() is limiter
