"""Thread safety tests.

Every limiter and the RateLimitManager must stay correct under heavy
concurrent access. Each test launches 50 threads that hammer a shared
limiter simultaneously and asserts the *exact* number admitted — an upper
bound alone would also be satisfied by a limiter that admits nothing.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, TypeVar

import pytest

from limitra import (
    FixedWindow,
    LeakyBucket,
    RateLimiter,
    RateLimitManager,
    SlidingLog,
    SlidingWindow,
    TokenBucket,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_THREADS = 50
OPS_PER_THREAD = 10
CAPACITY = 200

#: Every test here must finish well inside this; exceeding it means threads
#: are stuck, which is the failure these tests exist to catch.
DEADLINE = 30.0

ALL_ALGORITHMS = [TokenBucket, LeakyBucket, FixedWindow, SlidingWindow, SlidingLog]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_limiter(cls: type[RateLimiter]) -> RateLimiter:
    """Build any algorithm so exactly CAPACITY requests fit.

    The refill rate and window are chosen so that no capacity can come back
    while the test runs, which makes the admitted count deterministic.
    """
    if cls in (TokenBucket, LeakyBucket):
        return cls(rate=0.001, capacity=CAPACITY)
    return cls(limit=CAPACITY, window=30.0)


def hammer(
    call: Callable[[int, int], T],
    *,
    threads: int = NUM_THREADS,
    ops: int = OPS_PER_THREAD,
) -> list[T]:
    """Run ``call`` from many threads that all start at the same instant.

    Args:
        call: Invoked as ``call(thread_index, op_index)``.
        threads: Number of worker threads.
        ops: Calls made by each worker.

    Returns:
        Every value ``call`` returned, in no particular order.

    Raises:
        AssertionError: If a worker raised, or the workers did not finish
            within :data:`DEADLINE` seconds.
    """
    results: list[T] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    barrier = threading.Barrier(threads)

    def worker(index: int) -> None:
        try:
            barrier.wait()
            local = [call(index, n) for n in range(ops)]
        except BaseException as exc:  # surfaced as a test failure below
            with lock:
                errors.append(exc)
            return
        with lock:
            results.extend(local)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for thread in workers:
        thread.start()

    deadline = time.monotonic() + DEADLINE
    for thread in workers:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    stuck = [t for t in workers if t.is_alive()]
    assert not stuck, f"{len(stuck)} threads did not finish within {DEADLINE}s"
    assert not errors, f"a worker raised {errors[0]!r}"
    return results


# ---------------------------------------------------------------------------
# Individual algorithm concurrency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_admits_exactly_the_limit_under_contention(
    algorithm: type[RateLimiter],
) -> None:
    """500 concurrent requests against a limit of 200 admit exactly 200.

    Asserting the exact count is what makes this test meaningful: a limiter
    that admitted nothing at all would satisfy ``<= CAPACITY``.
    """
    limiter = make_limiter(algorithm)

    admitted = hammer(lambda _i, _n: limiter.allow().allowed)

    assert len(admitted) == NUM_THREADS * OPS_PER_THREAD
    assert sum(admitted) == CAPACITY


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_remaining_agrees_with_what_was_admitted(
    algorithm: type[RateLimiter],
) -> None:
    """After the storm, the limiter reports itself exhausted."""
    limiter = make_limiter(algorithm)

    hammer(lambda _i, _n: limiter.allow().allowed)

    assert limiter.remaining() == 0
    assert limiter.allow().allowed is False


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_cost_is_never_double_spent(algorithm: type[RateLimiter]) -> None:
    """Concurrent multi-unit requests consume exactly what they report."""
    limiter = make_limiter(algorithm)

    admitted = hammer(lambda _i, _n: limiter.allow(cost=4).allowed)

    assert sum(admitted) * 4 == CAPACITY


# ---------------------------------------------------------------------------
# Mixed operations
# ---------------------------------------------------------------------------

#: Every public operation, so the workers exercise each lock path.
MIXED_OPERATIONS: list[Callable[[RateLimiter], object]] = [
    lambda limiter: limiter.allow(),
    lambda limiter: limiter.peek(),
    lambda limiter: limiter.remaining(),
    lambda limiter: limiter.reset_after(),
    lambda limiter: limiter.reset(),
]


def test_mixed_operations_never_deadlock() -> None:
    """20 threads interleaving every public call must all finish.

    The operation each worker runs is a function of its index, so the
    interleaving is reproducible rather than random.
    """
    limiter = TokenBucket(rate=100.0, capacity=50)

    def call(index: int, n: int) -> None:
        MIXED_OPERATIONS[(index + n) % len(MIXED_OPERATIONS)](limiter)

    hammer(call, threads=20, ops=100)


def test_peek_never_consumes_under_contention() -> None:
    """Concurrent peeks leave the limiter exactly where they found it."""
    limiter = FixedWindow(limit=CAPACITY, window=30.0)
    limiter.allow(cost=50)
    before = limiter.remaining()

    hammer(lambda _i, _n: limiter.peek().allowed)

    assert limiter.remaining() == before


# ---------------------------------------------------------------------------
# RateLimitManager concurrency
# ---------------------------------------------------------------------------


def test_manager_shared_key_admits_exactly_the_limit() -> None:
    """Threads racing on one key see a single shared budget."""
    manager = RateLimitManager(TokenBucket, rate=0.001, capacity=CAPACITY)

    admitted = hammer(lambda _i, _n: manager.allow("shared-key").allowed)

    assert sum(admitted) == CAPACITY
    assert len(manager) == 1


def test_manager_distinct_keys_each_get_a_full_budget() -> None:
    """Concurrent keys do not steal from one another."""
    manager = RateLimitManager(TokenBucket, rate=0.001, capacity=OPS_PER_THREAD)

    admitted = hammer(lambda i, _n: manager.allow(f"user-{i}").allowed)

    assert sum(admitted) == NUM_THREADS * OPS_PER_THREAD, "every key had room"
    assert len(manager) == NUM_THREADS


def test_manager_stays_bounded_under_a_concurrent_key_flood() -> None:
    """max_keys holds even when every thread invents new keys at once."""
    manager = RateLimitManager(TokenBucket, max_keys=16, rate=0.001, capacity=CAPACITY)

    hammer(lambda i, n: manager.allow(f"ip-{i}-{n}").allowed)

    assert len(manager) <= 16


def test_manager_cleanup_races_with_traffic() -> None:
    """Sweeping the map while it is being written must not raise."""
    manager = RateLimitManager(TokenBucket, rate=100.0, capacity=CAPACITY)

    def call(index: int, n: int) -> None:
        if index % 5 == 0:
            manager.cleanup(max_idle=0.0)
        else:
            manager.allow(f"user-{n}")

    hammer(call, threads=20, ops=50)
