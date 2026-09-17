"""Tests for the blocking RateLimiter.wait() helper.

``wait()`` is the client-side counterpart to ``allow()``: instead of
returning a denial, it sleeps for the limiter's own ``retry_after`` and
tries again. Timings here are deliberately short and the assertions use
generous bounds so the suite stays reliable on a loaded CI box.
"""

from __future__ import annotations

import asyncio
import threading
import time

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

ALL_ALGORITHMS = [TokenBucket, LeakyBucket, FixedWindow, SlidingWindow, SlidingLog]


def make_limiter(cls: type[RateLimiter], *, capacity: int = 2) -> RateLimiter:
    """Build any algorithm with a short refill period and the given capacity."""
    if cls in (TokenBucket, LeakyBucket):
        return cls(rate=20.0, capacity=capacity)
    return cls(limit=capacity, window=0.1)


# ------------------------------------------------------------------ #
# Happy path
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_wait_returns_immediately_when_capacity_is_free(
    algorithm: type[RateLimiter],
) -> None:
    """A limiter with spare capacity does not sleep."""
    limiter = make_limiter(algorithm)
    start = time.monotonic()
    result = limiter.wait()
    assert result.allowed is True
    assert time.monotonic() - start < 1.0, "wait() should not have slept at all"


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_wait_blocks_then_succeeds(algorithm: type[RateLimiter]) -> None:
    """When exhausted, wait() sleeps until capacity returns and then consumes."""
    limiter = make_limiter(algorithm, capacity=1)
    assert limiter.allow().allowed is True

    start = time.monotonic()
    result = limiter.wait(timeout=5.0)
    elapsed = time.monotonic() - start

    assert result.allowed is True, "wait() should eventually succeed"
    assert elapsed > 0.0, "wait() should have actually waited"
    assert elapsed < 5.0


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_wait_consumes_capacity(algorithm: type[RateLimiter]) -> None:
    """A successful wait() spends capacity, exactly like allow()."""
    limiter = make_limiter(algorithm, capacity=5)
    before = limiter.remaining()
    limiter.wait()
    assert limiter.remaining() < before


def test_wait_honours_cost() -> None:
    """The cost argument is forwarded to the underlying allow()."""
    limiter = TokenBucket(rate=100.0, capacity=5)
    result = limiter.wait(cost=3)
    assert result.allowed is True
    # Read the count off the result: re-reading the limiter would race the
    # refill and make this flaky on a loaded runner.
    assert result.remaining == 2


# ------------------------------------------------------------------ #
# Timeout
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_wait_gives_up_at_timeout(algorithm: type[RateLimiter]) -> None:
    """On timeout, wait() returns the denial rather than raising."""
    if algorithm in (TokenBucket, LeakyBucket):
        limiter: RateLimiter = algorithm(rate=0.01, capacity=1)
    else:
        limiter = algorithm(limit=1, window=60.0)
    limiter.allow()

    start = time.monotonic()
    result = limiter.wait(timeout=0.15)
    elapsed = time.monotonic() - start

    assert result.allowed is False
    assert 0.1 <= elapsed < 2.0, f"timeout not respected: waited {elapsed}s"


def test_wait_with_zero_timeout_is_a_single_attempt() -> None:
    """timeout=0 degrades to one allow() call with no sleeping."""
    limiter = TokenBucket(rate=0.01, capacity=1)
    limiter.allow()

    start = time.monotonic()
    result = limiter.wait(timeout=0.0)
    assert result.allowed is False
    assert time.monotonic() - start < 1.0, "timeout=0 should not have slept"


def test_wait_rejects_negative_timeout() -> None:
    """A negative timeout is a programming error."""
    limiter = TokenBucket(rate=10.0, capacity=5)
    with pytest.raises(ValueError, match="timeout must be >= 0"):
        limiter.wait(timeout=-1.0)


def test_wait_validates_cost() -> None:
    """Cost validation happens before any sleeping."""
    limiter = TokenBucket(rate=10.0, capacity=5)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        limiter.wait(cost=99)
    with pytest.raises(ValueError):
        limiter.wait(cost=0)


# ------------------------------------------------------------------ #
# Concurrency
# ------------------------------------------------------------------ #


def test_wait_does_not_hold_the_lock_while_sleeping() -> None:
    """A blocked wait() must not stop other threads from using the limiter.

    If ``wait()`` slept while holding the limiter's lock, the ``allow()``
    below would block behind it and the elapsed time would balloon.
    """
    limiter = TokenBucket(rate=1.0, capacity=1)
    limiter.allow()

    waiter = threading.Thread(target=limiter.wait, kwargs={"timeout": 2.0})
    waiter.start()
    time.sleep(0.05)

    start = time.monotonic()
    limiter.allow()
    assert time.monotonic() - start < 1.0, "allow() blocked behind a sleeping wait()"

    waiter.join(timeout=5.0)
    assert not waiter.is_alive()


def test_concurrent_waiters_do_not_exceed_capacity() -> None:
    """Ten threads racing on wait() never over-admit within a window."""
    limiter = FixedWindow(limit=3, window=10.0)
    admitted: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait()
        result = limiter.wait(timeout=0.2)
        with lock:
            admitted.append(result.allowed)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert sum(admitted) == 3, f"expected exactly 3 admitted, got {sum(admitted)}"


# ------------------------------------------------------------------ #
# wait_async
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_wait_async_returns_immediately_when_free(
    algorithm: type[RateLimiter],
) -> None:
    """The async variant admits straight away when there is capacity."""
    limiter = make_limiter(algorithm)
    result = asyncio.run(limiter.wait_async())
    assert result.allowed is True


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_wait_async_blocks_then_succeeds(algorithm: type[RateLimiter]) -> None:
    """When exhausted, the async variant waits and then consumes."""
    limiter = make_limiter(algorithm, capacity=1)
    assert limiter.allow().allowed is True

    result = asyncio.run(limiter.wait_async(timeout=5.0))
    assert result.allowed is True


def test_wait_async_times_out_without_raising() -> None:
    """A timed-out async wait returns the denial, like the sync one."""
    limiter = FixedWindow(limit=1, window=60.0)
    limiter.allow()
    assert asyncio.run(limiter.wait_async(timeout=0.05)).allowed is False


def test_wait_async_validates_arguments() -> None:
    """Validation matches the synchronous path."""
    limiter = TokenBucket(rate=10.0, capacity=5)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        asyncio.run(limiter.wait_async(cost=99))
    with pytest.raises(ValueError, match="timeout must be >= 0"):
        asyncio.run(limiter.wait_async(timeout=-1.0))


def test_wait_async_yields_to_the_event_loop() -> None:
    """A blocked wait_async must not stall other coroutines.

    This is the whole reason it exists: the synchronous wait() would hold
    the thread and stop the loop from making any progress at all.
    """
    limiter = TokenBucket(rate=20.0, capacity=1)
    limiter.allow()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.005)
            ticks += 1

    async def main() -> bool:
        waiter = asyncio.create_task(limiter.wait_async(timeout=5.0))
        beat = asyncio.create_task(ticker())
        result = await waiter
        beat.cancel()
        return result.allowed

    assert asyncio.run(main()) is True
    assert ticks > 0, "the event loop made no progress while wait_async slept"


def test_manager_wait_async_delegates_per_key() -> None:
    """The manager's async wait blocks on that key's limiter."""
    manager = RateLimitManager(TokenBucket, rate=50.0, capacity=1)
    assert manager.allow("user-1").allowed is True

    result = asyncio.run(manager.wait_async("user-1", timeout=5.0))
    assert result.allowed is True


def test_wait_without_a_timeout_blocks_until_capacity_returns() -> None:
    """timeout=None waits as long as it takes rather than giving up."""
    limiter = TokenBucket(rate=25.0, capacity=1)
    limiter.allow()

    start = time.monotonic()
    result = limiter.wait()
    elapsed = time.monotonic() - start

    assert result.allowed is True
    assert elapsed > 0.0, "it should have had to wait"
    assert elapsed < 5.0
