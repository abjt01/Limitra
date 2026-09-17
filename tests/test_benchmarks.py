"""Performance benchmarks for all rate limiting algorithms.

These are opt-in — they are deselected by default and run with
``pytest -m benchmark -s``, which also prints the throughput figures.
Wall-clock numbers are far too noisy on a shared CI runner to gate a pull
request on, so the absolute floor here is deliberately generous and the
assertion that actually protects anything is the *scaling* check at the
bottom, which compares a limiter against itself.
"""

from __future__ import annotations

import time

import pytest

from limitra import (
    FixedWindow,
    LeakyBucket,
    RateLimiter,
    SlidingLog,
    SlidingWindow,
    TokenBucket,
)

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCH_OPS = 100_000

#: A floor low enough that nothing could ever trip it guards nothing. All
#: five algorithms do 800k+ ops/sec on ordinary hardware and several
#: hundred thousand on a shared runner, so this catches roughly an order of
#: magnitude of regression while leaving ample headroom for a slow machine.
MIN_THROUGHPUT = 50_000

ALL_ALGORITHMS = [TokenBucket, LeakyBucket, FixedWindow, SlidingWindow, SlidingLog]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_limiter(cls: type[RateLimiter], size: int) -> RateLimiter:
    """Build any algorithm with enough headroom that nothing is denied."""
    if cls in (TokenBucket, LeakyBucket):
        return cls(rate=float(size), capacity=size)
    return cls(limit=size, window=999.0)


def throughput(limiter: RateLimiter, ops: int) -> float:
    """Return ops/sec for ``ops`` consecutive allow() calls."""
    start = time.perf_counter()
    for _ in range(ops):
        limiter.allow()
    elapsed = time.perf_counter() - start
    return ops / elapsed


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_throughput(algorithm: type[RateLimiter]) -> None:
    """Every algorithm clears the throughput floor at the same N.

    All five are measured at ``BENCH_OPS``; giving one of them a smaller N
    would let it clear the floor on work the others are not doing.
    """
    limiter = make_limiter(algorithm, BENCH_OPS)
    ops_per_sec = throughput(limiter, BENCH_OPS)
    print(f"\n  {algorithm.__name__:<16s} {ops_per_sec:>12,.0f} ops/sec")
    assert ops_per_sec > MIN_THROUGHPUT, (
        f"{algorithm.__name__} managed {ops_per_sec:,.0f} ops/sec, below the "
        f"floor of {MIN_THROUGHPUT:,}"
    )


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_throughput_when_saturated(algorithm: type[RateLimiter]) -> None:
    """The denied path is the hot one under abuse, so measure it too."""
    limiter = make_limiter(algorithm, 1_000)
    while limiter.allow().allowed:
        pass

    ops_per_sec = throughput(limiter, 20_000)
    print(f"\n  {algorithm.__name__:<16s} {ops_per_sec:>12,.0f} ops/sec (denying)")
    assert ops_per_sec > MIN_THROUGHPUT


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_sliding_log_cost_per_call_does_not_grow_with_the_log() -> None:
    """SlidingLog must stay flat per call as its log gets longer.

    Pruning used to rebuild the whole list on every call, which made each
    call O(n) and a full window O(n^2). This compares the limiter against
    itself at two sizes, so it holds regardless of how fast the machine is.
    """
    small = SlidingLog(limit=2_000, window=999.0)
    large = SlidingLog(limit=20_000, window=999.0)
    for limiter in (small, large):
        while limiter.allow().allowed:
            pass

    probes = 20_000
    per_call_small = 1.0 / throughput(small, probes)
    per_call_large = 1.0 / throughput(large, probes)
    ratio = per_call_large / per_call_small

    print(f"\n  SlidingLog 10x longer log -> {ratio:.2f}x cost per call")
    assert ratio < 4.0, (
        f"a 10x longer log made each call {ratio:.1f}x more expensive; "
        f"pruning looks linear in the log length again"
    )
