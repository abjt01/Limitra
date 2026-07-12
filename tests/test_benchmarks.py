"""Performance benchmarks for all rate limiting algorithms.

These tests measure throughput (ops/sec) and assert a minimum sanity floor
of 10,000 ops/sec.  Run with ``pytest -s`` to see printed throughput figures.
"""

from __future__ import annotations

import time

from limitra import (
    FixedWindow,
    LeakyBucket,
    SlidingLog,
    SlidingWindow,
    TokenBucket,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCH_OPS = 100_000
BENCH_OPS_LOG = 10_000  # SlidingLog is O(n) per call — use smaller N
MIN_THROUGHPUT = 10_000  # sanity floor: ops/sec


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def test_token_bucket_throughput() -> None:
    """Time 100k allow() calls on TokenBucket, print ops/sec."""
    limiter = TokenBucket(rate=1_000_000.0, capacity=1_000_000)
    start = time.perf_counter()
    for _ in range(BENCH_OPS):
        limiter.allow()
    elapsed = time.perf_counter() - start
    ops_per_sec = BENCH_OPS / elapsed
    print(f"\n  TokenBucket:    {ops_per_sec:>12,.0f} ops/sec  ({elapsed:.3f}s)")
    assert ops_per_sec > MIN_THROUGHPUT


def test_leaky_bucket_throughput() -> None:
    """Time 100k allow() calls on LeakyBucket, print ops/sec."""
    limiter = LeakyBucket(rate=1_000_000.0, capacity=1_000_000)
    start = time.perf_counter()
    for _ in range(BENCH_OPS):
        limiter.allow()
    elapsed = time.perf_counter() - start
    ops_per_sec = BENCH_OPS / elapsed
    print(f"\n  LeakyBucket:    {ops_per_sec:>12,.0f} ops/sec  ({elapsed:.3f}s)")
    assert ops_per_sec > MIN_THROUGHPUT


def test_fixed_window_throughput() -> None:
    """Time 100k allow() calls on FixedWindow, print ops/sec."""
    limiter = FixedWindow(limit=1_000_000, window=999.0)
    start = time.perf_counter()
    for _ in range(BENCH_OPS):
        limiter.allow()
    elapsed = time.perf_counter() - start
    ops_per_sec = BENCH_OPS / elapsed
    print(f"\n  FixedWindow:    {ops_per_sec:>12,.0f} ops/sec  ({elapsed:.3f}s)")
    assert ops_per_sec > MIN_THROUGHPUT


def test_sliding_window_throughput() -> None:
    """Time 100k allow() calls on SlidingWindow, print ops/sec."""
    limiter = SlidingWindow(limit=1_000_000, window=999.0)
    start = time.perf_counter()
    for _ in range(BENCH_OPS):
        limiter.allow()
    elapsed = time.perf_counter() - start
    ops_per_sec = BENCH_OPS / elapsed
    print(f"\n  SlidingWindow:  {ops_per_sec:>12,.0f} ops/sec  ({elapsed:.3f}s)")
    assert ops_per_sec > MIN_THROUGHPUT


def test_sliding_log_throughput() -> None:
    """Time 10k allow() calls on SlidingLog (O(n) log), print ops/sec."""
    limiter = SlidingLog(limit=1_000_000, window=999.0)
    start = time.perf_counter()
    for _ in range(BENCH_OPS_LOG):
        limiter.allow()
    elapsed = time.perf_counter() - start
    ops_per_sec = BENCH_OPS_LOG / elapsed
    print(f"\n  SlidingLog:     {ops_per_sec:>12,.0f} ops/sec  ({elapsed:.3f}s)")
    assert ops_per_sec > MIN_THROUGHPUT


# ---------------------------------------------------------------------------
# Aggregate sanity check
# ---------------------------------------------------------------------------

def test_all_algorithms_minimum_throughput() -> None:
    """Assert ALL algorithms can do at least 10,000 ops/sec."""
    configs = [
        ("TokenBucket", TokenBucket(rate=1e6, capacity=1_000_000), BENCH_OPS),
        ("LeakyBucket", LeakyBucket(rate=1e6, capacity=1_000_000), BENCH_OPS),
        ("FixedWindow", FixedWindow(limit=1_000_000, window=999.0), BENCH_OPS),
        ("SlidingWindow", SlidingWindow(limit=1_000_000, window=999.0), BENCH_OPS),
        ("SlidingLog", SlidingLog(limit=1_000_000, window=999.0), BENCH_OPS_LOG),
    ]

    print()  # blank line for readability with -s
    for name, limiter, ops in configs:
        start = time.perf_counter()
        for _ in range(ops):
            limiter.allow()
        elapsed = time.perf_counter() - start
        ops_per_sec = ops / elapsed
        print(f"  {name:<16s} {ops_per_sec:>12,.0f} ops/sec")
        assert ops_per_sec > MIN_THROUGHPUT, (
            f"{name} throughput {ops_per_sec:,.0f} ops/sec is below "
            f"minimum floor of {MIN_THROUGHPUT:,} ops/sec"
        )
