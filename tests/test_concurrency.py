"""Thread safety tests.

Prove that every limiter and the RateLimitManager remain correct under
heavy concurrent access.  Each test launches 50 threads that hammer a
shared limiter simultaneously, then asserts that no more than ``capacity``
requests were admitted.
"""

from __future__ import annotations

import random
import threading

from limitra import (
    FixedWindow,
    LeakyBucket,
    RateLimitManager,
    SlidingLog,
    SlidingWindow,
    TokenBucket,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_THREADS = 50
OPS_PER_THREAD = 10
CAPACITY = 200


# ---------------------------------------------------------------------------
# Individual algorithm concurrency tests
# ---------------------------------------------------------------------------

def test_concurrent_token_bucket() -> None:
    """50 threads x 10 allow() on TokenBucket -- total allowed <= capacity."""
    limiter = TokenBucket(rate=0.001, capacity=CAPACITY)
    results: list[bool] = []
    barrier = threading.Barrier(NUM_THREADS)
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        local: list[bool] = []
        for _ in range(OPS_PER_THREAD):
            local.append(limiter.allow().allowed)
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_allowed = sum(1 for r in results if r)
    assert total_allowed <= CAPACITY
    assert len(results) == NUM_THREADS * OPS_PER_THREAD


def test_concurrent_leaky_bucket() -> None:
    """50 threads x 10 allow() on LeakyBucket -- total allowed <= capacity."""
    limiter = LeakyBucket(rate=0.001, capacity=CAPACITY)
    results: list[bool] = []
    barrier = threading.Barrier(NUM_THREADS)
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        local: list[bool] = []
        for _ in range(OPS_PER_THREAD):
            local.append(limiter.allow().allowed)
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_allowed = sum(1 for r in results if r)
    assert total_allowed <= CAPACITY
    assert len(results) == NUM_THREADS * OPS_PER_THREAD


def test_concurrent_fixed_window() -> None:
    """50 threads x 10 allow() on FixedWindow -- total allowed <= limit."""
    limiter = FixedWindow(limit=CAPACITY, window=10.0)
    results: list[bool] = []
    barrier = threading.Barrier(NUM_THREADS)
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        local: list[bool] = []
        for _ in range(OPS_PER_THREAD):
            local.append(limiter.allow().allowed)
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_allowed = sum(1 for r in results if r)
    assert total_allowed <= CAPACITY
    assert len(results) == NUM_THREADS * OPS_PER_THREAD


def test_concurrent_sliding_window() -> None:
    """50 threads x 10 allow() on SlidingWindow -- total allowed <= limit."""
    limiter = SlidingWindow(limit=CAPACITY, window=10.0)
    results: list[bool] = []
    barrier = threading.Barrier(NUM_THREADS)
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        local: list[bool] = []
        for _ in range(OPS_PER_THREAD):
            local.append(limiter.allow().allowed)
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_allowed = sum(1 for r in results if r)
    assert total_allowed <= CAPACITY
    assert len(results) == NUM_THREADS * OPS_PER_THREAD


def test_concurrent_sliding_log() -> None:
    """50 threads x 10 allow() on SlidingLog -- total allowed <= limit."""
    limiter = SlidingLog(limit=CAPACITY, window=10.0)
    results: list[bool] = []
    barrier = threading.Barrier(NUM_THREADS)
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        local: list[bool] = []
        for _ in range(OPS_PER_THREAD):
            local.append(limiter.allow().allowed)
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_allowed = sum(1 for r in results if r)
    assert total_allowed <= CAPACITY
    assert len(results) == NUM_THREADS * OPS_PER_THREAD


# ---------------------------------------------------------------------------
# RateLimitManager concurrency
# ---------------------------------------------------------------------------

def test_concurrent_manager() -> None:
    """50 threads call manager.allow('key') — total allowed ≤ capacity."""
    manager = RateLimitManager(TokenBucket, rate=0.001, capacity=CAPACITY)
    results: list[bool] = []
    barrier = threading.Barrier(NUM_THREADS)
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        local: list[bool] = []
        for _ in range(OPS_PER_THREAD):
            local.append(manager.allow("shared-key").allowed)
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_allowed = sum(1 for r in results if r)
    assert total_allowed <= CAPACITY
    assert len(results) == NUM_THREADS * OPS_PER_THREAD


# ---------------------------------------------------------------------------
# Deadlock detection
# ---------------------------------------------------------------------------

def test_no_deadlock() -> None:
    """20 threads randomly call allow/peek/remaining/reset_after — must complete in 5s.

    If any lock ordering issue or deadlock exists, the test will time out.
    """
    limiter = TokenBucket(rate=100.0, capacity=50)
    completed = threading.Event()
    num_deadlock_threads = 20
    ops = 100

    def worker() -> None:
        for _ in range(ops):
            op = random.choice(["allow", "peek", "remaining", "reset_after"])
            if op == "allow":
                limiter.allow()
            elif op == "peek":
                limiter.peek()
            elif op == "remaining":
                limiter.remaining()
            else:
                limiter.reset_after()

    threads = [threading.Thread(target=worker) for _ in range(num_deadlock_threads)]
    for t in threads:
        t.start()

    # Use a timer to detect deadlock
    def timeout_handler() -> None:
        if not completed.is_set():
            msg = "Deadlock detected: threads did not complete"
            raise AssertionError(msg)

    timer = threading.Timer(5.0, timeout_handler)
    timer.start()

    for t in threads:
        t.join(timeout=5.0)

    completed.set()
    timer.cancel()

    # Verify all threads actually finished
    for t in threads:
        assert not t.is_alive(), "Thread still alive — possible deadlock"
