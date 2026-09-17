"""Tests for the RateLimitManager."""

from __future__ import annotations

import threading
import time

import pytest

from limitra import (
    FixedWindow,
    RateLimitManager,
    RateLimitResult,
    SlidingWindow,
    TokenBucket,
)

from .conftest import wait_until_blocked

# ------------------------------------------------------------------ #
# Initialisation
# ------------------------------------------------------------------ #


def test_init() -> None:
    """RateLimitManager can be created with a TokenBucket algorithm."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=100)
    assert len(mgr) == 0
    assert mgr.keys() == []


# ------------------------------------------------------------------ #
# allow()
# ------------------------------------------------------------------ #


def test_allow_creates_limiter() -> None:
    """First allow for a key creates a new limiter."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    assert "user-1" not in mgr

    result = mgr.allow("user-1")
    assert result.allowed is True
    assert "user-1" in mgr
    assert len(mgr) == 1


def test_allow_per_key_isolation() -> None:
    """Different keys have independent limits."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=2)

    # Exhaust user-1
    mgr.allow("user-1")
    mgr.allow("user-1")
    denied = mgr.allow("user-1")
    assert denied.allowed is False

    # user-2 should still be fresh
    result = mgr.allow("user-2")
    assert result.allowed is True
    assert result.remaining == 1


def test_allow_cost_parameter() -> None:
    """Cost is forwarded to the underlying limiter."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)

    result = mgr.allow("user-1", cost=3)
    assert result.allowed is True
    assert result.remaining == 2

    # Only 2 tokens left, cost=3 should be denied
    denied = mgr.allow("user-1", cost=3)
    assert denied.allowed is False


# ------------------------------------------------------------------ #
# peek()
# ------------------------------------------------------------------ #


def test_peek() -> None:
    """peek() works and doesn't consume capacity."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)

    # First allow to create the limiter
    mgr.allow("user-1")

    # Peek should report status without consuming
    peek_result = mgr.peek("user-1")
    assert isinstance(peek_result, RateLimitResult)
    assert peek_result.allowed is True

    # Remaining should not have changed from the peek
    remaining_before = mgr.peek("user-1").remaining
    mgr.peek("user-1")
    mgr.peek("user-1")
    assert mgr.peek("user-1").remaining == remaining_before


# ------------------------------------------------------------------ #
# get()
# ------------------------------------------------------------------ #


def test_get_existing_key() -> None:
    """get() returns the limiter for an existing key."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("user-1")

    limiter = mgr.get("user-1")
    assert limiter is not None
    assert isinstance(limiter, TokenBucket)


def test_get_missing_key() -> None:
    """get() returns None for an unknown key."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    assert mgr.get("nonexistent") is None


# ------------------------------------------------------------------ #
# remove()
# ------------------------------------------------------------------ #


def test_remove_existing_key() -> None:
    """remove() returns True for an existing key and removes it."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("user-1")
    assert "user-1" in mgr

    assert mgr.remove("user-1") is True
    assert "user-1" not in mgr
    assert len(mgr) == 0


def test_remove_missing_key() -> None:
    """remove() returns False for an unknown key."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    assert mgr.remove("nonexistent") is False


# ------------------------------------------------------------------ #
# cleanup()
# ------------------------------------------------------------------ #


def test_cleanup() -> None:
    """cleanup() removes idle keys past max_idle."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("old-user")

    # Sleep so the key becomes stale
    time.sleep(0.15)

    removed = mgr.cleanup(max_idle=0.1)
    assert removed == 1
    assert "old-user" not in mgr
    assert len(mgr) == 0


def test_cleanup_keeps_active() -> None:
    """cleanup() keeps recently accessed keys."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("active-user")

    # Sleep a short time but less than max_idle
    time.sleep(0.05)

    # Access the key again to refresh it
    mgr.allow("active-user")

    # Cleanup with max_idle longer than time since last access
    removed = mgr.cleanup(max_idle=0.1)
    assert removed == 0
    assert "active-user" in mgr


# ------------------------------------------------------------------ #
# keys() / len() / contains
# ------------------------------------------------------------------ #


def test_keys() -> None:
    """keys() returns all tracked key names."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("alice")
    mgr.allow("bob")
    mgr.allow("charlie")

    keys = mgr.keys()
    assert sorted(keys) == ["alice", "bob", "charlie"]


def test_len() -> None:
    """len() returns the number of tracked keys."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    assert len(mgr) == 0

    mgr.allow("a")
    assert len(mgr) == 1

    mgr.allow("b")
    assert len(mgr) == 2

    mgr.remove("a")
    assert len(mgr) == 1


def test_contains() -> None:
    """'in' operator works for checking tracked keys."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)

    assert "user-1" not in mgr
    mgr.allow("user-1")
    assert "user-1" in mgr

    mgr.remove("user-1")
    assert "user-1" not in mgr


# ------------------------------------------------------------------ #
# repr
# ------------------------------------------------------------------ #


def test_repr() -> None:
    """Repr contains the algorithm name."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    r = repr(mgr)
    assert "RateLimitManager" in r
    assert "TokenBucket" in r


# ------------------------------------------------------------------ #
# Algorithm swap
# ------------------------------------------------------------------ #


def test_swap_algorithm() -> None:
    """The same manager interface works with different algorithms."""
    # Test with SlidingWindow instead of TokenBucket
    mgr = RateLimitManager(SlidingWindow, limit=3, window=10.0)

    result = mgr.allow("user-1")
    assert result.allowed is True
    assert isinstance(result, RateLimitResult)

    # Exhaust the limit
    mgr.allow("user-1")
    mgr.allow("user-1")
    denied = mgr.allow("user-1")
    assert denied.allowed is False

    # Verify the underlying limiter is a SlidingWindow
    limiter = mgr.get("user-1")
    assert isinstance(limiter, SlidingWindow)


# ------------------------------------------------------------------ #
# Lifecycle: cleanup, eviction and the access stamp
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("accessor", ["peek", "remaining", "reset_after", "get"])
def test_read_only_accessors_do_not_create_keys(accessor: str) -> None:
    """Looking at an unknown key must not start tracking it.

    A metrics scrape or a probe for a key that does not exist would
    otherwise fill the map — and under max_keys, evict real users.
    """
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    getattr(mgr, accessor)("ghost")

    assert "ghost" not in mgr
    assert len(mgr) == 0


def test_read_only_accessors_answer_for_untracked_keys() -> None:
    """An untracked key reports what a fresh limiter would."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)

    assert mgr.remaining("ghost") == 5
    assert mgr.reset_after("ghost") == 0.0
    assert mgr.peek("ghost").allowed is True
    assert len(mgr) == 0


def test_read_only_accessors_cannot_evict_under_max_keys() -> None:
    """A flood of observed-but-unknown keys leaves tracked state alone."""
    mgr = RateLimitManager(TokenBucket, max_keys=2, rate=10.0, capacity=5)
    mgr.allow("real-1")
    mgr.allow("real-2")

    for i in range(100):
        mgr.peek(f"ghost-{i}")
        mgr.remaining(f"ghost-{i}")

    assert sorted(mgr.keys()) == ["real-1", "real-2"]


@pytest.mark.parametrize("accessor", ["allow", "peek", "remaining", "reset_after"])
def test_every_accessor_refreshes_the_idle_timer(accessor: str) -> None:
    """Any access to a tracked key keeps it alive across a cleanup sweep."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("user-1")
    getattr(mgr, accessor)("user-1")

    # A generous idle budget: the key was just touched, so nothing is stale.
    assert mgr.cleanup(max_idle=60.0) == 0
    assert "user-1" in mgr


def test_cleanup_keeps_active_keys_and_drops_idle_ones(clock) -> None:
    """Only keys past max_idle are removed."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("old")
    clock.advance(60.0)
    mgr.allow("fresh")

    removed = mgr.cleanup(max_idle=30.0)
    assert removed == 1
    assert mgr.keys() == ["fresh"]


def test_cleanup_on_empty_manager() -> None:
    """Cleaning an empty manager is a no-op, not an error."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    assert mgr.cleanup() == 0


def test_cleanup_is_repeatable() -> None:
    """Running cleanup twice does not raise or double-count."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    mgr.allow("a")
    mgr.allow("b")
    assert mgr.cleanup(max_idle=0.0) == 2
    assert mgr.cleanup(max_idle=0.0) == 0
    assert len(mgr) == 0


def test_max_keys_evicts_least_recently_used() -> None:
    """With max_keys set, a new key pushes out the least recently used one."""
    mgr = RateLimitManager(TokenBucket, max_keys=2, rate=10.0, capacity=5)
    mgr.allow("a")
    mgr.allow("b")
    mgr.allow("a")  # 'a' is now more recent than 'b'
    mgr.allow("c")  # evicts 'b'

    assert len(mgr) == 2
    assert "b" not in mgr
    assert "a" in mgr
    assert "c" in mgr


def test_max_keys_bounds_memory_under_a_key_flood() -> None:
    """A flood of unique keys cannot grow the manager past max_keys."""
    mgr = RateLimitManager(TokenBucket, max_keys=10, rate=10.0, capacity=5)
    for i in range(1000):
        mgr.allow(f"ip-{i}")
    assert len(mgr) == 10


def test_max_keys_defaults_to_unbounded() -> None:
    """Without max_keys the manager tracks every key."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    for i in range(50):
        mgr.allow(f"key-{i}")
    assert len(mgr) == 50


def test_max_keys_must_be_positive() -> None:
    """max_keys below 1 is a programming error."""
    with pytest.raises(ValueError, match="max_keys must be >= 1"):
        RateLimitManager(TokenBucket, max_keys=0, rate=10.0, capacity=5)


# ------------------------------------------------------------------ #
# Delegating accessors
# ------------------------------------------------------------------ #


def test_remaining_delegates_to_the_keys_limiter() -> None:
    """remaining(key) reports that key's own budget."""
    mgr = RateLimitManager(TokenBucket, rate=1.0, capacity=5)
    mgr.allow("user-1", cost=2)

    assert mgr.remaining("user-1") == 3
    assert mgr.remaining("user-2") == 5, "a fresh key starts full"


def test_reset_after_delegates_to_the_keys_limiter() -> None:
    """reset_after(key) reports that key's refill time."""
    mgr = RateLimitManager(FixedWindow, limit=5, window=30.0)
    mgr.allow("user-1")
    assert 0.0 < mgr.reset_after("user-1") <= 30.0


def test_reset_restores_a_single_key() -> None:
    """reset(key) refills one key and leaves the others alone."""
    mgr = RateLimitManager(FixedWindow, limit=2, window=60.0)
    for key in ("user-1", "user-2"):
        mgr.allow(key)
        mgr.allow(key)
        assert mgr.allow(key).allowed is False

    assert mgr.reset("user-1") is True
    assert mgr.allow("user-1").allowed is True
    assert mgr.allow("user-2").allowed is False, "user-2 must be untouched"


def test_reset_reports_untracked_keys() -> None:
    """Resetting a key that was never seen returns False and creates nothing."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    assert mgr.reset("never-seen") is False
    assert len(mgr) == 0


def test_clear_removes_everything() -> None:
    """clear() drops all keys and reports how many went."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    for key in ("a", "b", "c"):
        mgr.allow(key)

    assert mgr.clear() == 3
    assert len(mgr) == 0
    assert mgr.clear() == 0


def test_clear_restores_capacity_for_returning_keys() -> None:
    """A key re-seen after clear() starts from a fresh limiter."""
    mgr = RateLimitManager(FixedWindow, limit=1, window=60.0)
    mgr.allow("user-1")
    assert mgr.allow("user-1").allowed is False

    mgr.clear()
    assert mgr.allow("user-1").allowed is True


# ------------------------------------------------------------------ #
# Cost validation flows through the manager
# ------------------------------------------------------------------ #


def test_cost_above_limit_raises_through_the_manager() -> None:
    """The per-algorithm cost contract is not softened by the manager."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        mgr.allow("user-1", cost=6)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        mgr.peek("user-1", cost=6)


# ------------------------------------------------------------------ #
# Configuration is validated up front
# ------------------------------------------------------------------ #


def test_bad_keyword_arguments_fail_at_construction() -> None:
    """A config typo surfaces when the manager is built, not mid-request."""
    with pytest.raises(TypeError, match="cannot build TokenBucket"):
        RateLimitManager(TokenBucket, rate=10.0, capacitty=5)


def test_out_of_range_arguments_fail_at_construction() -> None:
    """Algorithm-level range checks run once, at construction."""
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        RateLimitManager(TokenBucket, rate=10.0, capacity=0)


def test_algorithm_must_be_a_rate_limiter() -> None:
    """Passing something that is not a limiter is caught immediately."""
    with pytest.raises(TypeError, match="must be a RateLimiter subclass"):
        RateLimitManager(dict, rate=10.0)  # type: ignore[arg-type]


def test_invalid_cost_does_not_create_or_evict() -> None:
    """A request the library rejects must leave the map untouched."""
    mgr = RateLimitManager(TokenBucket, max_keys=2, rate=10.0, capacity=5)
    mgr.allow("real-1")
    mgr.allow("real-2")

    with pytest.raises(ValueError, match="cost must be <= limit"):
        mgr.allow("attacker", cost=999)

    assert "attacker" not in mgr
    assert sorted(mgr.keys()) == ["real-1", "real-2"]


# ------------------------------------------------------------------ #
# Eviction prefers keys with nothing to lose
# ------------------------------------------------------------------ #


def test_eviction_prefers_a_key_at_full_capacity() -> None:
    """Evicting a throttled key would hand back its whole budget."""
    mgr = RateLimitManager(FixedWindow, max_keys=2, limit=2, window=60.0)

    # 'throttled' has spent its budget; 'idle' has spent nothing.
    mgr.allow("throttled")
    mgr.allow("throttled")
    assert mgr.allow("throttled").allowed is False
    mgr.peek("idle")
    mgr.allow("idle")
    mgr.get("idle")
    mgr.reset("idle")  # back to full capacity, still tracked

    mgr.allow("newcomer")

    assert "throttled" in mgr, "the throttled key must survive eviction"
    assert mgr.allow("throttled").allowed is False, "and keep its state"


def test_eviction_falls_back_to_lru_when_everyone_is_throttled() -> None:
    """With nothing safe to drop, plain LRU still bounds the map."""
    mgr = RateLimitManager(FixedWindow, max_keys=3, limit=1, window=60.0)
    for key in ("a", "b", "c"):
        mgr.allow(key)

    mgr.allow("d")

    assert len(mgr) == 3
    assert "a" not in mgr


# ------------------------------------------------------------------ #
# wait()
# ------------------------------------------------------------------ #


def test_wait_delegates_per_key() -> None:
    """wait(key) blocks on that key's limiter and consumes from it."""
    mgr = RateLimitManager(TokenBucket, rate=50.0, capacity=1)
    assert mgr.allow("user-1").allowed is True

    result = mgr.wait("user-1", timeout=5.0)
    assert result.allowed is True
    assert "user-1" in mgr


def test_wait_times_out_without_raising() -> None:
    """A timed-out wait returns the denial."""
    mgr = RateLimitManager(FixedWindow, limit=1, window=60.0)
    mgr.allow("user-1")

    assert mgr.wait("user-1", timeout=0.05).allowed is False


def test_wait_validates_cost_before_creating_a_key() -> None:
    """An impossible wait is rejected without tracking the key."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        mgr.wait("user-1", cost=99)
    assert len(mgr) == 0


def test_manager_refund_credits_one_key() -> None:
    """refund(key) gives capacity back to that key alone."""
    mgr = RateLimitManager(FixedWindow, limit=2, window=60.0)
    for key in ("user-1", "user-2"):
        mgr.allow(key)
        mgr.allow(key)
        assert mgr.allow(key).allowed is False

    assert mgr.refund("user-1") is True

    assert mgr.allow("user-1").allowed is True
    assert mgr.allow("user-2").allowed is False, "user-2 must be untouched"


def test_manager_refund_ignores_untracked_keys() -> None:
    """There is nothing to credit back to a key that never spent anything."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    assert mgr.refund("never-seen") is False
    assert len(mgr) == 0


def test_manager_refund_validates_cost() -> None:
    """An impossible refund is rejected without touching the map."""
    mgr = RateLimitManager(TokenBucket, rate=10.0, capacity=5)
    with pytest.raises(ValueError, match="cost must be <= limit"):
        mgr.refund("user-1", cost=99)
    assert len(mgr) == 0


def test_eviction_cannot_reclaim_a_key_with_a_wait_in_flight() -> None:
    """max_keys must not drop the key a waiter is blocked on either."""
    mgr = RateLimitManager(FixedWindow, max_keys=2, limit=1, window=0.3)
    mgr.allow("waiter")

    thread = threading.Thread(
        target=mgr.wait, args=("waiter",), kwargs={"timeout": 3.0}
    )
    thread.start()
    wait_until_blocked(mgr, "waiter")

    for i in range(20):
        mgr.allow(f"flood-{i}")

    assert "waiter" in mgr, "an in-flight wait was evicted"
    thread.join(timeout=5.0)
    assert not thread.is_alive()


def test_eviction_falls_back_to_lru_when_every_candidate_is_waiting() -> None:
    """With nothing safe to drop, the map is still bounded.

    Holding keys for in-flight waits must not let the map grow without
    limit; if every candidate is busy, the least-recently-used one goes.
    """
    mgr = RateLimitManager(FixedWindow, max_keys=2, limit=1, window=0.3)
    threads = []
    for key in ("w1", "w2"):
        mgr.allow(key)
        thread = threading.Thread(target=mgr.wait, args=(key,), kwargs={"timeout": 3.0})
        thread.start()
        threads.append(thread)
        wait_until_blocked(mgr, key)

    mgr.allow("newcomer")

    assert len(mgr) <= 2
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()


def test_concurrent_waiters_on_one_key_are_counted() -> None:
    """Two waiters on the same key: the first to finish must not release it."""
    mgr = RateLimitManager(FixedWindow, limit=2, window=0.2)
    mgr.allow("shared")
    mgr.allow("shared")

    threads = [
        threading.Thread(target=mgr.wait, args=("shared",), kwargs={"timeout": 3.0})
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert "shared" in mgr
    assert mgr.cleanup(max_idle=0.0) == 1, "the key should be free again"
