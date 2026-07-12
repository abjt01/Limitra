"""Tests for the RateLimitManager."""

from __future__ import annotations

import time

from limitra import RateLimitManager, RateLimitResult, SlidingWindow, TokenBucket

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
