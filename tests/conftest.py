"""Shared test fixtures."""

from __future__ import annotations

import time

import pytest

from limitra import RateLimiter, RateLimitManager


class FakeClock:
    """A monotonic clock the test drives by hand.

    Every limiter reads the time through ``RateLimiter._now``, so replacing
    it lets the timing-dependent maths be tested exactly instead of with
    ``time.sleep`` and tolerance bands that flake on a loaded CI runner.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        """Start the clock at an arbitrary but realistic monotonic value."""
        self.now = start

    def advance(self, seconds: float) -> float:
        """Move the clock forward.

        Args:
            seconds: How far to advance.

        Returns:
            The new time.
        """
        self.now += seconds
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Freeze every limiter's clock and hand the test the controls.

    Do not combine this with ``wait()``, which sleeps against the real
    clock and would never observe the frozen one advancing.
    """
    fake = FakeClock()
    monkeypatch.setattr(RateLimiter, "_now", staticmethod(lambda: fake.now))
    return fake


def wait_until_blocked(
    manager: RateLimitManager, key: str, timeout: float = 10.0
) -> None:
    """Block until *key* has a wait() registered on *manager*.

    Tests that race a background ``wait()`` need to know it has actually
    started; sleeping a fixed amount instead is what makes such a test flaky
    on a slow or shared CI runner.

    Args:
        manager: The manager to watch.
        key: The key the wait is expected on.
        timeout: How long to wait for registration before giving up.

    Raises:
        AssertionError: If the wait never registered.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if key in manager._waiting:
            return
        time.sleep(0.001)
    raise AssertionError(f"no wait() registered on {key!r} within {timeout}s")
