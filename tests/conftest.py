"""Shared test fixtures."""

from __future__ import annotations

import pytest

from limitra import RateLimiter


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
