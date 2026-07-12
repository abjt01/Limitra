"""Limitra — A Modern Python Library for Rate Limiting Algorithms.

Provides clean, efficient, production-ready implementations of the most
widely used rate limiting algorithms through a unified Python API.

Example:
    >>> from limitra import TokenBucket
    >>> limiter = TokenBucket(rate=10.0, capacity=100)
    >>> result = limiter.allow()
    >>> result.allowed
    True
"""

from __future__ import annotations

from limitra._base import RateLimiter, RateLimitResult
from limitra._fixed_window import FixedWindow
from limitra._leaky_bucket import LeakyBucket
from limitra._manager import RateLimitManager
from limitra._sliding_log import SlidingLog
from limitra._sliding_window import SlidingWindow
from limitra._token_bucket import TokenBucket

__all__ = [
    "FixedWindow",
    "LeakyBucket",
    "RateLimitManager",
    "RateLimitResult",
    "RateLimiter",
    "SlidingLog",
    "SlidingWindow",
    "TokenBucket",
]

__version__ = "0.1.0"
