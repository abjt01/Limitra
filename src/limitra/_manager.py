"""Multi-key rate limit manager."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from itertools import islice
from typing import Any

from limitra._base import RateLimiter, RateLimitResult

#: How many of the oldest keys :meth:`RateLimitManager._evict_one` inspects
#: while looking for one that is at full capacity. Bounded so eviction stays
#: constant-time no matter how many keys are tracked.
_EVICTION_SCAN = 8


class RateLimitManager:
    """Orchestrates per-key rate limiters using any :class:`RateLimiter` algorithm.

    Creates and caches a separate :class:`RateLimiter` instance for each
    unique key (e.g., user ID, IP address, API key). Idle limiters can be
    reclaimed on demand via :meth:`cleanup`, and ``max_keys`` bounds the
    number tracked by evicting the least-recently-used key.

    This class does **not** extend :class:`RateLimiter` — it delegates to
    individual limiter instances instead.

    Only the spending methods — :meth:`allow`, :meth:`wait` and
    :meth:`wait_async` — create a key. The read-only ones answer for an
    unknown key without starting to track it, so a metrics scrape or a
    probe for a key that does not exist cannot fill the map.

    Bounding memory:
        Keys are usually attacker-controlled (an IP address, an API token),
        so a flood of distinct keys grows the map without limit unless you
        bound it. ``cleanup()`` is the safer of the two options: it only
        drops keys that have gone quiet. ``max_keys`` gives a hard ceiling
        on the number of entries, at the cost that evicting a key discards
        its state — so a key that was being throttled comes back with a
        full budget. Eviction prefers keys that are already at full
        capacity, which have nothing to lose, but under a large enough
        flood it will eventually drop a throttled one. Size ``max_keys``
        well above your expected number of active keys, and prefer
        ``cleanup()`` when you cannot.

    Args:
        algorithm: A :class:`RateLimiter` subclass to instantiate per key.
        max_keys: Maximum number of keys to track. When the map is full,
            adding a new key evicts one. ``None`` (the default) means
            unbounded.
        **kwargs: Keyword arguments forwarded to ``algorithm(...)`` when
            creating new limiter instances. They are validated once here,
            by building a limiter, rather than on the first request.

    Raises:
        TypeError: If ``algorithm`` is not a :class:`RateLimiter` subclass,
            or ``kwargs`` does not match its signature.
        ValueError: If ``max_keys`` is less than 1, or ``kwargs`` holds an
            out-of-range value for the algorithm.

    Example:
        >>> from limitra import RateLimitManager, SlidingWindow
        >>> mgr = RateLimitManager(SlidingWindow, limit=100, window=60.0)
        >>> mgr.allow("user-42").allowed
        True
    """

    __slots__ = ("_algorithm", "_entries", "_kwargs", "_lock", "_max_keys", "_probe")

    def __init__(
        self,
        algorithm: type[RateLimiter],
        *,
        max_keys: int | None = None,
        **kwargs: Any,
    ) -> None:
        if max_keys is not None and max_keys < 1:
            raise ValueError(f"max_keys must be >= 1 or None, got {max_keys}")
        if not (isinstance(algorithm, type) and issubclass(algorithm, RateLimiter)):
            raise TypeError(
                f"algorithm must be a RateLimiter subclass, got {algorithm!r}"
            )
        self._algorithm = algorithm
        self._kwargs = kwargs
        self._max_keys = max_keys
        # Build one limiter up front. It doubles as a check that the
        # keyword arguments actually fit the algorithm — otherwise the
        # first request for a new key would be the one to find out — and as
        # the stand-in used to answer read-only calls for untracked keys.
        # It is never spent from, so it stays a pristine "never seen" key.
        try:
            probe = algorithm(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"cannot build {algorithm.__name__} from the manager's keyword "
                f"arguments {sorted(kwargs)}: {exc}"
            ) from exc
        self._probe: RateLimiter = probe
        # key -> (limiter, last access time). A single mapping keeps the
        # limiter and its access stamp impossible to get out of sync.
        self._entries: OrderedDict[str, tuple[RateLimiter, float]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Drop one key to make room for a new one.

        Prefers a key that is at full capacity: evicting a throttled key
        would hand it back its whole budget, so a flood of fresh keys could
        be used to clear somebody else's limit. Only the oldest few are
        inspected, so this stays constant-time.

        Must be called while ``self._lock`` is held.
        """
        for key in islice(self._entries, _EVICTION_SCAN):
            limiter, _ = self._entries[key]
            if limiter.remaining() >= limiter.limit:
                del self._entries[key]
                return
        self._entries.popitem(last=False)

    def _touch(self, key: str) -> RateLimiter:
        """Return the limiter for *key*, creating it if necessary.

        Records the access time and marks *key* most-recently-used,
        evicting a key first if ``max_keys`` would be exceeded.

        Must be called while ``self._lock`` is held.

        Args:
            key: The rate-limit key.

        Returns:
            The :class:`RateLimiter` instance for *key*.
        """
        entry = self._entries.get(key)
        if entry is None:
            if self._max_keys is not None and len(self._entries) >= self._max_keys:
                self._evict_one()
            limiter = self._algorithm(**self._kwargs)
        else:
            limiter = entry[0]
        self._entries[key] = (limiter, time.monotonic())
        self._entries.move_to_end(key)
        return limiter

    def _observe(self, key: str) -> RateLimiter:
        """Return the limiter to answer a read-only question about *key*.

        An untracked key is answered from the pristine probe limiter, so
        looking at a key never starts tracking it.

        Must be called while ``self._lock`` is held.

        Args:
            key: The rate-limit key.

        Returns:
            The key's limiter, or the probe if the key is untracked.
        """
        entry = self._entries.get(key)
        if entry is None:
            return self._probe
        self._entries[key] = (entry[0], time.monotonic())
        self._entries.move_to_end(key)
        return entry[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self, key: str, cost: int = 1) -> RateLimitResult:
        """Consume ``cost`` units from the limiter associated with *key*.

        Args:
            key: The rate-limit key (e.g., user ID, IP address).
            cost: Number of units to consume. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` with the decision and metadata.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range for the algorithm.
        """
        # Validate before touching the map: a request the library is about
        # to reject as invalid must not create a key, still less evict a
        # real one under max_keys.
        self._probe._validate_cost(cost)
        with self._lock:
            limiter = self._touch(key)
        return limiter.allow(cost)

    def wait(
        self, key: str, cost: int = 1, timeout: float | None = None
    ) -> RateLimitResult:
        """Block until *key* has ``cost`` units available, then consume them.

        Args:
            key: The rate-limit key.
            cost: Number of units to consume. Defaults to 1.
            timeout: Maximum seconds to wait. ``None`` waits indefinitely.

        Returns:
            The successful :class:`RateLimitResult`, or the final denied
            result if ``timeout`` elapsed first.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range, or ``timeout`` is
                negative.
        """
        self._probe._validate_cost(cost)
        with self._lock:
            limiter = self._touch(key)
        return limiter.wait(cost, timeout)

    async def wait_async(
        self, key: str, cost: int = 1, timeout: float | None = None
    ) -> RateLimitResult:
        """Await until *key* has ``cost`` units available, then consume them.

        Args:
            key: The rate-limit key.
            cost: Number of units to consume. Defaults to 1.
            timeout: Maximum seconds to wait. ``None`` waits indefinitely.

        Returns:
            The successful :class:`RateLimitResult`, or the final denied
            result if ``timeout`` elapsed first.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range, or ``timeout`` is
                negative.
        """
        self._probe._validate_cost(cost)
        with self._lock:
            limiter = self._touch(key)
        return await limiter.wait_async(cost, timeout)

    def peek(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check whether a request for *key* would be allowed, without consuming.

        Does not start tracking an unknown key.

        Args:
            key: The rate-limit key.
            cost: Number of units to check. Defaults to 1.

        Returns:
            A :class:`RateLimitResult` representing what *would* happen.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range for the algorithm.
        """
        self._probe._validate_cost(cost)
        with self._lock:
            limiter = self._observe(key)
        return limiter.peek(cost)

    def remaining(self, key: str) -> int:
        """Return the units still available to *key*.

        Does not start tracking an unknown key.

        Args:
            key: The rate-limit key.

        Returns:
            Number of requests *key* could make right now.
        """
        with self._lock:
            limiter = self._observe(key)
        return limiter.remaining()

    def reset_after(self, key: str) -> float:
        """Return seconds until *key*'s limiter is back at full capacity.

        Does not start tracking an unknown key.

        Args:
            key: The rate-limit key.

        Returns:
            Seconds until full capacity is restored for *key*.
        """
        with self._lock:
            limiter = self._observe(key)
        return limiter.reset_after()

    def reset(self, key: str) -> bool:
        """Reset *key*'s limiter to full capacity, keeping it tracked.

        Args:
            key: The rate-limit key.

        Returns:
            ``True`` if the key was tracked and was reset, ``False`` if it
            was not tracked (in which case nothing happens — an untracked
            key already has full capacity).
        """
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return False
        entry[0].reset()
        return True

    def refund(self, key: str, cost: int = 1) -> bool:
        """Give ``cost`` units back to *key*'s limiter.

        Does not start tracking an unknown key — there is nothing to credit
        back to a key that has never spent anything.

        Args:
            key: The rate-limit key.
            cost: Number of units to return. Defaults to 1.

        Returns:
            ``True`` if the key was tracked and was credited, ``False``
            otherwise.

        Raises:
            TypeError: If ``cost`` is not an integer, or is a ``bool``.
            ValueError: If ``cost`` is out of range for the algorithm.
        """
        self._probe._validate_cost(cost)
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return False
        entry[0].refund(cost)
        return True

    def get(self, key: str) -> RateLimiter | None:
        """Return the limiter for *key*, or ``None`` if not tracked.

        A pure lookup: it neither creates a limiter nor counts as an
        access for :meth:`cleanup` or eviction purposes.

        Args:
            key: The rate-limit key.

        Returns:
            The :class:`RateLimiter` instance, or ``None``.
        """
        with self._lock:
            entry = self._entries.get(key)
        return None if entry is None else entry[0]

    def remove(self, key: str) -> bool:
        """Stop tracking *key*, discarding its limiter state.

        Args:
            key: The rate-limit key to remove.

        Returns:
            ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> int:
        """Stop tracking every key.

        Returns:
            Number of keys that were removed.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def cleanup(self, max_idle: float = 300.0) -> int:
        """Remove limiters that have been idle for more than *max_idle* seconds.

        A key's idle time is measured from its last :meth:`allow`,
        :meth:`wait`, :meth:`peek`, :meth:`remaining` or :meth:`reset_after`
        call.

        Args:
            max_idle: Maximum idle time in seconds before a limiter is
                removed. Defaults to 300 (5 minutes). Pass ``0`` to drop
                every key that is not being accessed right now.

        Returns:
            Number of limiters removed.
        """
        now = time.monotonic()
        with self._lock:
            stale = [
                key for key, (_, last) in self._entries.items() if now - last > max_idle
            ]
            for key in stale:
                del self._entries[key]
            return len(stale)

    def keys(self) -> list[str]:
        """Return a snapshot of all currently tracked keys.

        Ordered least- to most-recently-used.

        Returns:
            List of rate-limit keys.
        """
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        """Return the number of tracked keys."""
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        """Check whether *key* is currently tracked."""
        with self._lock:
            return key in self._entries

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        with self._lock:
            count = len(self._entries)
        return f"RateLimitManager(algorithm={self._algorithm.__name__}, keys={count})"
