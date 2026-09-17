# Changelog

All notable changes to this project are documented here. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — unreleased

### Fixed

- **Both buckets accepted a NaN or infinite `rate` and then admitted
  everything.** `rate <= 0` is `False` for NaN, so the guard let it through;
  `min(capacity, tokens + elapsed * nan)` then pinned the bucket permanently
  full while `remaining()` and `reset_after()` reported a healthy limiter. A
  capacity-10 bucket admitted 10,000 units instantly. `rate` and `window`
  must now be finite, and `capacity` and `limit` must be whole numbers.
- **`SlidingWindow.retry_after` was up to a full window too short.** It
  returned the time to the next window boundary rather than the moment the
  weighted count actually decays far enough, so a client that obeyed it was
  denied again and settled at roughly half the configured rate. It is now
  solved from the decay curve; obeying it to the microsecond gets you in.
- **`FixedWindow` re-anchored its window to whoever looked at it.** Any
  `peek()`, `remaining()` or `reset_after()` call could move the boundary,
  so a metrics scrape changed the `Retry-After` other callers were given.
  Boundaries now step in whole windows from construction and depend only on
  elapsed time.
- **`SlidingWindow` could clear the previous window without moving the
  boundary**, admitting twice the limit, when float division floored the
  elapsed window count to zero.
- **`SlidingLog` rebuilt its entire log on every call.** Pruning now deletes
  the expired prefix in place, which takes each call from `O(n)` to
  amortised `O(1)` and a full window from `O(n^2)` to `O(n)`. At 100,000
  operations it went from needing a 10x smaller benchmark to ~865,000
  ops/sec.
- **`SlidingLog.reset_after()` reported the oldest entry expiring, not full
  capacity**, so it could emit a `Retry-After` larger than its own
  `X-RateLimit-Reset`.
- **`SlidingLog.allow()` and `peek()` raised `IndexError`** when `cost` was
  greater than `limit`.
- **`peek()` disagreed with `allow()` on `SlidingWindow` and `SlidingLog`**,
  reporting the `remaining` from before the request rather than after it.
  All five algorithms now agree exactly.
- **`reset_after()` did not mean what the base class documented.**
  `FixedWindow` and `SlidingWindow` returned a full window on an idle,
  completely unused limiter. It is now `0.0` exactly when the limiter is at
  full capacity, on every algorithm.
- **A `retry_after` of exactly the right length could be one ULP short**, so
  waiting precisely as instructed was denied again.
- **`SlidingLog` kept entries whose age was exactly `window`**, for the same
  reason; the window is now half-open.
- **`RateLimitResult.__eq__` compared only three of five fields**, so
  results with different `reset_after` or `retry_after` compared equal.
  Defining `__eq__` without `__hash__` had also made the class unhashable.
- **`RateLimitManager.peek()` created a limiter without recording an access
  time**, so a key first seen through `peek()` was never reclaimed by
  `cleanup()` and leaked for the life of the manager.
- **`RateLimitManager.cleanup()` raised `KeyError`** if its limiter map and
  its access-time map fell out of step. They are now a single mapping.
- **An invalid request could evict a valid key.** The manager validated
  `cost` only after creating the key, so a rejected request still displaced
  a real user under `max_keys`.
- **The package could not be built at all.** `requires =
  ["setuptools>=61.0"]` combined with the PEP 639 `license = "MIT"`
  expression, which needs setuptools 77.0.3; anything older failed with a
  `project.license` configuration error.
- **The sdist shipped an unrunnable test suite**, omitting `conftest.py` and
  `tests/__init__.py`. A `MANIFEST.in` now makes its contents explicit.
- The ruff config both selected and ignored `TCH`, disabling
  flake8-type-checking entirely while warning on every run.

### Changed

- A `cost` greater than the limiter's `limit` now raises `ValueError` rather
  than returning a denial. Such a request can never succeed, so the old
  behaviour handed back a `retry_after` that would never come true.
- `cost=True` now raises `TypeError`. `bool` is a subclass of `int`, so a
  boolean passed by mistake used to be silently treated as `cost=1`.
- `RateLimitResult` is frozen. It is a snapshot of a decision, and mutating
  one was never meaningful.
- `as_headers()` emits both the `RateLimit-*` and `X-RateLimit-*` spellings,
  since clients disagree about which to read, and documents that every value
  is seconds from now rather than a Unix timestamp.
- The manager's read-only methods — `peek`, `remaining`, `reset_after`,
  `get` — no longer create a key. Only `allow`, `wait` and `wait_async` do.
- `RateLimitManager` builds one limiter at construction, so a mistyped
  keyword argument fails immediately instead of inside the first request.
- `LeakyBucket`'s documentation no longer claims a behaviour it does not
  have. As admission control it is the token bucket's dual and admits
  exactly the same requests; the docs now say so and explain how to get
  genuinely paced output.
- `FixedWindow` documents that it can admit `2 * limit` across a boundary.
- The version is defined once, in `limitra.__version__`, and read from there
  at build time.

### Added

- `RateLimiter.wait(cost=1, timeout=None)` blocks until capacity is
  available, sleeping for the limiter's own `retry_after` instead of
  busy-looping, and never holding the lock while it sleeps.
- `RateLimiter.wait_async(...)`, the asyncio counterpart, which yields to
  the event loop rather than blocking it.
- `RateLimitResult.as_headers()` for HTTP responses.
- Limiters expose the configuration they were built with: `rate` and
  `capacity` on the buckets, `limit` and `window` on the windows, and
  `limit` on every algorithm.
- `RateLimitManager` gained `wait`, `wait_async`, `remaining(key)`,
  `reset_after(key)`, `reset(key)` and `clear()`, plus a `max_keys` bound
  that evicts the least-recently-used key — preferring one already at full
  capacity, so a flood of fresh keys is less able to clear someone else's
  limit.
- A CI workflow running lint, formatting, type checks, doctests and the test
  suite across Python 3.10 through 3.14 on Linux, macOS and Windows, and
  verifying that the built wheel installs and imports. The release workflow
  now also runs `twine check` and refuses to publish if the tag and the
  package version disagree.

### Testing

- The concurrency tests asserted only an upper bound on admissions, so all
  seven passed against a limiter that admitted nothing. They now assert the
  exact count, and the deadlock test enforces its deadline instead of
  raising an assertion in a timer thread where nothing could see it.
- Benchmarks are opt-in (`pytest -m benchmark`) rather than gating every
  run on wall-clock timings, and one of them now compares `SlidingLog`
  against itself at two log sizes, which catches the quadratic pruning
  regression regardless of how fast the machine is.
- Timing-dependent tests run on a controllable clock instead of `sleep`,
  making the decay and retry maths exact.
- Coverage is 100% of statements and branches.

## [0.1.0] — 2026-07-12

Initial release: `TokenBucket`, `LeakyBucket`, `FixedWindow`,
`SlidingWindow`, `SlidingLog` and `RateLimitManager` behind one interface.
