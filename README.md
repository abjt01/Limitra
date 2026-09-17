# abjt-limitter

Rate limiting for Python. No dependencies, no frameworks.

```bash
pip install abjt-limitter
```

```python
from limitra import TokenBucket

limiter = TokenBucket(rate=10, capacity=100)

result = limiter.allow()

if result.allowed:
    print("ok")
else:
    print(f"slow down, retry in {result.retry_after:.1f}s")
```

## Picking an algorithm

Five of them, all behind the same interface:

| class | behaviour | pick it when |
|---|---|---|
| `TokenBucket` | refills steadily, lets you spend the whole bucket at once | you want to tolerate bursts |
| `LeakyBucket` | the same policy as `TokenBucket`, expressed as a queue filling and draining | you'd rather think in "how full is it" than "how much credit is left" |
| `FixedWindow` | counter resets every N seconds | you need a hard "100 per minute" cap and nothing subtler |
| `SlidingWindow` | weights the previous window as it decays | you want fixed-window cost without the boundary burst |
| `SlidingLog` | keeps a timestamp per request | you need exact limits and can afford `O(limit)` memory per key |

Two things worth knowing before you choose:

**`LeakyBucket` and `TokenBucket` admit the same requests.** As admission
control the water-level leaky bucket is the token bucket's dual —
`capacity - water` *is* the token count — so for the same `rate` and
`capacity` they accept and reject identically. The smoothing a leaky bucket
is famous for belongs to the queueing variant, which delays requests rather
than refusing them. If you want output genuinely paced, set `capacity=1` so
no burst is possible, or use `wait()` below.

**`FixedWindow` can admit `2 * limit` across a boundary.** Its counter
clears all at once, so `limit` requests at the end of one window and `limit`
at the start of the next land within one window-length of each other. Use
`SlidingWindow` or `SlidingLog` if that matters.

Swapping algorithms only changes the constructor — the windows take `limit`
and `window`, the buckets take `rate` and `capacity`, and everything after
that is identical:

```python
from limitra import SlidingWindow

limiter = SlidingWindow(limit=100, window=60)
result = limiter.allow()
```

## The result

Every `allow()` call returns the same object:

| field | type | what it is |
|---|---|---|
| `allowed` | bool | whether the request went through |
| `remaining` | int | units left after this decision |
| `limit` | int | the max |
| `reset_after` | float | seconds until back at full capacity, `0.0` if already there |
| `retry_after` | float | seconds to wait if denied, `0.0` if allowed |

It's frozen and hashable, and `if result:` means `if result.allowed:`.

`retry_after` is exact: wait that long and the next attempt gets through.
It isn't rounded up to the next window, so a client that obeys it isn't
throttled to less than the rate you configured.

For HTTP, hand it straight to your framework:

```python
result = limiter.allow()
response.headers.update(result.as_headers())
```

That sets `RateLimit-Limit`, `RateLimit-Remaining` and `RateLimit-Reset`,
plus the older `X-RateLimit-*` spellings with the same values, and
`Retry-After` when the request was denied. **Every time value is seconds
from now, not a Unix timestamp**, and is rounded up, so a client that obeys
the headers never comes back early.

## The rest of the API

```python
limiter.allow(cost=5)      # spend more than one unit
limiter.peek()             # what would allow() say? doesn't spend anything
limiter.remaining()        # units available right now
limiter.reset_after()      # seconds until back to full
limiter.reset()            # wipe it back to a fresh limiter
limiter.limit              # the largest cost this limiter could ever admit
```

`peek()` returns exactly what the next `allow()` would, including the
`remaining` you'd be left with.

`cost` must be between `1` and `limit`. Asking for more than `limit` raises
`ValueError` rather than denying you, because no amount of waiting would
make that request succeed.

Limiters also report the settings they were built with — `rate` and
`capacity` on the buckets, `limit` and `window` on the windows.

## Waiting instead of refusing

If you're the client rather than the server — pacing yourself against
someone else's API — `wait()` blocks until there's room:

```python
for item in queue:
    limiter.wait()          # sleeps until there's capacity
    send(item)

if not limiter.wait(timeout=5).allowed:
    print("gave up after 5s")
```

It sleeps for the limiter's own `retry_after` and never holds the lock
while sleeping, so other threads keep working. In asyncio use
`wait_async()`, which is identical but yields to the event loop:

```python
async def handler(request):
    await limiter.wait_async()
    return await upstream.fetch(request)
```

`allow()` and `peek()` never block, so they're safe to call from a
coroutine directly.

## Multiple keys

Use `RateLimitManager` for per-user or per-IP limits:

```python
from limitra import RateLimitManager, TokenBucket

manager = RateLimitManager(TokenBucket, rate=10, capacity=50)

manager.allow("user_123")
manager.allow("user_456")   # completely separate bucket
```

Keys are usually attacker-controlled, so the map needs a bound. Sweeping
idle keys is the safer option, because it only drops keys that have gone
quiet:

```python
manager.cleanup(max_idle=300)   # drop keys idle for 5 minutes, returns the count
```

`max_keys` caps the number of entries instead:

```python
manager = RateLimitManager(TokenBucket, max_keys=100_000, rate=10, capacity=50)
```

Be aware of the trade: when the map is full, adding a key evicts one, and
an evicted key loses its state and comes back with a full budget. Eviction
prefers keys already at full capacity, which have nothing to lose, but a
large enough flood of fresh keys will eventually drop a throttled one.
Size `max_keys` well above your expected number of active keys, and prefer
`cleanup()` when you can.

Only `allow()`, `wait()` and `wait_async()` create a key. `peek()`,
`remaining()`, `reset_after()` and `get()` answer for an unknown key
without starting to track it, so a metrics scrape can't fill the map. The
manager also has `reset`, `remove`, `clear`, `keys`, `len()` and `in`, and
its constructor arguments are checked when you build it rather than on the
first request.

## Threads and processes

Every limiter is thread-safe — a `threading.Lock` guards each one, and
timing uses `time.monotonic()`, so changing the system clock can't move a
window. The suite runs on free-threaded Python (3.14t) as well, where the
GIL isn't there to paper over a missing lock.

State lives in memory, in one process. Run four gunicorn workers and you
get four independent limiters, so the effective limit is four times what
you configured. If you need one limit across processes or machines, you
need a shared backend; this library doesn't provide one.

## Development

```bash
pip install -e ".[dev]"

pytest                                  # the suite
pytest -m benchmark -s                  # throughput and scaling, opt-in
pytest --doctest-modules src/limitra    # the examples in the docstrings
ruff check . && ruff format --check .
mypy src/limitra
```

Ships with type hints and a `py.typed` marker. Requires Python 3.10+.
Changes are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
