# abjt-limitter

Rate limiting for Python. No dependencies, no frameworks.

```bash
pip install abjt-limitter
```

## What's in it

Five algorithms, all with the same interface:

- **TokenBucket** — allows bursts, refills over time
- **LeakyBucket** — smooth constant rate
- **FixedWindow** — counter resets every N seconds
- **SlidingWindow** — smoother version of fixed window
- **SlidingLog** — exact tracking, most accurate

## Usage

```python
from limitra import TokenBucket

limiter = TokenBucket(rate=10, capacity=100)

result = limiter.allow()

if result.allowed:
    print("ok")
else:
    print(f"slow down, retry in {result.retry_after:.1f}s")
```

Same API works for every algorithm:

```python
from limitra import LeakyBucket, FixedWindow, SlidingWindow, SlidingLog

# just swap the class, nothing else changes
limiter = FixedWindow(limit=100, window=60)
result = limiter.allow()
```

## Result fields

Every `allow()` call returns the same object:

| field | type | what it is |
|---|---|---|
| `allowed` | bool | whether the request went through |
| `remaining` | int | how many requests are left |
| `limit` | int | the max |
| `reset_after` | float | seconds until full reset |
| `retry_after` | float | seconds to wait if denied |

## Multiple keys

Use `RateLimitManager` if you need per-user or per-IP limits:

```python
from limitra import RateLimitManager, TokenBucket

manager = RateLimitManager(TokenBucket, rate=10, capacity=50)

result = manager.allow("user_123")
result = manager.allow("user_456")  # completely separate bucket
```

## License

MIT
