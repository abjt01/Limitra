"""Tests for the RateLimitResult value object.

``RateLimitResult`` is frozen, hashable, and compares across all five
fields — these tests pin that contract down, since results are handed to
callers and commonly stuffed into sets, dict keys and assertions.
"""

from __future__ import annotations

import math

import pytest

from limitra import FixedWindow, RateLimitResult, TokenBucket

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def make_result(**overrides: object) -> RateLimitResult:
    """Build a result with sensible defaults, overriding named fields."""
    fields: dict[str, object] = {
        "allowed": True,
        "remaining": 7,
        "limit": 10,
        "reset_after": 1.5,
        "retry_after": 0.0,
    }
    fields.update(overrides)
    return RateLimitResult(**fields)  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# Equality
# ------------------------------------------------------------------ #


def test_equal_when_all_fields_match() -> None:
    """Two results built from identical field values compare equal."""
    assert make_result() == make_result()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed", False),
        ("remaining", 6),
        ("limit", 11),
        ("reset_after", 99.0),
        ("retry_after", 42.0),
    ],
)
def test_not_equal_when_any_field_differs(field: str, value: object) -> None:
    """Every field participates in equality, including the timing fields."""
    assert make_result() != make_result(**{field: value})


def test_not_equal_to_other_types() -> None:
    """Comparing against an unrelated type is False, not an error."""
    assert make_result() != "not a result"
    assert make_result() is not None


# ------------------------------------------------------------------ #
# Hashing and immutability
# ------------------------------------------------------------------ #


def test_is_hashable() -> None:
    """Results can be placed in sets and used as dict keys."""
    a, b = make_result(), make_result()
    different = make_result(remaining=0)
    assert hash(a) == hash(b)
    assert len({a, b, different}) == 2
    assert {a: "ok"}[b] == "ok"


def test_is_frozen() -> None:
    """Fields cannot be reassigned after construction."""
    result = make_result()
    with pytest.raises(AttributeError):
        result.allowed = False  # type: ignore[misc]


def test_has_no_dict() -> None:
    """__slots__ is in effect, so results stay cheap."""
    with pytest.raises(AttributeError):
        make_result().__dict__  # noqa: B018


# ------------------------------------------------------------------ #
# Truthiness and repr
# ------------------------------------------------------------------ #


def test_bool_mirrors_allowed() -> None:
    """``if result:`` is equivalent to ``if result.allowed:``."""
    assert bool(make_result(allowed=True)) is True
    assert bool(make_result(allowed=False)) is False


def test_repr_contains_every_field() -> None:
    """Repr is debuggable — it names all five fields."""
    text = repr(make_result())
    for field in ("allowed", "remaining", "limit", "reset_after", "retry_after"):
        assert field in text


# ------------------------------------------------------------------ #
# HTTP headers
# ------------------------------------------------------------------ #


def test_headers_when_allowed_omit_retry_after() -> None:
    """An allowed request gets no Retry-After header."""
    headers = make_result(allowed=True, remaining=7, limit=10).as_headers()
    assert headers == {
        "RateLimit-Limit": "10",
        "RateLimit-Remaining": "7",
        "RateLimit-Reset": "2",
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "7",
        "X-RateLimit-Reset": "2",
    }


def test_headers_emit_both_naming_conventions_with_equal_values() -> None:
    """Clients disagree on the prefix, so both are sent and must agree."""
    headers = make_result(allowed=False, remaining=0, limit=10).as_headers()
    for field in ("Limit", "Remaining", "Reset"):
        assert headers[f"RateLimit-{field}"] == headers[f"X-RateLimit-{field}"]


def test_reset_header_is_seconds_not_a_timestamp() -> None:
    """X-RateLimit-Reset is a delta, so it must stay small, not epoch-sized."""
    headers = make_result(reset_after=30.0).as_headers()
    assert headers["X-RateLimit-Reset"] == "30"


def test_headers_when_denied_include_retry_after() -> None:
    """A denied request carries Retry-After."""
    headers = make_result(allowed=False, remaining=0, retry_after=1.2).as_headers()
    assert headers["Retry-After"] == "2"
    assert headers["X-RateLimit-Remaining"] == "0"


def test_headers_round_retry_after_up() -> None:
    """Retry-After rounds up, so an obedient client never retries too early."""
    result = make_result(allowed=False, reset_after=0.1, retry_after=0.1)
    headers = result.as_headers()
    assert headers["Retry-After"] == "1"
    assert headers["X-RateLimit-Reset"] == "1"


def test_headers_are_all_strings() -> None:
    """Header values are strings, ready to hand to a web framework."""
    headers = make_result(allowed=False).as_headers()
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items())


def test_headers_from_a_real_limiter() -> None:
    """Headers derived from a live limiter carry honest values."""
    limiter = FixedWindow(limit=3, window=60.0)
    for _ in range(3):
        limiter.allow()
    denied = limiter.allow()

    headers = denied.as_headers()
    assert headers["X-RateLimit-Limit"] == "3"
    assert headers["X-RateLimit-Remaining"] == "0"
    assert 0 < int(headers["Retry-After"]) <= 60


def test_retry_after_header_never_under_reports() -> None:
    """The rounded header is never shorter than the true wait."""
    limiter = TokenBucket(rate=3.0, capacity=2)
    limiter.allow(cost=2)
    denied = limiter.allow()
    assert denied.allowed is False
    assert int(denied.as_headers()["Retry-After"]) >= math.floor(denied.retry_after)
