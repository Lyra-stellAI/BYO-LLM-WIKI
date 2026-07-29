"""Shared counter store behind the public demo's spend ledger and rate limiter.

On a single-process host (gunicorn ``-w 1``, the Flask dev server) an in-process
dict IS the shared state, which is what ``demo_budget`` and ``app._demo_gate``
originally assumed. On a serverless host the app is cloned per instance, so a
process-local ledger silently turns a $5 global cap into N x $5 on live API keys.
This module is the one place that knows the difference.

Two backends, chosen from the environment:

  * ``redis``  - Upstash Redis over its HTTP REST API. Picked up automatically
    from the env vars the Vercel Upstash integration injects. Counters are
    server-side and atomic, so every instance shares one budget. Uses ``requests``
    (already a dependency) rather than a Redis client, because one short-lived
    HTTPS call per gate suits serverless better than a pooled TCP connection.
  * ``memory`` - process-local dict + lock. The original behavior; what local
    runs, Render, Railway and Fly get when no REST credentials are set.

A process-local mirror is maintained in BOTH modes and every read returns
``max(remote, local)``. That is the safety property: if Redis is unreachable the
demo degrades to a per-instance cap (today's behavior) instead of either failing
every request or, worse, silently spending without a ceiling. Remote is normally
the larger value since it aggregates all instances, so the max is a no-op.

Keys carry their own expiry; nothing here needs a sweeper. Callers pass a TTL and
put any rotating component (window id, epoch minute) in the key itself.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Iterable, Sequence

# Credential pairs in priority order. DEMO_REDIS_* is the explicit override;
# KV_REST_API_* and UPSTASH_REDIS_REST_* are what Vercel's Upstash marketplace
# integration writes into the project env (the names differ by integration
# version, so accept both rather than making the operator rename them).
_REST_ENV: tuple[tuple[str, str], ...] = (
    ("DEMO_REDIS_REST_URL", "DEMO_REDIS_REST_TOKEN"),
    ("KV_REST_API_URL", "KV_REST_API_TOKEN"),
    ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"),
)

_lock = threading.Lock()
_local: dict[str, tuple[float, float]] = {}  # key -> (value, expires_at)

_last_warn = 0.0
_WARN_EVERY_SEC = 60.0


# --- configuration -----------------------------------------------------------
def _credentials() -> tuple[str, str] | None:
    for url_var, token_var in _REST_ENV:
        url = os.environ.get(url_var, "").strip().rstrip("/")
        token = os.environ.get(token_var, "").strip()
        if url and token:
            return url, token
    return None


def backend() -> str:
    """``"redis"`` when REST credentials are configured, else ``"memory"``."""
    return "redis" if _credentials() else "memory"


def _timeout() -> float:
    """Kept short: this call sits in front of every metered request, and a slow
    ledger must not become the demo's latency floor. A timeout falls back to the
    local mirror rather than failing the request."""
    return float(os.environ.get("DEMO_REDIS_TIMEOUT_SEC", "2.0"))


# --- process-local mirror ----------------------------------------------------
def _local_add(key: str, delta: float, ttl: float) -> float:
    now = time.time()
    with _lock:
        value, expires = _local.get(key, (0.0, 0.0))
        if expires <= now:
            value, expires = 0.0, now + ttl
        value += delta
        _local[key] = (value, expires)
        return value


def _local_get(key: str) -> float:
    now = time.time()
    with _lock:
        value, expires = _local.get(key, (0.0, 0.0))
        return value if expires > now else 0.0


def _local_release(key: str) -> None:
    now = time.time()
    with _lock:
        value, expires = _local.get(key, (0.0, 0.0))
        if expires > now and value > 0:
            _local[key] = (value - 1.0, expires)


def _purge_expired() -> None:
    """Drop expired mirror entries. Called on the write paths so a long-lived
    process doesn't accumulate one entry per visitor per window forever."""
    now = time.time()
    with _lock:
        for key in [k for k, (_, expires) in _local.items() if expires <= now]:
            del _local[key]


def reset() -> None:
    """Clear the process-local mirror. For tests; never called by the app."""
    with _lock:
        _local.clear()


# --- Upstash REST ------------------------------------------------------------
def _warn(exc: Exception) -> None:
    """Rate-limited stderr warning. A ledger outage must be visible in the logs —
    it silently weakens the spend cap to per-instance — but must not spam."""
    global _last_warn
    now = time.time()
    if now - _last_warn < _WARN_EVERY_SEC:
        return
    _last_warn = now
    print(f"[demo store] Redis ledger unavailable ({type(exc).__name__}: {exc}) — "
          "falling back to the process-local cap for now.", file=sys.stderr)


def _pipeline(commands: Sequence[Sequence[str]]) -> list | None:
    """Run commands through Upstash's ``/pipeline`` endpoint in one round trip.

    Returns the list of results, or ``None`` when the backend is unconfigured or
    the call failed (caller then uses the local mirror)."""
    creds = _credentials()
    if not creds or not commands:
        return None
    url, token = creds
    try:
        import requests

        resp = requests.post(
            f"{url}/pipeline",
            json=[list(c) for c in commands],
            headers={"Authorization": f"Bearer {token}"},
            timeout=_timeout(),
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            raise ValueError(f"unexpected pipeline response: {payload!r}")
        results = []
        for item in payload:
            if isinstance(item, dict):
                if item.get("error"):
                    raise RuntimeError(str(item["error"]))
                results.append(item.get("result"))
            else:
                results.append(item)
        return results
    except Exception as exc:  # noqa: BLE001  (any failure -> local mirror)
        _warn(exc)
        return None


def _as_float(raw, fallback: float) -> float:
    try:
        return fallback if raw is None else float(raw)
    except (TypeError, ValueError):
        return fallback


def _redis_float(value: float) -> str:
    """Format a float the way INCRBYFLOAT accepts it.

    Fixed-point, never exponential: ``repr(3e-05)`` is ``'3e-05'`` and Redis
    rejects that with "value is not a valid float". Per-call demo costs are
    routinely that small, so this is load-bearing, not cosmetic."""
    return f"{float(value):.12f}"


# --- public API --------------------------------------------------------------
def add_floats(items: Iterable[tuple[str, float, float]]) -> list[float]:
    """Add to several float counters in one round trip.

    ``items`` is ``(key, delta, ttl_seconds)``. Returns the new value per key,
    ``max(remote, local)``."""
    items = list(items)
    if not items:
        return []
    _purge_expired()
    locals_ = [_local_add(key, delta, ttl) for key, delta, ttl in items]
    commands: list[list[str]] = []
    for key, delta, ttl in items:
        commands.append(["INCRBYFLOAT", key, _redis_float(delta)])
        commands.append(["EXPIRE", key, str(int(ttl))])
    results = _pipeline(commands)
    if results is None or len(results) < 2 * len(items):
        return locals_
    return [max(_as_float(results[2 * i], locals_[i]), locals_[i])
            for i in range(len(items))]


def add_float(key: str, delta: float, ttl: float) -> float:
    return add_floats([(key, delta, ttl)])[0]


def get_floats(keys: Sequence[str]) -> list[float]:
    """Read several float counters in one round trip, ``max(remote, local)``."""
    keys = list(keys)
    if not keys:
        return []
    locals_ = [_local_get(k) for k in keys]
    results = _pipeline([["GET", k] for k in keys])
    if results is None or len(results) < len(keys):
        return locals_
    return [max(_as_float(results[i], locals_[i]), locals_[i]) for i in range(len(keys))]


def get_float(key: str) -> float:
    return get_floats([key])[0]


def bump(items: Iterable[tuple[str, float]]) -> list[int]:
    """Increment several integer counters by one, in one round trip.

    ``items`` is ``(key, ttl_seconds)``. Returns the new count per key."""
    items = list(items)
    if not items:
        return []
    _purge_expired()
    locals_ = [int(_local_add(key, 1.0, ttl)) for key, ttl in items]
    commands: list[list[str]] = []
    for key, ttl in items:
        commands.append(["INCR", key])
        commands.append(["EXPIRE", key, str(int(ttl))])
    results = _pipeline(commands)
    if results is None or len(results) < 2 * len(items):
        return locals_
    return [max(int(_as_float(results[2 * i], locals_[i])), locals_[i])
            for i in range(len(items))]


def release(key: str) -> None:
    """Decrement an integer counter, clamped at zero.

    The clamp matters for the in-flight gauge: an instance that dies mid-request
    never runs its decrement, and without the clamp a counter driven negative
    would quietly disable the cap for that visitor."""
    _local_release(key)
    results = _pipeline([["DECR", key]])
    if results and _as_float(results[0], 0.0) < 0:
        _pipeline([["SET", key, "0"]])
