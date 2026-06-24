"""Environment / config helpers: optional .env loading and tracing status.

Loading a local ``.env`` lets users keep their LangSmith and model API keys in
one place so both the web app (``app.py``) and the CLI (``runner.py``) pick
them up. LangChain/deepagents emit traces automatically when the ``LANGSMITH_*``
variables are present, so enabling tracing is purely a matter of configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def load_env(path: str | os.PathLike = ".env", *, override: bool = False) -> bool:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Dependency-free: tolerates ``export`` prefixes, quotes, and ``#`` comments.
    Existing environment variables win unless ``override=True``. Returns whether
    a file was found.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = val
    except OSError:
        return False
    return True


def tracing_enabled() -> bool:
    """True when LangSmith tracing is switched on via env vars."""
    return (
        os.environ.get("LANGSMITH_TRACING", "").lower() in _TRUTHY
        or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in _TRUTHY
    )


_resolved_project: str | None = None


def ensure_tracing_project() -> str | None:
    """Resolve LANGSMITH_PROJECT_ID -> current project name and set LANGSMITH_PROJECT.

    The LangSmith tracer addresses projects by name, but names can be renamed;
    referencing the stable project ID and resolving it at runtime keeps tracing
    pointed at the right project. Best-effort: falls back to any existing
    LANGSMITH_PROJECT when offline or unset. Cached after first success.
    """
    global _resolved_project
    pid = os.environ.get("LANGSMITH_PROJECT_ID")
    if not pid:
        return os.environ.get("LANGSMITH_PROJECT")
    if _resolved_project:
        return _resolved_project
    if not os.environ.get("LANGSMITH_API_KEY"):
        return os.environ.get("LANGSMITH_PROJECT")
    try:
        from langsmith import Client
        name = Client().read_project(project_id=pid).name
        if name:
            os.environ["LANGSMITH_PROJECT"] = name
            _resolved_project = name
            return name
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("LANGSMITH_PROJECT")


def _demo_meter(client, family: str):
    """In DEMO_MODE, wrap a raw SDK client with the spend meter (outermost) so
    every create() is budget-checked + recorded. No-op otherwise. This single
    hook covers every raw-SDK path that flows through traced_* (summaries, KG
    extraction, ICL answers, and embeddings)."""
    try:
        import demo_budget
        if demo_budget.demo_enabled():
            return demo_budget.meter_client(client, family)
    except Exception:  # noqa: BLE001
        pass
    return client


def traced_openai(client):
    """Wrap an OpenAI(-compatible) SDK client so its calls trace to LangSmith
    (latency, token usage -> cost, errors) and, in DEMO_MODE, are budget-metered.
    Best-effort, never breaks a call."""
    if tracing_enabled():
        try:
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
        except Exception:  # noqa: BLE001
            pass
    return _demo_meter(client, "openai")


def traced_anthropic(client):
    """LangSmith-tracing + DEMO_MODE budget metering for an Anthropic SDK client
    (see traced_openai)."""
    if tracing_enabled():
        try:
            from langsmith.wrappers import wrap_anthropic
            client = wrap_anthropic(client)
        except Exception:  # noqa: BLE001
            pass
    return _demo_meter(client, "anthropic")


def tracing_status() -> dict:
    """Summarize the current LangSmith tracing configuration."""
    return {
        "enabled": tracing_enabled(),
        "project": (
            os.environ.get("LANGSMITH_PROJECT")
            or os.environ.get("LANGCHAIN_PROJECT")
            or "default"
        ),
        "project_id": os.environ.get("LANGSMITH_PROJECT_ID"),
        "endpoint": (
            os.environ.get("LANGSMITH_ENDPOINT")
            or os.environ.get("LANGCHAIN_ENDPOINT")
            or "https://api.smith.langchain.com"
        ),
        "api_key_set": bool(
            os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        ),
    }
