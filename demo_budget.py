"""Public-demo budget meter + model pin (active only when DEMO_MODE is set).

A visitor demo must (a) be forced onto the cheapest models and (b) never exceed
an API spend budget. The hard part is that BYO-WIKI spends money through several
paths — raw provider SDKs (summaries, KG extraction, ICL answers, embeddings),
the LangChain factory (rerank, RAG answers, ingestion, skill phases), and the
embeddings API. A missed path = unbounded spend, so this module gives ONE shared
ledger that every path funnels into:

  - Raw SDK clients are wrapped by ``config.traced_*`` -> in demo we return a
    metering proxy (``meter_client``) that reads ``resp.usage`` and records cost.
  - The LangChain factory ``providers.build_chat_model`` attaches
    ``langchain_callback()`` so every ``.invoke()`` is checked + recorded.
  - Embeddings flow through ``config.traced_openai`` too, so they're covered.

Everything here is a NO-OP unless ``DEMO_MODE`` is truthy, so the owner's normal
app is completely unaffected. Cost is tracked per visitor key (set per request)
AND globally, both refreshed on a rolling window. Pricing is intentionally
CONSERVATIVE (rounded up) — a budget guardrail should over-estimate, never
under-estimate, and any unknown model is charged at a high default so nothing
that escapes the model pin can spend for free.
"""

from __future__ import annotations

import os
import threading
import time
from contextvars import ContextVar

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


def demo_enabled() -> bool:
    """True when the process is running as the public demo."""
    return _flag("DEMO_MODE")


# --- pinned (cheapest) models, role-based; all env-overridable ---------------
def general_model() -> tuple[str, str]:
    """(provider, model) for general tasks: summaries, Q&A, KG extraction.

    gemini-2.5-flash is the cheap, real, stable flash model on Google's
    OpenAI-compatible endpoint (gemini-3.5-flash does NOT exist there and returns
    empty). Override with DEMO_MODEL."""
    return (os.environ.get("DEMO_GENERAL_PROVIDER", "gemini").strip(),
            os.environ.get("DEMO_MODEL", "gemini-2.5-flash").strip())


def code_model() -> tuple[str, str]:
    """(provider, model) for code / skill generation (Qwen coder)."""
    return (os.environ.get("DEMO_CODE_PROVIDER", "qwen").strip(),
            os.environ.get("DEMO_CODE_MODEL", "qwen3-coder-next").strip())


def max_tokens_cap() -> int:
    return int(os.environ.get("DEMO_MAX_TOKENS", "1024"))


# --- budget knobs ------------------------------------------------------------
def _global_cap() -> float:
    return float(os.environ.get("DEMO_GLOBAL_USD", "5.0"))


def _visitor_cap() -> float:
    return float(os.environ.get("DEMO_VISITOR_USD", "0.05"))


def _window_sec() -> float:
    return float(os.environ.get("DEMO_WINDOW_SEC", "86400"))  # daily reset


# Conservative $/1M (input, output). Rounded UP vs published rates so the cap
# trips early rather than late. Unknown models hit _DEFAULT_PRICE (high).
_DEFAULT_PRICE = (2.0, 10.0)
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.10, 0.40),
    "gemini-flash-latest": (0.10, 0.40),
    "gemini-3-flash-preview": (0.20, 0.80),
    "gemini-2.5-pro": (1.50, 6.00),
    "qwen3-coder-next": (1.00, 5.00),
    "qwen3-coder-plus": (1.00, 5.00),
    "qwen3-coder-flash": (0.30, 1.20),
    "qwen-plus": (0.50, 1.50),
    "deepseek-chat": (0.20, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o-mini": (0.30, 1.20),
    "text-embedding-3-small": (0.05, 0.0),
    "text-embedding-3-large": (0.20, 0.0),
}


def _price(model: str) -> tuple[float, float]:
    m = (model or "").strip()
    if m in _PRICES:
        return _PRICES[m]
    for k, v in _PRICES.items():  # prefix match (handles dated/suffixed ids)
        if m.startswith(k) or k.startswith(m):
            return v
    return _DEFAULT_PRICE


def cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    pin, pout = _price(model)
    return (max(0, in_tokens) / 1e6) * pin + (max(0, out_tokens) / 1e6) * pout


# --- shared ledger -----------------------------------------------------------
class BudgetExceededError(RuntimeError):
    """Raised when the demo's global or per-visitor budget is exhausted."""


_lock = threading.Lock()
_window_start = time.time()
_global_spent = 0.0
_key_spent: dict[str, float] = {}

# Per-request visitor identity (IP/session); set by app.before_request in demo.
_KEY: ContextVar[str] = ContextVar("demo_budget_key", default="global")


def set_key(key: str) -> None:
    _KEY.set(key or "global")


def get_key() -> str:
    return _KEY.get()


def _roll_window_locked() -> None:
    global _window_start, _global_spent, _key_spent
    if time.time() - _window_start >= _window_sec():
        _window_start = time.time()
        _global_spent = 0.0
        _key_spent = {}


def check_budget() -> None:
    """Raise BudgetExceededError if the current visitor's slice or the global
    budget is exhausted. No-op outside demo mode."""
    if not demo_enabled():
        return
    key = get_key()
    with _lock:
        _roll_window_locked()
        if _global_spent >= _global_cap():
            raise BudgetExceededError(
                "The demo's shared daily budget is used up — please try again later.")
        if _key_spent.get(key, 0.0) >= _visitor_cap():
            raise BudgetExceededError(
                "You've reached this demo session's usage limit. "
                "Thanks for trying it — come back tomorrow or run your own instance.")


def record_usage(model: str, in_tokens: int, out_tokens: int) -> None:
    """Charge the current visitor + the global pool for one model call."""
    if not demo_enabled():
        return
    c = cost_usd(model, in_tokens, out_tokens)
    if c <= 0:
        return
    key = get_key()
    with _lock:
        _roll_window_locked()
        global _global_spent
        _global_spent += c
        _key_spent[key] = _key_spent.get(key, 0.0) + c


def record_response(kind: str, model: str | None, resp) -> None:
    """Extract token usage from a raw-SDK response and record it.

    kind: 'anthropic' | 'openai-chat' | 'openai-embed'. On any parse failure we
    charge a conservative flat estimate so a malformed-usage path can't be
    spammed for free."""
    if not demo_enabled():
        return
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            return record_usage(model or "", 2000, 2000)  # conservative
        if kind == "anthropic":
            in_t = ((getattr(u, "input_tokens", 0) or 0)
                    + (getattr(u, "cache_creation_input_tokens", 0) or 0)
                    + (getattr(u, "cache_read_input_tokens", 0) or 0))
            out_t = getattr(u, "output_tokens", 0) or 0
        elif kind == "openai-embed":
            in_t = getattr(u, "prompt_tokens", 0) or 0
            out_t = 0
        else:  # openai-chat
            in_t = getattr(u, "prompt_tokens", 0) or 0
            out_t = getattr(u, "completion_tokens", 0) or 0
        record_usage(model or "", in_t, out_t)
    except Exception:  # noqa: BLE001
        record_usage(model or "", 2000, 2000)


def status(key: str | None = None) -> dict:
    """Budget snapshot for /api/demo-status (frontend shows remaining)."""
    if not demo_enabled():
        return {"demo": False}
    k = key or get_key()
    with _lock:
        _roll_window_locked()
        gcap, vcap = _global_cap(), _visitor_cap()
        return {
            "demo": True,
            "visitor_remaining_usd": round(max(0.0, vcap - _key_spent.get(k, 0.0)), 6),
            "visitor_cap_usd": vcap,
            "global_remaining_usd": round(max(0.0, gcap - _global_spent), 6),
            "global_cap_usd": gcap,
            "window_resets_in_sec": int(max(0, _window_sec() - (time.time() - _window_start))),
            "general_model": general_model()[1],
            "code_model": code_model()[1],
        }


# --- metering proxy for raw provider SDK clients -----------------------------
# config.traced_* returns meter_client(client, family) in demo, so every
# client.chat.completions.create / client.embeddings.create / client.messages.create
# is checked + recorded. The proxy is transparent for everything else.
_SUBNS = {"chat", "completions", "embeddings", "messages", "beta", "responses"}


class _Meter:
    def __init__(self, target, family: str, kind: str | None = None):
        object.__setattr__(self, "_t", target)
        object.__setattr__(self, "_family", family)   # 'openai' | 'anthropic'
        object.__setattr__(self, "_kind", kind)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_t"), name)
        family = object.__getattribute__(self, "_family")
        kind = object.__getattribute__(self, "_kind")
        if name == "embeddings":
            kind = "openai-embed"
        elif name in ("chat", "completions"):
            kind = kind or "openai-chat"
        elif name == "messages":
            kind = "anthropic"
        if name == "create" and callable(attr):
            return _wrap_create(attr, kind or family)
        if name in _SUBNS:
            return _Meter(attr, family, kind)
        return attr


def _wrap_create(fn, kind: str):
    def wrapper(*args, **kwargs):
        check_budget()
        resp = fn(*args, **kwargs)
        record_response(kind, kwargs.get("model"), resp)
        return resp
    return wrapper


def meter_client(client, family: str):
    """Wrap a raw OpenAI/Anthropic SDK client so its create() calls are metered.
    family: 'openai' (chat + embeddings) or 'anthropic'."""
    return _Meter(client, family)


# --- LangChain path: callback attached by providers.build_chat_model ---------
def langchain_callback():
    """A BaseCallbackHandler that checks budget before each LLM call and records
    usage after. Returns None if langchain isn't importable."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:  # noqa: BLE001
        return None

    class _DemoUsage(BaseCallbackHandler):
        raise_error = True  # propagate BudgetExceededError so the call aborts

        def on_llm_start(self, serialized, prompts, **kw):
            check_budget()

        def on_chat_model_start(self, serialized, messages, **kw):
            check_budget()

        def on_llm_end(self, response, **kw):
            try:
                model, in_t, out_t = _usage_from_llm_result(response)
                record_usage(model, in_t, out_t)
            except Exception:  # noqa: BLE001
                record_usage("", 2000, 2000)

    return _DemoUsage()


def _usage_from_llm_result(result) -> tuple[str, int, int]:
    """Pull (model, input_tokens, output_tokens) out of a LangChain LLMResult."""
    model, in_t, out_t = "", 0, 0
    lo = getattr(result, "llm_output", None) or {}
    if isinstance(lo, dict):
        model = lo.get("model_name") or lo.get("model") or ""
        tu = lo.get("token_usage") or lo.get("usage") or {}
        if isinstance(tu, dict):
            in_t = tu.get("prompt_tokens") or tu.get("input_tokens") or 0
            out_t = tu.get("completion_tokens") or tu.get("output_tokens") or 0
    if not (in_t or out_t):  # chat models carry usage_metadata on the message
        try:
            gen = result.generations[0][0]
            msg = getattr(gen, "message", None)
            um = getattr(msg, "usage_metadata", None) or {}
            in_t = um.get("input_tokens", 0) or 0
            out_t = um.get("output_tokens", 0) or 0
            if not model:
                model = (getattr(msg, "response_metadata", {}) or {}).get("model_name", "")
        except Exception:  # noqa: BLE001
            pass
    return model or "", int(in_t or 0), int(out_t or 0)
