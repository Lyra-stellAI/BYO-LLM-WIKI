"""Tests for public DEMO_MODE: model pin, budget metering on every spend path,
feature gates, rate limiting, and cloud-storage isolation.

Fully offline — no network, no API keys needed. DEMO_MODE is toggled per test;
the conftest already forces local backends + a temp data dir, so importing app
here is safe. Run directly or under pytest.
"""

import importlib
import os

import demo_budget

# A full env snapshot is restored on teardown. This matters because route tests
# reload app.py, which runs config.load_env() and would otherwise pull the real
# OPENAI_API_KEY (and other keys) from .env into the process — leaking into later
# test files (e.g. test_memory asserts embeddings are off). Snapshot → restore
# makes every demo test hermetic regardless of run order.
_ENV_SNAPSHOT: dict | None = None


def _demo_on(**env):
    global _ENV_SNAPSHOT
    _ENV_SNAPSHOT = dict(os.environ)
    os.environ["DEMO_MODE"] = "1"
    # Force the isolation-safe local config the demo requires, so reloading app
    # never trips the cloud-storage guard no matter what a prior test left set.
    os.environ["CACHED_STORE_BACKEND"] = "local"
    os.environ["MEMORY_BACKEND"] = "local"
    for v in ("SUPABASE_DB_URL", "CACHED_STORE_DB_URL", "MEMORY_DB_URL", "SKILL_GRAPH_DB_URL"):
        os.environ[v] = ""
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    for k, v in env.items():
        os.environ[k] = str(v)
    importlib.reload(demo_budget)


def _demo_off():
    # Restore the exact pre-test environment and reset the ledger. We do NOT
    # reload app/config/providers here: they all read DEMO_MODE live (per request
    # / per call), so unsetting the env is enough — and reloading app would re-run
    # config.load_env() and re-pollute the env after we just restored it.
    if _ENV_SNAPSHOT is not None:
        os.environ.clear()
        os.environ.update(_ENV_SNAPSHOT)
    else:
        os.environ.pop("DEMO_MODE", None)
    importlib.reload(demo_budget)


# --- pricing + ledger --------------------------------------------------------
def test_unknown_model_charged_conservatively():
    _demo_on()
    try:
        # unknown model must NOT be free — defends against escaping the pin
        assert demo_budget.cost_usd("some-expensive-unknown", 1_000_000, 0) >= 1.0
    finally:
        _demo_off()


def test_per_visitor_cap_isolated():
    _demo_on(DEMO_VISITOR_USD="0.001", DEMO_GLOBAL_USD="100")
    try:
        demo_budget.set_key("A")
        demo_budget.check_budget()
        demo_budget.record_usage("qwen3-coder-next", 500, 500)  # > $0.001
        raised = False
        try:
            demo_budget.check_budget()
        except demo_budget.BudgetExceededError:
            raised = True
        assert raised, "visitor A should be capped"
        demo_budget.set_key("B")
        demo_budget.check_budget()  # B unaffected
    finally:
        _demo_off()


def test_global_cap_blocks_all():
    _demo_on(DEMO_VISITOR_USD="100", DEMO_GLOBAL_USD="0.001")
    try:
        demo_budget.set_key("A")
        demo_budget.record_usage("qwen3-coder-next", 500, 500)
        for k in ("A", "B", "C"):
            demo_budget.set_key(k)
            raised = False
            try:
                demo_budget.check_budget()
            except demo_budget.BudgetExceededError:
                raised = True
            assert raised, f"{k} should be blocked by global cap"
    finally:
        _demo_off()


def test_record_response_reads_each_usage_shape():
    _demo_on(DEMO_VISITOR_USD="100", DEMO_GLOBAL_USD="100")
    try:
        class _U:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class _R:
            def __init__(self, u):
                self.usage = u

        demo_budget.set_key("shapes")
        base = demo_budget.status("shapes")["visitor_remaining_usd"]
        demo_budget.record_response("openai-chat", "gemini-3.5-flash",
                                    _R(_U(prompt_tokens=1000, completion_tokens=1000)))
        demo_budget.record_response("anthropic", "claude-haiku-4-5",
                                    _R(_U(input_tokens=1000, output_tokens=1000,
                                          cache_creation_input_tokens=0, cache_read_input_tokens=0)))
        demo_budget.record_response("openai-embed", "text-embedding-3-small",
                                    _R(_U(prompt_tokens=1000)))
        after = demo_budget.status("shapes")["visitor_remaining_usd"]
        assert after < base, "all three usage shapes must charge"
    finally:
        _demo_off()


def test_missing_usage_still_charges():
    _demo_on(DEMO_VISITOR_USD="100", DEMO_GLOBAL_USD="100")
    try:
        class _R:  # response with no .usage
            usage = None
        demo_budget.set_key("nousage")
        b = demo_budget.status("nousage")["visitor_remaining_usd"]
        demo_budget.record_response("openai-chat", "gemini-3.5-flash", _R())
        assert demo_budget.status("nousage")["visitor_remaining_usd"] < b
    finally:
        _demo_off()


# --- SDK metering proxy ------------------------------------------------------
def test_meter_proxy_charges_chat_and_embeddings():
    _demo_on(DEMO_VISITOR_USD="100", DEMO_GLOBAL_USD="100")
    try:
        class _U:
            prompt_tokens = 1000; completion_tokens = 1000
        class _Resp:
            usage = _U()
        class _Create:
            def create(self, **kw):
                return _Resp()
        class _Chat:
            completions = _Create()
        class _FakeOpenAI:
            chat = _Chat()
            embeddings = _Create()

        demo_budget.set_key("proxy")
        mc = demo_budget.meter_client(_FakeOpenAI(), "openai")
        b = demo_budget.status("proxy")["visitor_remaining_usd"]
        mc.chat.completions.create(model="gemini-3.5-flash", messages=[])
        m = demo_budget.status("proxy")["visitor_remaining_usd"]
        mc.embeddings.create(model="text-embedding-3-small", input=["x"])
        e = demo_budget.status("proxy")["visitor_remaining_usd"]
        assert b > m > e, "chat and embeddings must both be metered through the proxy"
    finally:
        _demo_off()


def test_config_traced_wraps_with_meter_in_demo():
    _demo_on()
    try:
        import config  # traced_* check demo live; no reload needed

        class _Client:
            pass
        wrapped = config.traced_openai(_Client())
        assert isinstance(wrapped, demo_budget._Meter), "traced_openai must meter in demo"
        wrapped_a = config.traced_anthropic(_Client())
        assert isinstance(wrapped_a, demo_budget._Meter)
    finally:
        _demo_off()


# --- model pin ---------------------------------------------------------------
def test_resolvers_pin_cheap_models_ignoring_request():
    _demo_on()
    try:
        import providers  # resolvers check demo live; no reload needed
        assert providers.resolve_provider_model("anthropic", "claude-opus-4-8") == ("gemini", "gemini-3.5-flash")
        assert providers.skill_generator("anthropic", "claude-opus-4-8") == ("qwen", "qwen3-coder-next")
        assert providers.judge_panel("gemini") == []
        jp, jm, cross = providers.resolve_judge("gemini", "gemini-3.5-flash", "openai", "gpt-4o")
        assert (jp, cross) == ("gemini", False)
    finally:
        _demo_off()


# --- app routes: gates, isolation, status ------------------------------------
def _client(**env):
    # app (and config/providers) check DEMO_MODE live per request, so the
    # already-imported module behaves as demo once DEMO_MODE is set — no reload
    # (reloading would re-run config.load_env() and pollute the env).
    _demo_on(**env)
    import app
    return app.app.test_client(), app


def test_blocked_features_return_403():
    c, app = _client(DEMO_RPM="1000", DEMO_GLOBAL_RPM="100000")
    try:
        for path in ("/api/agent/ask", "/api/agent/maintain", "/api/rag/eval",
                     "/api/rag/crossdoc", "/api/rag/ragas", "/api/mcp/write",
                     "/api/skill/x/eval", "/api/skill/graph/build"):
            assert c.post(path, json={}).status_code == 403, path
        # destructive DELETEs blocked
        assert c.delete("/api/cache/item/x").status_code == 403
        assert c.delete("/api/memory/x").status_code == 403
        # allowed features are not 403
        assert c.post("/api/rag/ask", json={}).status_code != 403
        assert c.post("/api/skill/build", json={}).status_code != 403
    finally:
        _demo_off()


def test_demo_status_reports_pins_and_features():
    c, app = _client()
    try:
        st = c.get("/api/demo-status").get_json()
        assert st["demo"] is True
        assert st["general_model"] == "gemini-3.5-flash"
        assert st["code_model"] == "qwen3-coder-next"
        assert st["features"]["agent"] is False and st["features"]["eval"] is False
        assert st["features"]["qa"] is True and st["features"]["skill_build"] is True
    finally:
        _demo_off()


def test_rate_limit_429():
    c, app = _client(DEMO_RPM="3", DEMO_GLOBAL_RPM="100000")
    try:
        # Unique visitor IP so this test's window is isolated from other tests
        # that share the default test-client remote_addr.
        h = {"X-Forwarded-For": "203.0.113.77"}
        codes = [c.get("/api/kg/stats", headers=h).status_code for _ in range(5)]
        assert codes[0] == 200 and codes[-1] == 429, codes
    finally:
        _demo_off()


def test_isolation_guard_refuses_cloud():
    # Importing app under DEMO_MODE + supabase backend must raise SystemExit (the
    # one import-time check). This is the only test that reloads app, so it heals
    # the module afterward and wipes the env LAST (the healing reload re-runs
    # config.load_env, which would otherwise leak real keys to later tests).
    import sys
    snap = dict(os.environ)
    os.environ["DEMO_MODE"] = "1"
    os.environ["CACHED_STORE_BACKEND"] = "supabase"
    os.environ["SUPABASE_DB_URL"] = "postgresql://x"
    importlib.reload(demo_budget)
    raised = False
    try:
        importlib.reload(sys.modules["app"]) if "app" in sys.modules else __import__("app")
    except SystemExit:
        raised = True
    finally:
        # Heal app under a benign (non-demo) env so the module is usable again...
        os.environ.pop("DEMO_MODE", None)
        os.environ["CACHED_STORE_BACKEND"] = "local"
        os.environ.pop("SUPABASE_DB_URL", None)
        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])
        # ...then restore the exact pre-test env LAST, wiping load_env's additions.
        os.environ.clear()
        os.environ.update(snap)
        importlib.reload(demo_budget)
    assert raised, "DEMO_MODE must refuse to start against cloud storage"


def test_off_mode_is_noop():
    _demo_off()
    import providers
    importlib.reload(providers)
    # With demo off, resolver honors the caller (no pin).
    assert providers.resolve_provider_model("anthropic", "claude-opus-4-8") == ("anthropic", "claude-opus-4-8")
    assert demo_budget.demo_enabled() is False
    demo_budget.check_budget()  # no-op, never raises
    demo_budget.record_usage("claude-opus-4-8", 10**9, 10**9)  # no-op


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} demo-mode tests passed.")


if __name__ == "__main__":
    _run_all()
