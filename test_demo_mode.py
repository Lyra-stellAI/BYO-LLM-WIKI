"""Tests for public DEMO_MODE: model pin, budget metering on every spend path,
feature gates, rate limiting, and cloud-storage isolation.

Fully offline — no network, no API keys needed. DEMO_MODE is toggled per test;
the conftest already forces local backends + a temp data dir, so importing app
here is safe. Run directly or under pytest.
"""

import importlib
import os

import demo_budget
import demo_store

# A full env snapshot is restored on teardown. This matters because route tests
# reload app.py, which runs config.load_env() and would otherwise pull the real
# OPENAI_API_KEY (and other keys) from .env into the process — leaking into later
# test files (e.g. test_memory asserts embeddings are off). Snapshot → restore
# makes every demo test hermetic regardless of run order.
_ENV_SNAPSHOT: dict | None = None


# Credential pairs demo_store looks for. Blanked on every _demo_on so a developer
# whose shell (or .env) points at a real Upstash database can't have the suite
# write to it — and so every test runs on the deterministic memory backend unless
# it opts in.
_REST_VARS = ("DEMO_REDIS_REST_URL", "DEMO_REDIS_REST_TOKEN",
              "KV_REST_API_URL", "KV_REST_API_TOKEN",
              "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN")


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
    for v in _REST_VARS:
        os.environ[v] = ""
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    for k, v in env.items():
        os.environ[k] = str(v)
    # The ledger and the rate-limit counters live in demo_store now, so reloading
    # demo_budget no longer clears them — clear the store explicitly instead.
    demo_store.reset()
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
    demo_store.reset()
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
        demo_budget.record_response("openai-chat", "gemini-2.5-flash",
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
        demo_budget.record_response("openai-chat", "gemini-2.5-flash", _R())
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
        mc.chat.completions.create(model="gemini-2.5-flash", messages=[])
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
        assert providers.resolve_provider_model("anthropic", "claude-opus-4-8") == ("gemini", "gemini-2.5-flash")
        assert providers.skill_generator("anthropic", "claude-opus-4-8") == ("qwen", "qwen3-coder-next")
        assert providers.judge_panel("gemini") == []
        jp, jm, cross = providers.resolve_judge("gemini", "gemini-2.5-flash", "openai", "gpt-4o")
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
        assert st["general_model"] == "gemini-2.5-flash"
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


# --- adversarial-review regressions: confirmed bypasses must stay closed ------
def test_summarize_pins_model_in_demo():
    # /api/summarize must ignore a visitor-supplied expensive model and run the
    # pinned cheap one (was a model-pin escape).
    c, app = _client(DEMO_RPM="1000")
    try:
        captured = {}
        orig = app.generate_ai_summary
        app.generate_ai_summary = lambda provider, model, *a, **k: (
            captured.update(provider=provider, model=model) or "summary text")
        try:
            r = c.post("/api/summarize", json={"input": "x" * 500,
                       "provider": "anthropic", "model": "claude-opus-4-8"})
        finally:
            app.generate_ai_summary = orig
        assert r.status_code == 200, r.get_json()
        assert captured.get("model") == "gemini-2.5-flash", captured
        assert captured.get("provider") == "gemini", captured
    finally:
        _demo_off()


def test_skill_build_forces_pipeline_backend_in_demo():
    # claude_code backend (opus CLI subprocess, unmetered) must be overridden to
    # the pinned/metered pipeline backend.
    c, app = _client(DEMO_RPM="1000")
    try:
        import skill_agent
        captured = {}
        orig = skill_agent.build_skill
        skill_agent.build_skill = lambda **kw: (captured.update(kw) or
            {"skill": {"id": "s"}, "status": "ok"})
        try:
            c.post("/api/skill/build", json={"text": "make a thing",
                   "backend": "claude_code", "model": "claude-opus-4-8"})
        finally:
            skill_agent.build_skill = orig
        assert captured.get("backend") == "pipeline", captured
    finally:
        _demo_off()


def test_claude_code_agent_refuses_in_demo():
    # Defense in depth: the subprocess generator itself refuses under DEMO_MODE.
    _demo_on()
    try:
        import skill_claude_agent
        raised = False
        try:
            skill_claude_agent.generate({}, model="claude-opus-4-8")
        except skill_claude_agent.ClaudeAgentError as e:
            raised = "demo" in str(e).lower()
        assert raised, "claude_code generate must refuse in demo"
    finally:
        _demo_off()


def test_bulk_ingestion_routes_blocked_in_demo():
    c, app = _client(DEMO_RPM="1000")
    try:
        for path in ("/api/rag/ingest", "/api/kg/ingest/urls", "/api/kg/ingest/files"):
            assert c.post(path, json={}).status_code == 403, path
    finally:
        _demo_off()


def test_xff_spoofing_does_not_mint_fresh_budget():
    # Without DEMO_TRUST_PROXY, the visitor key must be the peer address, NOT a
    # spoofable X-Forwarded-For — so rotating XFF can't reset the per-visitor key.
    c, app = _client(DEMO_RPM="1000")
    try:
        with app.app.test_request_context("/api/x", headers={"X-Forwarded-For": "1.2.3.4"},
                                          environ_base={"REMOTE_ADDR": "10.0.0.9"}):
            assert app._demo_key() == "10.0.0.9"
        # With trust enabled + a valid IP, XFF is honored.
        os.environ["DEMO_TRUST_PROXY"] = "1"
        with app.app.test_request_context("/api/x", headers={"X-Forwarded-For": "1.2.3.4"},
                                          environ_base={"REMOTE_ADDR": "10.0.0.9"}):
            assert app._demo_key() == "1.2.3.4"
        # ...but a garbage XFF falls back to the peer.
        with app.app.test_request_context("/api/x", headers={"X-Forwarded-For": "not-an-ip"},
                                          environ_base={"REMOTE_ADDR": "10.0.0.9"}):
            assert app._demo_key() == "10.0.0.9"
    finally:
        _demo_off()


def test_extract_batch_capped_in_demo():
    c, app = _client(DEMO_RPM="1000", DEMO_EXTRACT_MAX="5")
    try:
        many = "\n".join(f"https://example.com/{i}" for i in range(20))
        r = c.post("/api/cache/extract", json={"urls": many})
        assert r.status_code == 400 and "at most 5" in r.get_json().get("error", "")
    finally:
        _demo_off()


def test_ssrf_guard_blocks_internal_hosts():
    _demo_on()
    try:
        import app
        # metadata IP + loopback + private must be rejected when block_internal
        assert app.is_valid_url("http://169.254.169.254/latest/meta-data/", block_internal=True) is False
        assert app.is_valid_url("http://127.0.0.1:5000/", block_internal=True) is False
        assert app.is_valid_url("http://localhost/", block_internal=True) is False
        # a normal public URL still validates
        assert app.is_valid_url("https://example.com/page", block_internal=True) is True
    finally:
        _demo_off()


def test_inflight_cap_429():
    # The per-visitor in-flight counter rejects excess concurrent POSTs (bounds
    # the budget TOCTOU window). Simulate by leaving the counter elevated.
    c, app = _client(DEMO_RPM="1000", DEMO_INFLIGHT="2")
    try:
        slot = app._inflight_key("7.7.7.7")
        app.demo_store.bump([(slot, 300), (slot, 300)])
        r = c.post("/api/rag/ask", json={"question": "hi"},
                   headers={}, environ_base={"REMOTE_ADDR": "7.7.7.7"})
        assert r.status_code == 429 and "concurrent" in r.get_json().get("error", "")
    finally:
        _demo_off()


def test_inflight_slot_released_after_request():
    # A rejected claim must give its slot back, and a served request must release
    # its own — otherwise a visitor would lock themselves out after DEMO_INFLIGHT
    # requests without a single concurrent one.
    c, app = _client(DEMO_RPM="1000", DEMO_INFLIGHT="1")
    try:
        slot = app._inflight_key("8.8.8.8")
        for _ in range(3):
            c.post("/api/rag/ask", json={"question": "hi"},
                   environ_base={"REMOTE_ADDR": "8.8.8.8"})
        assert demo_store.get_float(slot) == 0, "in-flight slots must be released"
    finally:
        _demo_off()


def test_preflight_flags_empty_reply():
    # Preflight must catch a model that returns empty (the gemini-3.5-flash class
    # of silent failure) and pass a healthy one. Stub build_chat_model + embeddings.
    _demo_on()
    try:
        import providers, embeddings
        class _Msg:
            def __init__(self, c): self.content = c
        class _Chat:
            def __init__(self, c): self._c = c
            def invoke(self, _): return _Msg(self._c)
        # general → empty, code → "OK"
        orig_bcm, orig_avail, orig_eq = (providers.build_chat_model,
            embeddings.embeddings_available, embeddings.embed_query)
        providers.build_chat_model = lambda p, m, **k: _Chat("" if "gemini" in m else "OK")
        embeddings.embeddings_available = lambda: True
        import numpy as np
        embeddings.embed_query = lambda *a, **k: np.zeros(4, dtype="float32")
        try:
            res = demo_budget.preflight(verbose=False)
        finally:
            providers.build_chat_model, embeddings.embeddings_available, embeddings.embed_query = (
                orig_bcm, orig_avail, orig_eq)
        by_role = {r["role"]: r for r in res}
        assert by_role["general"]["ok"] is False, "empty reply must be flagged"
        assert by_role["code"]["ok"] is True
        assert by_role["embeddings"]["ok"] is True
    finally:
        _demo_off()


# --- shared ledger backend (demo_store) --------------------------------------
class _FakeRedis:
    """Minimal Upstash /pipeline stand-in over a dict.

    Implements only the commands demo_store issues. ``calls`` records every
    pipeline body so tests can assert on the wire format, and ``fail`` makes the
    endpoint unreachable to exercise the fallback path."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.calls: list = []
        self.fail = False

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(json)
        if self.fail:
            raise OSError("upstash unreachable")
        out = []
        for cmd in json:
            name = str(cmd[0]).upper()
            key = cmd[1] if len(cmd) > 1 else None
            if name == "INCRBYFLOAT":
                new = float(self.data.get(key, "0")) + float(cmd[2])
                self.data[key] = f"{new:.12f}"
                out.append({"result": self.data[key]})
            elif name == "INCR":
                new = int(float(self.data.get(key, "0"))) + 1
                self.data[key] = str(new)
                out.append({"result": new})
            elif name == "DECR":
                new = int(float(self.data.get(key, "0"))) - 1
                self.data[key] = str(new)
                out.append({"result": new})
            elif name == "GET":
                out.append({"result": self.data.get(key)})
            elif name == "SET":
                self.data[key] = str(cmd[2])
                out.append({"result": "OK"})
            elif name == "EXPIRE":
                out.append({"result": 1})
            else:
                raise AssertionError(f"unexpected command {name}")

        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return out

        return _Resp()


def _with_fake_redis(**env):
    """Turn the demo on with a fake Upstash wired in. Returns (fake, restore)."""
    import requests
    _demo_on(**env)
    os.environ["DEMO_REDIS_REST_URL"] = "https://fake.upstash.io"
    os.environ["DEMO_REDIS_REST_TOKEN"] = "token"
    fake = _FakeRedis()
    original = requests.post
    requests.post = fake.post

    def restore():
        requests.post = original
        _demo_off()

    return fake, restore


def test_store_backend_selection():
    _demo_on()
    try:
        assert demo_store.backend() == "memory", "no credentials -> process-local"
        os.environ["DEMO_REDIS_REST_URL"] = "https://fake.upstash.io"
        os.environ["DEMO_REDIS_REST_TOKEN"] = "token"
        assert demo_store.backend() == "redis"
    finally:
        _demo_off()


def test_store_never_serializes_floats_in_exponential_notation():
    # Redis rejects "3e-05" with "value is not a valid float", and per-call demo
    # costs are routinely that small — so this is the difference between a working
    # ledger and one that silently never records anything.
    fake, restore = _with_fake_redis(DEMO_VISITOR_USD="100", DEMO_GLOBAL_USD="100")
    try:
        demo_budget.set_key("sci")
        demo_budget.record_usage("text-embedding-3-small", 10, 0)  # ~5e-7 USD
        args = [str(c[2]) for body in fake.calls for c in body if c[0] == "INCRBYFLOAT"]
        assert args, "a cost must have been sent"
        assert not any("e" in a.lower() for a in args), args
    finally:
        restore()


def test_store_ledger_is_shared_across_instances():
    # The whole point of the Redis backend: instance B must see instance A's
    # spend. Clearing the process-local mirror simulates a fresh instance.
    fake, restore = _with_fake_redis(DEMO_VISITOR_USD="100", DEMO_GLOBAL_USD="0.01")
    try:
        demo_budget.set_key("shared")
        demo_budget.record_usage("qwen3-coder-next", 10_000, 10_000)  # > $0.01
        demo_store.reset()  # <- "instance B" boots with an empty mirror
        raised = False
        try:
            demo_budget.check_budget()
        except demo_budget.BudgetExceededError:
            raised = True
        assert raised, "a second instance must inherit the global spend"
    finally:
        restore()


def test_store_redis_outage_falls_back_to_local_cap():
    # A ledger outage must degrade to the per-instance cap (what a single-process
    # host always had), never to an uncapped demo.
    fake, restore = _with_fake_redis(DEMO_VISITOR_USD="0.001", DEMO_GLOBAL_USD="100")
    try:
        fake.fail = True
        demo_budget.set_key("outage")
        demo_budget.check_budget()  # fresh visitor still allowed
        demo_budget.record_usage("qwen3-coder-next", 500, 500)  # > $0.001
        raised = False
        try:
            demo_budget.check_budget()
        except demo_budget.BudgetExceededError:
            raised = True
        assert raised, "local mirror must still enforce the cap when Redis is down"
    finally:
        restore()


def test_store_release_clamps_at_zero():
    # An instance that dies mid-request never releases its slot; a decrement that
    # is allowed to go negative would quietly disable the in-flight cap.
    fake, restore = _with_fake_redis()
    try:
        key = "byowiki:demo:test:inflight"
        demo_store.release(key)
        demo_store.release(key)
        assert float(fake.data.get(key, 0)) >= 0, fake.data
        assert demo_store.get_float(key) == 0
    finally:
        restore()


def test_ledger_window_rolls_over():
    # The window lives in the key, so crossing a boundary frees the budget with no
    # coordination between instances.
    import time as _t
    _demo_on(DEMO_VISITOR_USD="0.001", DEMO_GLOBAL_USD="100", DEMO_WINDOW_SEC="1")
    try:
        demo_budget.set_key("roller")
        demo_budget.record_usage("qwen3-coder-next", 500, 500)
        raised = False
        try:
            demo_budget.check_budget()
        except demo_budget.BudgetExceededError:
            raised = True
        assert raised, "visitor should be capped inside the window"
        _t.sleep(1.05)
        demo_budget.check_budget()  # next window: clean slate
    finally:
        _demo_off()


def test_global_rate_limit_429():
    c, app = _client(DEMO_RPM="100000", DEMO_GLOBAL_RPM="3")
    try:
        codes = [c.get("/api/kg/stats").status_code for _ in range(5)]
        assert codes[0] == 200 and codes[-1] == 429, codes
    finally:
        _demo_off()


# --- Vercel bootstrap --------------------------------------------------------
def test_bootstrap_is_inert_without_vercel_env():
    import vercel_bootstrap
    snap = dict(os.environ)
    try:
        os.environ.pop("VERCEL", None)
        assert vercel_bootstrap.on_vercel() is False
    finally:
        os.environ.clear()
        os.environ.update(snap)


def test_bootstrap_seeds_writable_dir_and_repoints():
    import shutil
    import tempfile

    import vercel_bootstrap
    snap = dict(os.environ)
    seed = tempfile.mkdtemp(prefix="byowiki_seed_")
    holder = tempfile.mkdtemp(prefix="byowiki_tmp_")
    target = os.path.join(holder, "data")
    try:
        with open(os.path.join(seed, "overall.json"), "w", encoding="utf-8") as fh:
            fh.write('{"nodes": []}')
        os.environ["KG_DATA_DIR"] = seed
        os.environ["DEMO_RUNTIME_DATA_DIR"] = target
        vercel_bootstrap._done = False
        assert vercel_bootstrap.ensure_writable_data_dir() == target
        assert os.environ["KG_DATA_DIR"] == target, "modules must import against the copy"
        assert os.path.exists(os.path.join(target, "overall.json")), "seed must be copied"

        # Warm instance: the marker makes a re-run skip the copy, so visitor
        # writes already in /tmp are not silently reverted to the seed.
        os.remove(os.path.join(target, "overall.json"))
        os.environ["KG_DATA_DIR"] = seed
        vercel_bootstrap._done = False
        vercel_bootstrap.ensure_writable_data_dir()
        assert not os.path.exists(os.path.join(target, "overall.json")), \
            "re-seeding a warm instance would discard visitor state"
    finally:
        vercel_bootstrap._done = False
        os.environ.clear()
        os.environ.update(snap)
        shutil.rmtree(seed, ignore_errors=True)
        shutil.rmtree(holder, ignore_errors=True)


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
