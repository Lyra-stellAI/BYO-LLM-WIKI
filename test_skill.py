"""Smoke tests for the agent-skill layer (skill_library / skill_eval / skill_agent).

Runs fully offline: no API keys, no network. We exercise the deterministic
checks, the accept/reject gate logic, the store lifecycle (draft → eval → human
review), context gathering from the knowledge graph, and SKILL.md export — every
part of the build loop that does not require an LLM. Run directly
(``python test_skill.py``) or under pytest.
"""

import os
import tempfile

# Isolate state BEFORE importing the layer: a throwaway data dir + no embeddings.
os.environ["KG_DATA_DIR"] = tempfile.mkdtemp(prefix="skill_test_")
os.environ.pop("OPENAI_API_KEY", None)

import json  # noqa: E402

import knowledge_graph as kg  # noqa: E402
import skill_library as sk  # noqa: E402
import skill_eval  # noqa: E402
import skill_agent  # noqa: E402
import skill_runs  # noqa: E402
import skill_runtime  # noqa: E402
import providers  # noqa: E402


def _good_skill_spec():
    return {
        "name": "Draft Release Notes",
        "description": "Draft concise release notes from a project changelog grounded in the context.",
        "instructions": ("Turn a raw changelog into reader-friendly release notes.\n"
                         "1. Read the changelog entries provided.\n"
                         "2. Group them into Features, Fixes, and Breaking changes.\n"
                         "3. Write a one-line summary per group with the most important item first."),
        "steps": ["Read the changelog", "Group entries by type", "Write grouped summaries"],
        "triggers": ["When asked to summarize a changelog into release notes"],
        "anti_triggers": ["When there is no changelog or version history to summarize"],
        "tools": ["text editor"],
        "success_criteria": {"outcome": "accurate notes", "process": "grouped",
                             "style": "concise", "efficiency": "one pass"},
        "tests": [
            {"prompt": "Write release notes for v2.0 from this changelog", "should_trigger": True,
             "expect": "grouped notes"},
            {"prompt": "What is the capital of France?", "should_trigger": False,
             "expect": "defer — unrelated"},
        ],
        "provenance": {"chunk_ids": ["chunk_demo"], "source_titles": ["Changelog"],
                       "context_chars": 120, "context_digest": "v2.0 added X, fixed Y."},
    }


def setup_function(_=None):
    sk.clear()
    skill_runs.clear()
    kg.clear("current")
    kg.clear("overall")


# --- deterministic checks ----------------------------------------------------
def test_deterministic_passes_good_skill():
    sk.clear()
    det = skill_eval.run_deterministic(_good_skill_spec())
    assert det["ratio"] == 1.0, det["failures"]
    assert not det["critical_failed"]


def test_deterministic_flags_bad_skill():
    bad = {"name": "X", "description": "too short", "instructions": "hi", "triggers": [],
           "anti_triggers": [], "tools": [], "tests": [], "provenance": {}}
    det = skill_eval.run_deterministic(bad)
    assert det["ratio"] < 1.0
    # has_instructions and has_steps are critical and must fail here.
    assert "has_instructions" in det["critical_failed"]
    assert "has_steps" in det["critical_failed"]


def test_deterministic_catches_placeholders():
    spec = _good_skill_spec()
    spec["instructions"] += "\n4. TODO: finish this step."
    det = skill_eval.run_deterministic(spec)
    assert any(c["key"] == "no_placeholders" and not c["passed"] for c in det["checks"])


# --- gate logic --------------------------------------------------------------
def test_gate_rejects_on_critical_failure():
    bad_det = {"ratio": 0.5, "critical_failed": ["has_instructions"], "failures": ["has_instructions"]}
    out = skill_eval.decide_gate(bad_det, {"mean": 0.95})
    assert out["gate"] == "reject"


def test_gate_review_without_rubric():
    det = skill_eval.run_deterministic(_good_skill_spec())
    assert skill_eval.decide_gate(det, None)["gate"] == "review"


def test_gate_accept_and_reject_with_rubric():
    det = skill_eval.run_deterministic(_good_skill_spec())
    assert skill_eval.decide_gate(det, {"mean": 0.85})["gate"] == "accept"
    assert skill_eval.decide_gate(det, {"mean": 0.30})["gate"] == "reject"
    assert skill_eval.decide_gate(det, {"mean": 0.55})["gate"] == "review"


# --- store lifecycle ---------------------------------------------------------
def test_create_and_get():
    sk.clear()
    s = sk.upsert_skill(_good_skill_spec())
    assert s and s["id"].startswith("skill_")
    assert s["status"] == sk.DRAFT and s["version"] == 1
    got = sk.get_skill(s["id"])
    assert got["name"] == "Draft Release Notes"
    assert sk.get_skill("draft-release-notes")["id"] == s["id"]  # resolve by slug


def test_record_eval_drives_status():
    sk.clear()
    s = sk.upsert_skill(_good_skill_spec())
    sk.record_eval(s["id"], {"gate": "accept", "deterministic": {"passed": 11, "total": 11},
                             "rubric": {"mean": 0.8}})
    assert sk.get_skill(s["id"])["status"] == sk.PENDING_REVIEW
    s2 = sk.upsert_skill({**_good_skill_spec(), "name": "Another Skill"})
    sk.record_eval(s2["id"], {"gate": "reject", "deterministic": {"passed": 3, "total": 11},
                              "rubric": {"mean": 0.2}})
    assert sk.get_skill(s2["id"])["status"] == sk.REJECTED


def test_human_review_accept_makes_active():
    sk.clear()
    s = sk.upsert_skill(_good_skill_spec())
    sk.record_eval(s["id"], {"gate": "accept", "deterministic": {}, "rubric": {"mean": 0.8}})
    reviewed = sk.record_human_review(s["id"], decision="accept", score=0.9, notes="lgtm")
    assert reviewed["status"] == sk.ACCEPTED
    assert reviewed["accepted_at"]
    assert reviewed["human"]["aligned_with_gate"] is True  # gate said accept, human agreed
    assert sk.stats()["accepted"] == 1


def test_human_review_revise_and_reject():
    sk.clear()
    s = sk.upsert_skill(_good_skill_spec())
    sk.record_eval(s["id"], {"gate": "accept", "deterministic": {}, "rubric": {"mean": 0.8}})
    revised = sk.record_human_review(s["id"], decision="revise", notes="add an example")
    assert revised["status"] == sk.NEEDS_REVISION
    assert revised["human"]["aligned_with_gate"] is False  # gate said accept, human did not
    rejected = sk.record_human_review(s["id"], decision="reject")
    assert rejected["status"] == sk.REJECTED


def test_revise_bumps_version_and_resets():
    sk.clear()
    s = sk.upsert_skill(_good_skill_spec())
    sk.record_eval(s["id"], {"gate": "accept", "deterministic": {}, "rubric": {"mean": 0.8}})
    sk.record_human_review(s["id"], decision="accept")
    again = sk.upsert_skill({**_good_skill_spec(), "revision_note": "tighter steps"})
    assert again["id"] == s["id"]  # same slug -> revised in place
    assert again["version"] == 2 and again["revisions"] == 1
    assert again["status"] == sk.DRAFT and again["eval"] is None and again["human"] is None


# --- recall (only accepted skills are live) ----------------------------------
def test_recall_returns_only_accepted():
    sk.clear()
    s = sk.upsert_skill(_good_skill_spec())
    # not accepted yet -> not recallable
    assert sk.recall("summarize a changelog into release notes") == []
    sk.record_eval(s["id"], {"gate": "accept", "deterministic": {}, "rubric": {"mean": 0.8}})
    sk.record_human_review(s["id"], decision="accept")
    hits = sk.recall("turn a changelog into release notes")
    assert hits and hits[0]["name"] == "Draft Release Notes"
    assert sk.find_skill("release notes")["id"] == s["id"]


# --- SKILL.md render + export ------------------------------------------------
def test_skill_md_and_export():
    sk.clear()
    s = sk.upsert_skill(_good_skill_spec())
    md = sk.to_skill_md(sk.get_skill(s["id"]))
    assert "name: Draft Release Notes" in md and "## Instructions" in md
    res = sk.export_skill(s["id"])
    assert res["ok"] and os.path.exists(res["path"])


def test_stats_counts():
    sk.clear()
    a = sk.upsert_skill(_good_skill_spec())
    sk.record_eval(a["id"], {"gate": "accept", "deterministic": {}, "rubric": {"mean": 0.8}})
    sk.record_human_review(a["id"], decision="accept")
    b = sk.upsert_skill({**_good_skill_spec(), "name": "Pending Skill"})
    sk.record_eval(b["id"], {"gate": "accept", "deterministic": {}, "rubric": {"mean": 0.8}})
    s = sk.stats()
    assert s["total"] == 2 and s["accepted"] == 1 and s["pending_review"] == 1


# --- context gathering (no LLM) ----------------------------------------------
def test_gather_context_from_chunk():
    sk.clear()
    kg.clear("current")
    chunk = kg.add_chunk("Deep agents use a planner, tools, and a filesystem workspace.",
                         source_title="Deep Agents", source_url="https://example.com/da")
    bundle = skill_agent.gather_context(chunk_ids=[chunk["id"]])
    assert "Deep agents use a planner" in bundle["context"]
    assert chunk["id"] in bundle["provenance"]["chunk_ids"]
    assert bundle["provenance"]["context_chars"] > 0


def test_gather_context_requires_something():
    sk.clear()
    try:
        skill_agent.gather_context()
    except skill_agent.SkillError:
        return
    raise AssertionError("expected SkillError for empty context")


# --- observability store -----------------------------------------------------
def test_runs_record_and_benchmark():
    skill_runs.clear()
    skill_runs.record(kind="build", skill_id="skill_a", skill_name="A", provider="anthropic",
                      model="claude-opus-4-8", gate="accept", status="pending_review",
                      duration_ms=1200, tokens=900,
                      metrics={"deterministic_ratio": 1.0, "rubric_mean": 0.8, "trigger_f1": 1.0})
    skill_runs.record(kind="build", skill_id="skill_b", skill_name="B", provider="openai",
                      model="gpt", gate="reject", status="rejected", duration_ms=800, tokens=500,
                      metrics={"deterministic_ratio": 0.6, "rubric_mean": 0.3})
    bench = skill_runs.benchmark()
    assert bench["total_runs"] == 2 and bench["builds"] == 2
    assert bench["gate_pass_rate"] == 0.5  # 1 of 2 accepted
    assert bench["total_tokens"] == 1400
    assert bench["gate_distribution"] == {"accept": 1, "reject": 1}
    rows = skill_runs.list_runs(skill_id="skill_a")
    assert len(rows) == 1 and rows[0]["model"] == "claude-opus-4-8"


# --- latest-Claude generator preference --------------------------------------
def test_skill_generator_prefers_claude():
    saved = {k: os.environ.get(k) for k in
             ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    try:
        os.environ["ANTHROPIC_API_KEY"] = "x"
        os.environ.pop("OPENAI_API_KEY", None)
        assert providers.skill_generator("auto", None) == ("anthropic", providers.SKILL_GENERATOR_MODEL)
        # explicit provider is honored as-is
        assert providers.skill_generator("openai", "gpt-4o")[0] == "openai"
        # no anthropic key -> falls back to whatever is configured
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["OPENAI_API_KEY"] = "y"
        assert providers.skill_generator("auto", None)[0] == "openai"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- triggering eval (false positive / negative) -----------------------------
class _FakeTriggerChat:
    """Returns trigger=true only when the USER REQUEST mentions a changelog."""
    def invoke(self, prompt):
        import re as _re
        m = _re.search(r'User request: "(.*?)"', prompt, _re.DOTALL)
        request = (m.group(1) if m else prompt).lower()
        return json.dumps({"trigger": "changelog" in request})


def test_trigger_eval_precision_recall():
    saved = (skill_eval.build_chat_model, skill_eval.resolve_provider_model)
    skill_eval.build_chat_model = lambda *a, **k: _FakeTriggerChat()
    skill_eval.resolve_provider_model = lambda p, m: ("openai", "gpt-test")
    try:
        out = skill_eval.run_trigger_eval(_good_skill_spec())
        assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0
        assert out["tp"] == 1 and out["tn"] == 1 and out["fp"] == 0 and out["fn"] == 0
    finally:
        skill_eval.build_chat_model, skill_eval.resolve_provider_model = saved


# --- tool-use author loop ----------------------------------------------------
class _FakeToolChat:
    """Calls check_draft once, then returns the final skill JSON."""
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self.calls += 1
        if self.calls == 1:
            draft = {k: _good_skill_spec()[k] for k in
                     ("name", "description", "instructions", "steps", "triggers",
                      "anti_triggers", "tools", "tests")}
            return AIMessage(content="", tool_calls=[
                {"name": "check_draft", "args": {"draft_json": json.dumps(draft)}, "id": "c1"}])
        return AIMessage(content=json.dumps(_good_skill_spec()))


def test_tool_loop_uses_tools_and_returns_skill():
    bundle = {"context": "v2.0 added X, fixed Y. The changelog lists features.",
              "provenance": {"chunk_ids": ["c1"], "source_titles": ["Changelog"]}}
    tools = skill_runtime.builder_tools(bundle)
    assert {t.name for t in tools} >= {"read_context", "check_draft", "list_existing_skills", "recall_memory"}
    out = skill_runtime.run_tool_loop(_FakeToolChat(), tools, "system", "user", max_iters=4)
    assert out["tool_calls"] == 1
    parsed = json.loads(out["text"])
    assert parsed["name"] == "Draft Release Notes"


def test_check_draft_tool_reports_failures():
    bundle = {"context": "ctx", "provenance": {"chunk_ids": ["c1"]}}
    tools = {t.name: t for t in skill_runtime.builder_tools(bundle)}
    bad = json.dumps({"name": "X", "description": "too short", "instructions": "hi"})
    res = json.loads(tools["check_draft"].invoke({"draft_json": bad}))
    assert "has_steps" in res["failures"] and res["fix"]


# --- refine guidance ---------------------------------------------------------
def test_refine_guidance_from_eval():
    skill = _good_skill_spec()
    skill["eval"] = {"deterministic": {"failures": ["has_anti_triggers"]},
                     "triggering": {"precision": 0.5, "recall": 1.0},
                     "rubric": {"per_dimension": {"style": 0.4, "outcome": 0.9}}}
    g = skill_agent._refine_guidance(skill)
    assert "has_anti_triggers" in g and "false positives" in g and "style" in g


# --- Claude Code subprocess backend ------------------------------------------
def test_claude_code_command_and_parsing():
    import skill_claude_agent as cc
    cmd = cc.build_command("claude-opus-4-8", allow_tools=True)
    assert cmd[0] == cc.CLAUDE_BIN and "-p" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--model" in cmd and "claude-opus-4-8" in cmd
    assert cc.ALLOWED_TOOLS_FLAG in cmd and "--permission-mode" in cmd
    # allow_tools=False drops the tool/permission flags
    assert cc.ALLOWED_TOOLS_FLAG not in cc.build_command("m", allow_tools=False)
    env = cc.parse_envelope(json.dumps({"type": "result", "result": "hi",
        "usage": {"input_tokens": 10, "output_tokens": 5}, "total_cost_usd": 0.01, "num_turns": 2}))
    assert env["result"] == "hi" and env["tokens"] == 15 and env["num_turns"] == 2
    assert cc.extract_skill("```json\n{\"name\": \"X\"}\n```") == {"name": "X"}


def test_claude_code_generate_with_injected_runner():
    import skill_claude_agent as cc
    from pathlib import Path

    def fake_runner(cmd, prompt, workspace, timeout):
        Path(workspace, "skill.json").write_text(json.dumps(_good_skill_spec()), encoding="utf-8")
        Path(workspace, "SKILL.md").write_text("# Draft Release Notes\n", encoding="utf-8")
        env = {"type": "result", "is_error": False, "result": "done",
               "usage": {"input_tokens": 100, "output_tokens": 200},
               "total_cost_usd": 0.02, "num_turns": 4}
        return {"returncode": 0, "stdout": json.dumps(env), "stderr": ""}

    bundle = {"context": "changelog v2.0 added X, fixed Y",
              "provenance": {"chunk_ids": ["c1"], "source_titles": ["Changelog"]}}
    out = cc.generate(bundle, _runner=fake_runner)
    assert out["artifact"]["name"] == "Draft Release Notes"     # recovered from skill.json
    assert out["meta"]["num_turns"] == 4 and out["meta"]["tokens"] == 300
    assert out["meta"]["skill_md"].startswith("# Draft Release Notes")


def test_claude_code_backend_build_runs_eval():
    import skill_claude_agent as cc
    saved = cc.generate
    cc.generate = lambda bundle, **kw: {
        "artifact": _good_skill_spec(),
        "meta": {"backend": "claude_code", "model": "claude-opus-4-8", "tokens": 1234,
                 "num_turns": 3, "cost_usd": 0.01, "understanding": "u", "analysis": "a",
                 "skill_md": "# md"},
        "workspace": "/tmp/x"}
    try:
        # run_rubric off so no LLM judge is needed -> deterministic-only gate = review
        res = skill_agent.build_skill(text="context about changelogs and releases",
                                      backend="claude_code", run_rubric=False, run_triggering=False)
    finally:
        cc.generate = saved
    assert res["backend"] == "claude_code"
    assert res["skill"]["status"] == sk.PENDING_REVIEW   # eval ran; deterministic clean -> review
    assert res["observability"]["tokens"] == 1234 and res["observability"]["tools_used"] == 3
    runs = skill_runs.list_runs(kind="build")
    assert runs and runs[0]["provider"] == "claude_code"


def test_claude_code_unavailable_raises_clean_error():
    import skill_claude_agent as cc
    saved = cc.cli_available
    cc.cli_available = lambda: False
    try:
        try:
            skill_agent.build_skill(text="x" * 80, backend="claude_code")
        except skill_agent.SkillError as e:
            assert "Claude Code CLI" in str(e)
            return
        raise AssertionError("expected SkillError when the CLI is unavailable")
    finally:
        cc.cli_available = saved


# --- LangSmith / OTel tracing export -----------------------------------------
def test_tracing_payload_shape():
    import skill_tracing as st
    run = {"id": "run_1", "kind": "build", "skill_name": "S", "provider": "claude_code",
           "model": "claude-opus-4-8", "gate": "accept", "status": "pending_review",
           "duration_ms": 1000, "tokens": 500, "tools_used": 3,
           "metrics": {"deterministic_ratio": 1.0, "rubric_mean": 0.8, "trigger_f1": 1.0}, "ok": True}
    rec = st.build_run_record(run)
    assert rec["name"] == "skill.build" and rec["run_type"] == "chain"
    assert rec["outputs"]["gate"] == "accept" and rec["outputs"]["rubric_mean"] == 0.8
    assert rec["feedback"]["gate_pass"] == 1.0 and rec["feedback"]["rubric_mean"] == 0.8
    assert "agent-skill" in rec["tags"] and "build" in rec["tags"]
    assert rec["metadata"]["skill_id"] is None and rec["metadata"]["tokens"] == 500


def test_tracing_disabled_is_noop_and_fake_export():
    import skill_tracing as st
    run = {"id": "r", "kind": "eval", "gate": "reject", "status": "rejected",
           "duration_ms": 50, "tokens": 10, "metrics": {"deterministic_ratio": 0.5}}
    keys = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "SKILL_TRACING",
            "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    try:
        assert st.langsmith_enabled() is False
        assert st.export(run) == {"langsmith": False, "otel": False}

        os.environ["LANGSMITH_API_KEY"] = "test-key"
        os.environ["SKILL_TRACING"] = "true"
        calls = {"create": 0, "update": 0, "fb": 0}

        class FakeClient:
            def create_run(self, **kw):
                calls["create"] += 1
                assert kw["project_name"] == st.SKILL_PROJECT
                assert kw["name"] == "skill.eval"

            def update_run(self, rid, **kw):
                calls["update"] += 1

            def create_feedback(self, rid, key=None, score=None):
                calls["fb"] += 1

        st._client = FakeClient()
        st._project_ready = True
        out = st.export(run)
        assert out["langsmith"] is True
        assert calls["create"] == 1 and calls["update"] == 1 and calls["fb"] >= 1
    finally:
        st._client = None
        st._project_ready = False
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        setup_function()
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} skill tests passed.")


if __name__ == "__main__":
    _run_all()
