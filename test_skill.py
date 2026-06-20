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

import knowledge_graph as kg  # noqa: E402
import skill_library as sk  # noqa: E402
import skill_eval  # noqa: E402
import skill_agent  # noqa: E402


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
