"""Agent-skill library — the library's layer of reusable, evaluated skills (layer 7).

The knowledge graph (sources → chunks → entities → topics → syntheses), the
memory layer (layer 6), and the vector index all capture *what the library
knows*. The skill library adds *what the library can do*: from selective context
the user feeds in, a sub-agent pipeline (``skill_agent.py``) drafts a reusable
**agent skill** — a named, described, step-by-step instruction set the library
(or any agent) can later invoke to perform a repeatable task.

A skill is not knowledge quoted from a source (a chunk) nor a learning recalled
across sessions (a memory): it is an *executable competence* — a name, a
description (the primary signal an agent uses to decide whether to invoke it),
step-by-step instructions, the tools it expects, explicit trigger / anti-trigger
cases, success criteria, and a small test set. Each skill carries the **eval
report** that graded it and the **human review** that signed off on it, because a
skill only enters the active library once it passes evaluation *and* a human
aligns on it (the accept/reject gate and human-in-the-loop step of the
build loop in ``skill_agent.py``).

Like the memory layer, this lives in its own plain-JSON store
(``data/skills.json``) to match the repo's local-first design; per-skill
embeddings are stored inline so skill *recall* (matching a task to a skill) needs
no extra index files. No new heavy dependencies — just the existing OpenAI
embeddings (optional) and numpy.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import numpy as np

DATA_DIR = Path(os.environ.get("KG_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SKILLS_PATH = DATA_DIR / "skills.json"
SKILL_MD_DIR = DATA_DIR / "skills"  # exported SKILL.md files live here

SCHEMA_VERSION = 1
LAYER = 7  # sits above memory (6) in the layered knowledge model

# --- Skill lifecycle (the build loop's states) -------------------------------
# draft → evaluated → (gate) → pending_review → accepted | rejected
#                                              ↘ needs_revision → (re-run) → …
DRAFT = "draft"                      # authored by codeact, not yet evaluated
EVALUATED = "evaluated"              # eval ran; awaiting the gate decision
PENDING_REVIEW = "pending_review"    # passed the automated gate; needs a human
ACCEPTED = "accepted"               # human-aligned and live in the library
REJECTED = "rejected"               # failed the gate or a human declined it
NEEDS_REVISION = "needs_revision"   # a human asked for changes; re-buildable

STATUSES = {DRAFT, EVALUATED, PENDING_REVIEW, ACCEPTED, REJECTED, NEEDS_REVISION}
# Statuses an agent may actually invoke / recall by default.
ACTIVE_STATUSES = {ACCEPTED}

DEDUP_THRESHOLD = float(os.environ.get("SKILL_DEDUP_THRESHOLD", "0.93"))
_SEMANTIC_FLOOR = float(os.environ.get("SKILL_RECALL_FLOOR", "0.20"))
_KEYWORD_FLOOR = float(os.environ.get("SKILL_RECALL_FLOOR_KEYWORD", "0.05"))

_lock = RLock()
# Small mtime-keyed cache so repeated reads don't re-parse the JSON file.
_cache: dict = {"mtime": None, "data": None}


# --- low-level store ---------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty() -> dict:
    return {"version": SCHEMA_VERSION, "created_at": _now(),
            "updated_at": _now(), "skills": []}


def _load() -> dict:
    if not SKILLS_PATH.exists():
        return _empty()
    try:
        mtime = SKILLS_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if _cache["data"] is not None and _cache["mtime"] == mtime:
        return _cache["data"]
    try:
        data = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  (corrupt file -> start clean rather than crash)
        return _empty()
    data.setdefault("version", SCHEMA_VERSION)
    data.setdefault("skills", [])
    _cache["data"], _cache["mtime"] = data, mtime
    return data


def _save(data: dict) -> None:
    data["updated_at"] = _now()
    SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SKILLS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SKILLS_PATH)
    try:
        _cache["data"], _cache["mtime"] = data, SKILLS_PATH.stat().st_mtime
    except OSError:
        _cache["data"], _cache["mtime"] = data, None


def _slugify(name: str) -> str:
    s = (name or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s.strip("-")[:60]


def _preview(text: str, n: int = 200) -> str:
    text = (text or "").strip()
    return (text[:n] + "…") if len(text) > n else text


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        return [v.strip() for v in value.splitlines() if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


# --- embeddings (optional, best-effort) --------------------------------------
def embeddings_on() -> bool:
    try:
        import embeddings as emb
        return emb.embeddings_available()
    except Exception:  # noqa: BLE001
        return False


def _embed(text: str) -> list[float] | None:
    """Embed a single string, or None when embeddings are unavailable.

    Best-effort: any failure (no key, network, quota) degrades to the keyword
    path rather than raising, so skill recall never blocks an answer."""
    if not text or not embeddings_on():
        return None
    try:
        import embeddings as emb
        return emb.embed_query(text).astype(np.float32).tolist()
    except Exception:  # noqa: BLE001
        return None


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb / (na * nb))


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _keyword_sim(query: str, text: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(text)) / len(q)


def _embed_text(skill: dict) -> str:
    """The text used to embed/recall a skill: its trigger surface."""
    return "\n".join([
        skill.get("name", ""),
        skill.get("description", ""),
        " ".join(skill.get("triggers", []) or []),
        _preview(skill.get("instructions", ""), 400),
    ]).strip()


# --- public shape ------------------------------------------------------------
_PUBLIC_FIELDS = (
    "id", "name", "slug", "description", "instructions", "steps", "triggers",
    "anti_triggers", "tools", "success_criteria", "tests", "provenance",
    "status", "version", "revisions", "eval", "human", "history",
    "created_at", "updated_at", "accepted_at",
)


def _public(skill: dict) -> dict:
    out = {f: skill.get(f) for f in _PUBLIC_FIELDS}
    out["preview"] = skill.get("preview") or _preview(skill.get("description", ""))
    out["layer"] = LAYER
    return out


def _by_id(data: dict, skill_id: str) -> dict | None:
    for s in data["skills"]:
        if s.get("id") == skill_id:
            return s
    return None


def _by_slug(data: dict, slug: str) -> dict | None:
    for s in data["skills"]:
        if s.get("slug") == slug:
            return s
    return None


def _resolve(data: dict, name_or_id: str) -> dict | None:
    if not name_or_id:
        return None
    key = name_or_id.strip()
    return (_by_id(data, key) or _by_slug(data, key)
            or _by_slug(data, _slugify(key)))


# --- write -------------------------------------------------------------------
def _new_skill(spec: dict) -> dict:
    name = (spec.get("name") or "Untitled skill").strip()
    slug = _slugify(name) or uuid.uuid4().hex[:8]
    skill = {
        "id": f"skill_{uuid.uuid4().hex[:12]}",
        "type": "agent_skill",
        "layer": LAYER,
        "name": name,
        "slug": slug,
        "description": (spec.get("description") or "").strip(),
        "instructions": (spec.get("instructions") or "").strip(),
        "steps": _str_list(spec.get("steps")),
        "triggers": _str_list(spec.get("triggers")),
        "anti_triggers": _str_list(spec.get("anti_triggers")),
        "tools": _str_list(spec.get("tools")),
        "success_criteria": spec.get("success_criteria") or {},
        "tests": spec.get("tests") if isinstance(spec.get("tests"), list) else [],
        "provenance": spec.get("provenance") or {},
        "status": DRAFT,
        "version": 1,
        "revisions": 0,
        "eval": None,
        "human": None,
        "history": [],
        "preview": _preview(spec.get("description", "")),
        "embedding": None,
        "created_at": _now(),
        "updated_at": _now(),
        "accepted_at": None,
    }
    skill["embedding"] = _embed(_embed_text(skill))
    return skill


def upsert_skill(spec: dict, *, where_id: str | None = None) -> dict | None:
    """Create a new skill draft, or revise an existing one in place.

    When a skill of the same slug already exists (or ``where_id`` is given), the
    skill is *revised*: its fields are replaced, ``version`` and ``revisions``
    bump, status resets to ``draft``, and the prior eval/human review are
    cleared (they must be re-earned). This is how the build loop folds a human's
    revision notes back into a fresh draft. Returns the public skill record."""
    name = (spec.get("name") or "").strip()
    if not name and not spec.get("instructions"):
        return None
    slug = _slugify(name)
    with _lock:
        data = _load()
        existing = (_by_id(data, where_id) if where_id else None) or (
            _by_slug(data, slug) if slug else None)
        fresh = _new_skill(spec)
        if existing is None:
            data["skills"].append(fresh)
            _save(data)
            return _public(fresh)
        # Revise in place: keep id/created_at/history, bump version, reset state.
        history = existing.get("history", [])
        history.append({"at": _now(), "phase": "revise",
                        "outcome": f"v{existing.get('version', 1)} → v{existing.get('version', 1) + 1}",
                        "note": _preview(spec.get("revision_note", ""), 200)})
        existing.update({
            "name": fresh["name"], "slug": fresh["slug"],
            "description": fresh["description"], "instructions": fresh["instructions"],
            "steps": fresh["steps"], "triggers": fresh["triggers"],
            "anti_triggers": fresh["anti_triggers"], "tools": fresh["tools"],
            "success_criteria": fresh["success_criteria"], "tests": fresh["tests"],
            "provenance": fresh["provenance"] or existing.get("provenance", {}),
            "status": DRAFT, "eval": None, "human": None,
            "version": existing.get("version", 1) + 1,
            "revisions": existing.get("revisions", 0) + 1,
            "preview": fresh["preview"], "embedding": fresh["embedding"],
            "history": history, "updated_at": _now(), "accepted_at": None,
        })
        _save(data)
        return _public(existing)


def append_history(skill_id: str, phase: str, outcome: str, *, note: str = "") -> bool:
    with _lock:
        data = _load()
        s = _by_id(data, skill_id)
        if s is None:
            return False
        s.setdefault("history", []).append(
            {"at": _now(), "phase": phase, "outcome": outcome, "note": _preview(note, 240)})
        s["updated_at"] = _now()
        _save(data)
        return True


def set_status(skill_id: str, status: str, *, note: str = "") -> dict | None:
    if status not in STATUSES:
        return None
    with _lock:
        data = _load()
        s = _by_id(data, skill_id)
        if s is None:
            return None
        s["status"] = status
        s["updated_at"] = _now()
        if status == ACCEPTED:
            s["accepted_at"] = _now()
        s.setdefault("history", []).append(
            {"at": _now(), "phase": "status", "outcome": status, "note": _preview(note, 240)})
        _save(data)
        return _public(s)


def record_eval(skill_id: str, report: dict) -> dict | None:
    """Attach an eval report (from ``skill_eval``) and advance status.

    The report's ``gate`` decides the next state:
      - ``accept`` → ``pending_review`` (passed the automated bar; a human signs off)
      - ``reject`` → ``rejected``
      - ``review`` → ``pending_review`` (borderline; defer to a human)
    """
    with _lock:
        data = _load()
        s = _by_id(data, skill_id)
        if s is None:
            return None
        s["eval"] = report
        gate = (report or {}).get("gate", "review")
        s["status"] = REJECTED if gate == "reject" else PENDING_REVIEW
        s["updated_at"] = _now()
        det = report.get("deterministic") or {}
        rub = report.get("rubric") or {}
        s.setdefault("history", []).append({
            "at": _now(), "phase": "eval", "outcome": gate,
            "note": f"deterministic {det.get('passed', '?')}/{det.get('total', '?')} · "
                    f"rubric {rub.get('mean', '?')}"})
        _save(data)
        return _public(s)


def record_human_review(skill_id: str, *, decision: str, score: float | None = None,
                        notes: str = "", reviewer: str = "user") -> dict | None:
    """Fold a human's alignment decision into the skill (the human-in-the-loop step).

    ``decision`` is one of ``accept`` / ``reject`` / ``revise``:
      - ``accept`` → status ``accepted``; the skill goes live in the library.
      - ``reject`` → status ``rejected``.
      - ``revise`` → status ``needs_revision``; ``notes`` guide the next build.
    Records a ``human`` block (score + notes) and an ``alignment`` flag comparing
    the human's call to the automated gate, so the loop can report how well the
    eval gate tracked human judgment."""
    decision = (decision or "").strip().lower()
    if decision not in {"accept", "reject", "revise"}:
        return None
    status = {"accept": ACCEPTED, "reject": REJECTED, "revise": NEEDS_REVISION}[decision]
    with _lock:
        data = _load()
        s = _by_id(data, skill_id)
        if s is None:
            return None
        gate = ((s.get("eval") or {}).get("gate")) or "review"
        gate_said_yes = gate == "accept"
        human_said_yes = decision == "accept"
        s["human"] = {
            "decision": decision,
            "score": float(score) if score is not None else None,
            "notes": notes.strip(),
            "reviewer": reviewer or "user",
            "reviewed_at": _now(),
            "gate": gate,
            "aligned_with_gate": gate_said_yes == human_said_yes,
        }
        s["status"] = status
        s["updated_at"] = _now()
        if status == ACCEPTED:
            s["accepted_at"] = _now()
        s.setdefault("history", []).append(
            {"at": _now(), "phase": "human_review", "outcome": decision,
             "note": _preview(notes, 240)})
        _save(data)
        return _public(s)


def forget(skill_id: str) -> bool:
    with _lock:
        data = _load()
        before = len(data["skills"])
        data["skills"] = [s for s in data["skills"] if s.get("id") != skill_id]
        changed = len(data["skills"]) != before
        if changed:
            _save(data)
        return changed


def clear() -> None:
    with _lock:
        _save(_empty())


# --- read --------------------------------------------------------------------
def get_skill(name_or_id: str) -> dict | None:
    with _lock:
        data = _load()
        s = _resolve(data, name_or_id)
    return _public(s) if s else None


def list_skills(*, status: str | None = None, limit: int = 100) -> list[dict]:
    with _lock:
        data = _load()
        skills = list(data["skills"])
    if status:
        skills = [s for s in skills if s.get("status") == status]
    # Group by status (accepted first, rejected last); within a group, newest first.
    order = {ACCEPTED: 0, PENDING_REVIEW: 1, NEEDS_REVISION: 2,
             EVALUATED: 3, DRAFT: 4, REJECTED: 5}
    skills.sort(key=lambda s: s.get("updated_at", ""), reverse=True)  # stable: newest first
    skills.sort(key=lambda s: order.get(s.get("status"), 9))          # then by status group
    return [_public(s) for s in skills[:max(1, limit)]]


def pending_review(limit: int = 100) -> list[dict]:
    """Skills awaiting human alignment (the review queue)."""
    return list_skills(status=PENDING_REVIEW, limit=limit)


def recall(query: str, *, k: int = 6, statuses: set | None = None,
           min_score: float | None = None) -> list[dict]:
    """Return up to ``k`` skills whose trigger surface best matches ``query``.

    This is how an agent decides *which* skill applies to a task — matching the
    task description against each skill's name/description/triggers. Defaults to
    accepted (live) skills only. Scores semantically when embeddings are
    available, else by keyword overlap."""
    statuses = statuses if statuses is not None else ACTIVE_STATUSES
    with _lock:
        data = _load()
        skills = [s for s in data["skills"] if s.get("status") in statuses]
    if not skills:
        return []
    qv = _embed(query) if query else None
    semantic = qv is not None
    floor = min_score if min_score is not None else (_SEMANTIC_FLOOR if semantic else _KEYWORD_FLOOR)
    scored = []
    for s in skills:
        if semantic and s.get("embedding"):
            sim = _cosine(qv, s["embedding"])
        else:
            sim = _keyword_sim(query, _embed_text(s))
        scored.append((sim, s))
    scored.sort(key=lambda x: -x[0])
    out = []
    for sim, s in scored[:k]:
        if sim < floor:
            continue
        row = _public(s)
        row["similarity"] = round(sim, 4)
        out.append(row)
    return out


def find_skill(query: str, *, statuses: set | None = None) -> dict | None:
    """Best single skill match for a task, or None."""
    hits = recall(query, k=1, statuses=statuses)
    return hits[0] if hits else None


def stats() -> dict:
    with _lock:
        data = _load()
        skills = list(data["skills"])
    by_status: dict[str, int] = {}
    for s in skills:
        by_status[s.get("status", DRAFT)] = by_status.get(s.get("status", DRAFT), 0) + 1
    aligned = [s for s in skills if (s.get("human") or {}).get("aligned_with_gate") is not None]
    n_aligned = sum(1 for s in aligned if s["human"]["aligned_with_gate"])
    return {
        "total": len(skills),
        "accepted": by_status.get(ACCEPTED, 0),
        "pending_review": by_status.get(PENDING_REVIEW, 0),
        "needs_revision": by_status.get(NEEDS_REVISION, 0),
        "rejected": by_status.get(REJECTED, 0),
        "draft": by_status.get(DRAFT, 0) + by_status.get(EVALUATED, 0),
        "by_status": by_status,
        "embedded": sum(1 for s in skills if s.get("embedding")),
        "embeddings": embeddings_on(),
        # How often the human's accept/reject matched the automated gate.
        "gate_human_alignment": (round(n_aligned / len(aligned), 3) if aligned else None),
        "reviewed": len(aligned),
    }


# --- SKILL.md rendering / export --------------------------------------------
def to_skill_md(skill: dict) -> str:
    """Render a skill as a portable SKILL.md document (YAML front-matter + body).

    Mirrors the conventional agent-skill on-disk format: a ``name``/``description``
    header an agent reads to decide whether to invoke the skill, followed by the
    instructions and supporting sections."""
    s = skill
    name = s.get("name", "Untitled skill")
    desc = (s.get("description") or "").replace("\n", " ").strip()
    lines = ["---", f"name: {name}", f"description: {desc}"]
    if s.get("tools"):
        lines.append("tools: " + ", ".join(s["tools"]))
    lines += [f"status: {s.get('status', DRAFT)}", f"version: {s.get('version', 1)}", "---", ""]
    lines.append(f"# {name}\n")
    if desc:
        lines.append(desc + "\n")
    if s.get("triggers"):
        lines.append("## When to use")
        lines += [f"- {t}" for t in s["triggers"]]
        lines.append("")
    if s.get("anti_triggers"):
        lines.append("## When NOT to use")
        lines += [f"- {t}" for t in s["anti_triggers"]]
        lines.append("")
    lines.append("## Instructions")
    lines.append((s.get("instructions") or "").strip() + "\n")
    if s.get("steps"):
        lines.append("## Steps")
        lines += [f"{i}. {step}" for i, step in enumerate(s["steps"], 1)]
        lines.append("")
    crit = s.get("success_criteria") or {}
    if crit:
        lines.append("## Success criteria")
        for key in ("outcome", "process", "style", "efficiency"):
            val = crit.get(key)
            if val:
                rendered = ", ".join(val) if isinstance(val, list) else str(val)
                lines.append(f"- **{key.title()}:** {rendered}")
        lines.append("")
    prov = s.get("provenance") or {}
    if prov.get("source_titles") or prov.get("chunk_ids"):
        lines.append("## Grounded in")
        for t in (prov.get("source_titles") or [])[:10]:
            lines.append(f"- {t}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_skill(name_or_id: str) -> dict:
    """Write a skill's SKILL.md to ``data/skills/<slug>/SKILL.md``. Returns its path."""
    skill = get_skill(name_or_id)
    if not skill:
        return {"ok": False, "error": "skill not found"}
    folder = SKILL_MD_DIR / (skill.get("slug") or "skill")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(to_skill_md(skill), encoding="utf-8")
    return {"ok": True, "path": str(path), "slug": skill.get("slug")}
