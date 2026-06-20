"""Cross-document RAG evaluation.

Builds a reusable evaluation dataset whose questions each require synthesizing
across MULTIPLE documents (so a single chunk/doc is not enough), commits it as a
template, uploads it to LangSmith, and runs an experiment scoring multi-source
**retrieval recall** + RAGAS metrics + an LLM-judged synthesis correctness.

The synthesis judge is graded against real evidence, not the judge's priors:
each judge sees (a) excerpts of the gold source documents and (b) a gold
``key_points`` reference for the question. Human labels in
``eval/crossdoc_human_labels.json`` supply that reference and a human score; the
run reports how closely each LLM judge tracks the human (the human-in-the-loop
calibration step of the agent-improvement loop).

The dataset is referenced by ID (rename-proof) like the single-doc one.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import config
import rag
import rag_experiment
from providers import build_chat_model, resolve_provider_model, resolve_judge, judge_panel
from vectorstore import VectorStore

TEMPLATE_PATH = Path(__file__).parent / "eval" / "rag_eval_dataset_crossdoc.json"
HUMAN_LABELS_PATH = Path(__file__).parent / "eval" / "crossdoc_human_labels.json"
DATASET_NAME = "Cross-document RAG eval"

# RAGAS judge must avoid reasoning models (they burn the token budget on hidden
# reasoning); this matches the selection used when wiring the RAGAS evaluators.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def load_template() -> dict:
    if TEMPLATE_PATH.exists():
        return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return {}


def _dataset_id() -> str | None:
    return os.environ.get("LANGSMITH_CROSSDOC_DATASET_ID") or load_template().get("dataset_id")


def _docs(vs: VectorStore) -> dict:
    docs: dict[str, dict] = {}
    for s in vs.sections:
        u = s.get("url")
        if u and u not in docs:
            docs[u] = {"title": s.get("title"), "overview": s.get("overview", "") or s.get("summary", "")}
    return docs


_GEN_PROMPT = """You are building a CROSS-DOCUMENT evaluation set for a RAG system.
Below are documents in a knowledge library (each line: URL :: title :: overview).

Write {n} questions that each REQUIRE combining information from 2-3 DIFFERENT
documents to answer well — comparisons, syntheses, shared themes/differences,
"how do X and Y relate", trends across papers. Prefer thematically related
documents. Do NOT write questions answerable from a single document.

For each question also list 3-6 key_points: the essential, factual points a
correct answer MUST cover, each grounded in the named documents. These become the
gold reference the answer is graded against.

Documents:
{docs}

Return ONLY JSON:
{{"questions": [{{"question": "<question>", "expected_urls": ["<url>", ...],
"key_points": ["<point>", "<point>", ...]}}, ...]}}
Each expected_urls must list the 2-3 documents needed, using the exact URLs above."""


def build_eval_set(*, vs_name: str = "library", provider: str = "auto",
                   model: str | None = None, n_questions: int = 12) -> list[dict]:
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise rag.RagError("No LLM provider configured for cross-doc generation.")
    vs = VectorStore.load(vs_name)
    docs = _docs(vs)
    if len(docs) < 2:
        raise rag.RagError("Need at least 2 ingested documents for a cross-document eval.")
    url_set = set(docs)
    listing = "\n".join(f"- {u} :: {d['title']} :: {d['overview'][:240]}" for u, d in docs.items())
    chat = build_chat_model(rp, rm, max_tokens=2400)
    raw = rag._gen(chat, _GEN_PROMPT.format(n=n_questions, docs=listing))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    items = []
    if m:
        try:
            items = json.loads(m.group()).get("questions", []) or []
        except Exception:  # noqa: BLE001
            items = []
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        exp = [u for u in (it.get("expected_urls") or []) if u in url_set]
        exp = list(dict.fromkeys(exp))  # dedupe, keep order
        kps = [str(p).strip() for p in (it.get("key_points") or []) if str(p).strip()]
        if q and len(exp) >= 2:
            out.append({"question": q, "expected_urls": exp,
                        "titles": [docs[u]["title"] for u in exp], "key_points": kps})
    return out


def save_template(eval_set: list[dict], *, dataset_id: str | None = None) -> dict:
    def _outputs(e: dict) -> dict:
        out = {"expected_urls": e["expected_urls"], "titles": e.get("titles", [])}
        if e.get("key_points"):
            out["key_points"] = e["key_points"]
        return out

    tmpl = {
        "name": DATASET_NAME,
        "dataset_id": dataset_id or load_template().get("dataset_id"),
        "langsmith_project_id": os.environ.get("LANGSMITH_PROJECT_ID", load_template().get("langsmith_project_id")),
        "description": "Reusable CROSS-DOCUMENT RAG eval: each question requires synthesizing "
                       "across 2-3 documents. outputs.expected_urls lists the required sources; "
                       "outputs.key_points is the gold reference a correct answer should cover.",
        "schema": {"inputs": ["question"], "outputs": ["expected_urls", "titles", "key_points"]},
        "examples": [{"inputs": {"question": e["question"]}, "outputs": _outputs(e)} for e in eval_set],
    }
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_PATH.write_text(json.dumps(tmpl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return tmpl


def sync_dataset(client=None, *, eval_set: list[dict] | None = None,
                 provider: str = "auto", model: str | None = None) -> dict:
    """Ensure the cross-doc dataset exists in LangSmith (by ID, else from template,
    else generate). Returns {id, name, examples}."""
    client = client or rag_experiment._client()
    dsid = _dataset_id()
    if dsid:
        try:
            ds = client.read_dataset(dataset_id=dsid)
            return {"id": str(ds.id), "name": ds.name, "examples": ds.example_count}
        except Exception:  # noqa: BLE001
            pass
    tmpl = load_template()
    examples = ([{"inputs": e["inputs"], "outputs": e["outputs"]} for e in tmpl.get("examples", [])]
                or [{"inputs": {"question": e["question"]},
                     "outputs": {k: v for k, v in (("expected_urls", e["expected_urls"]),
                                                   ("titles", e.get("titles", [])),
                                                   ("key_points", e.get("key_points", []))) if v}}
                    for e in (eval_set or build_eval_set(provider=provider, model=model))])
    if not examples:
        raise rag.RagError("No cross-document examples available.")
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(dataset_name=DATASET_NAME,
                              description=tmpl.get("description", "Cross-document RAG eval."))
        client.create_examples(dataset_name=DATASET_NAME, examples=examples)
    ds = client.read_dataset(dataset_name=DATASET_NAME)
    # Persist the resolved id back into the template for reuse.
    if str(ds.id) != (tmpl.get("dataset_id") or ""):
        save_template([{"question": e["inputs"]["question"], **e["outputs"]} for e in examples],
                      dataset_id=str(ds.id))
    return {"id": str(ds.id), "name": ds.name, "examples": ds.example_count or len(examples)}


# --- gold source content (so the judge grades against evidence, not priors) ---
def _doc_digests(vs: VectorStore, *, per_doc_chars: int = 900) -> dict:
    """Map url -> a compact excerpt of that document's own material (overview +
    a couple of representative chunk snippets), used to ground the judge."""
    secs_by_url: dict[str, list] = {}
    chunks_by_url: dict[str, list] = {}
    for s in vs.sections:
        u = s.get("url")
        if u:
            secs_by_url.setdefault(u, []).append(s)
    for c in vs.chunks:
        u = c.get("url")
        if u:
            chunks_by_url.setdefault(u, []).append(c)
    digests: dict[str, str] = {}
    for u in set(secs_by_url) | set(chunks_by_url):
        overview = ""
        for s in secs_by_url.get(u, []):
            overview = s.get("overview") or s.get("summary") or ""
            if overview:
                break
        snippets = " … ".join((c.get("text") or "")[:300] for c in chunks_by_url.get(u, [])[:2])
        body = (overview + ("\n" + snippets if snippets else "")).strip()
        if body:
            digests[u] = body[:per_doc_chars]
    return digests


def _sources_content(expected_urls: list[str], digests: dict) -> str:
    blocks = [f"[{u}]\n{digests[u]}" for u in expected_urls if digests.get(u)]
    return "\n\n".join(blocks)


# --- human labels: the gold reference + human score (calibration loop) --------
def load_human_labels() -> dict:
    """Return {question -> {key_points, human_score, reviewed_answer, notes, reviewed}}."""
    if HUMAN_LABELS_PATH.exists():
        try:
            data = json.loads(HUMAN_LABELS_PATH.read_text(encoding="utf-8"))
            return data.get("labels", {}) if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_human_labels(labels: dict, *, description: str | None = None) -> dict:
    payload = {
        "description": description or (
            "Human-in-the-loop calibration for the cross-document synthesis judge. "
            "key_points = gold reference each answer is graded against; human_score = a "
            "reviewer's 0-1 rating of reviewed_answer. The crossdoc run reports how closely "
            "each LLM judge tracks human_score (fresh pairs only; an answer that no longer "
            "matches reviewed_answer is flagged stale -> re-review)."),
        "labels": labels,
    }
    HUMAN_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_LABELS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(HUMAN_LABELS_PATH), "labeled": len(labels)}


_KEYPOINTS_PROMPT = """You are writing the GOLD reference for a cross-document RAG eval question.
List the 3-6 essential points a correct answer MUST cover, each grounded in the
source material below. Be specific and factual; one short sentence per point. Do
not invent facts beyond the source material.

Question: {question}

Source material:
{content}

Return ONLY JSON: {{"key_points": ["<point>", "<point>", ...]}}"""


def _gen_key_points(chat, question: str, content: str) -> list[str]:
    raw = rag._gen(chat, _KEYPOINTS_PROMPT.format(question=question, content=content or "(no content)"))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        return [str(p).strip() for p in json.loads(m.group()).get("key_points", []) if str(p).strip()]
    except Exception:  # noqa: BLE001
        return []


def scaffold_human_labels(*, provider: str = "auto", model: str | None = None,
                          vs_name: str = "library", overwrite: bool = False) -> dict:
    """Seed eval/crossdoc_human_labels.json with LLM-drafted key_points for every
    dataset question lacking a label, leaving human_score null for a human to fill
    in. This is the 'seed from traces, human curates' step of the loop."""
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise rag.RagError("No LLM provider configured for key-point drafting.")
    tmpl = load_template()
    examples = tmpl.get("examples", [])
    if not examples:
        raise rag.RagError("No cross-document examples available to label.")
    digests = _doc_digests(VectorStore.load(vs_name))
    labels = load_human_labels()
    chat = build_chat_model(rp, rm, max_tokens=600)
    added = 0
    for e in examples:
        q = e["inputs"]["question"]
        if q in labels and not overwrite:
            continue
        urls = e["outputs"].get("expected_urls", [])
        kp = e["outputs"].get("key_points") or _gen_key_points(chat, q, _sources_content(urls, digests))
        labels[q] = {"key_points": kp, "human_score": None, "reviewed_answer": "",
                     "notes": "", "reviewed": False}
        added += 1
    res = save_human_labels(labels)
    res.update({"added": added, "needs_review": sum(1 for v in labels.values() if not v.get("reviewed"))})
    return res


# --- multi-expected evaluators ----------------------------------------------
def _retrieval_recall(run, example):
    got = set((run.outputs or {}).get("retrieved_urls") or [])
    exp = set((example.outputs or {}).get("expected_urls") or [])
    return {"key": "retrieval_recall", "score": (len(got & exp) / len(exp)) if exp else 0.0}


def _retrieval_any(run, example):
    got = set((run.outputs or {}).get("retrieved_urls") or [])
    exp = set((example.outputs or {}).get("expected_urls") or [])
    return {"key": "retrieval_any_hit", "score": 1.0 if (got & exp) else 0.0}


def _make_human_score(human_labels: dict):
    """Surface the human gold score as a per-row column, next to each judge."""
    def human_score(run, example):
        q = (example.inputs or {}).get("question", "")
        hs = (human_labels.get(q) or {}).get("human_score")
        return {"key": "human_score", "score": float(hs) if hs is not None else None}
    return human_score


_CROSSDOC_JUDGE = """Grade this cross-document RAG answer.

Question: {question}
Documents that should be synthesized: {sources}

Source material (excerpts from those documents):
{sources_content}

Reference key points a correct answer should cover:
{reference}

Answer under test:
{answer}

Grade the answer on correct, well-grounded synthesis across the documents:
- Reward claims supported by the source material and coverage of the reference key points.
- Penalize unsupported or hallucinated claims, single-source answers, and refusals.
Grade ONLY against the source material and reference key points when they are
provided; do not rely on outside knowledge. Where a section reads "(none provided)",
fall back to judging the answer's internal consistency with the listed documents.

Score 0.0-1.0 (1.0 = correct, well-grounded synthesis; 0.0 = wrong, single-source, or missing).
Return ONLY JSON: {{"score": <float>, "reason": "<one sentence>"}}"""


def _parse_judge(raw: str):
    if not raw or not raw.strip():
        return None, "empty response"
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            j = json.loads(m.group())
            return float(j.get("score", 0.0)), j.get("reason", "")
        except Exception:  # noqa: BLE001
            pass  # fall through to bare-number rescue
    m2 = re.search(r"(?<![\w.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\w.])", raw)
    if m2:
        try:
            return float(m2.group(1)), "parsed bare score"
        except Exception:  # noqa: BLE001
            pass
    return None, "unparseable judge response"


def _make_synthesis_judge(provider, model, *, key="crossdoc_correctness",
                          digests: dict | None = None, human_labels: dict | None = None):
    # Larger budget: gpt-5* spend "reasoning" tokens out of max_completion_tokens.
    judge = build_chat_model(provider, model, max_tokens=2048)
    digests = digests or {}
    human_labels = human_labels or {}

    def crossdoc_correctness(run, example):
        inp, exp = example.inputs or {}, example.outputs or {}
        q = inp.get("question", "")
        expected_urls = exp.get("expected_urls", [])
        sources = "; ".join(f"{t} ({u})" for t, u in
                            zip(exp.get("titles", []), expected_urls))
        sources_content = _sources_content(expected_urls, digests) or "(none provided)"
        kp = (human_labels.get(q) or {}).get("key_points") or exp.get("key_points") or []
        reference = "\n".join(f"- {p}" for p in kp) if kp else "(none provided)"
        prompt = _CROSSDOC_JUDGE.format(
            question=q, sources=sources, sources_content=sources_content,
            reference=reference, answer=(run.outputs or {}).get("answer", ""))
        # Retry once: some judges (e.g. flaky/throttled endpoints) intermittently
        # return an empty body or transient 5xx; a single retry recovers most.
        score, reason = None, ""
        for _ in range(2):
            try:
                raw = rag._gen(judge, prompt)
            except Exception as exc:  # noqa: BLE001
                reason = f"judge error: {type(exc).__name__}"
                continue
            score, reason = _parse_judge(raw)
            if score is not None:
                break
        # score=None -> excluded from this judge's mean (NOT counted as 0.0), so a
        # flaky judge never unfairly penalizes the answer under test.
        return {"key": key, "score": score, "comment": reason}

    crossdoc_correctness.__name__ = key
    return crossdoc_correctness


# --- human <-> judge alignment (calibration readout) -------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _judge_alignment(rows, panel_keys: list[str], human_labels: dict) -> dict:
    """How closely each LLM judge tracks the human score, over examples whose
    current answer still matches the reviewed_answer (fresh). Stale examples (the
    answer changed since review) are counted but excluded from the means."""
    per_judge: dict[str, list] = {k: [] for k in panel_keys}
    n_labeled = n_stale = 0
    for row in rows:
        try:
            ex = row.get("example") if isinstance(row, dict) else None
            run = row.get("run") if isinstance(row, dict) else None
            q = (getattr(ex, "inputs", None) or {}).get("question", "")
            lab = human_labels.get(q) or {}
            hs = lab.get("human_score")
            if hs is None:
                continue
            n_labeled += 1
            answer = (getattr(run, "outputs", None) or {}).get("answer", "")
            reviewed = lab.get("reviewed_answer") or ""
            fresh = (not reviewed) or (_norm(answer) == _norm(reviewed))
            if not fresh:
                n_stale += 1
                continue
            ev = row.get("evaluation_results", {}) if isinstance(row, dict) else {}
            scores = {getattr(r, "key", None): getattr(r, "score", None)
                      for r in (ev.get("results", []) or [])}
            for k in panel_keys:
                s = scores.get(k)
                if s is not None:
                    per_judge[k].append((float(s), float(hs)))
        except Exception:  # noqa: BLE001
            continue
    out_judges = {}
    for k, pairs in per_judge.items():
        if not pairs:
            continue
        mae = sum(abs(j - h) for j, h in pairs) / len(pairs)
        within = sum(1.0 for j, h in pairs if abs(j - h) <= 0.2) / len(pairs)
        out_judges[k] = {"n": len(pairs), "mae": round(mae, 3),
                         "alignment": round(1.0 - mae, 3), "within_0.2": round(within, 3)}
    aligns = {k: v["alignment"] for k, v in out_judges.items()}
    return {"n_labeled": n_labeled, "n_stale": n_stale, "per_judge": out_judges,
            "panel_mean_alignment": round(sum(aligns.values()) / len(aligns), 3) if aligns else None,
            "best_aligned": max(aligns, key=aligns.get) if aligns else None,
            "worst_aligned": min(aligns, key=aligns.get) if aligns else None}


def _make_target(provider, model, k, rerank, mmr=False):
    def target(inputs: dict) -> dict:
        res = rag.answer_with_contexts(inputs["question"], provider=provider, model=model,
                                       k=k, rerank_hits=rerank, mmr=mmr)
        return {"answer": res["answer"], "retrieved_contexts": res["contexts"],
                "retrieved_urls": res["urls"]}
    return target


def run_experiment(*, provider: str = "auto", model: str | None = None,
                   judge_provider: str | None = None, judge_model: str | None = None,
                   k: int = 8, rerank: bool = True, ragas: bool = True,
                   mmr: bool = False,
                   n_questions: int = 12, max_concurrency: int = 1) -> dict:
    from langsmith import evaluate
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise rag.RagError("No LLM provider configured for the cross-document experiment.")
    # Panel of judges from DIFFERENT families than the generator (averages out
    # any single judge's idiosyncratic strictness). Honors an explicit override.
    if judge_provider:
        jp, jm, _ = resolve_judge(rp, rm, judge_provider, judge_model)
        panel = [(jp, jm)]
    else:
        panel = judge_panel(rp) or [resolve_judge(rp, rm)[:2]]
    # The RAGAS judge avoids reasoning models (they spend the budget on hidden
    # reasoning). Resolve it up front so metadata records the judge ACTUALLY used.
    rjp, rjm = (next(((p, m) for p, m in panel if not m.startswith(_REASONING_PREFIXES)), panel[0])
                if ragas else (None, None))
    config.ensure_tracing_project()

    client = rag_experiment._client()
    # Generate + save the template the first time, so it is committable + reusable.
    if not load_template().get("examples"):
        save_template(build_eval_set(provider=rp, model=rm, n_questions=n_questions))
    ds = sync_dataset(client, provider=rp, model=rm)
    name = client.read_dataset(dataset_id=ds["id"]).name

    # Gold evidence + human reference, loaded once and shared by every judge.
    try:
        digests = _doc_digests(VectorStore.load("library"))
    except Exception:  # noqa: BLE001
        digests = {}
    human_labels = load_human_labels()

    # One synthesis-correctness evaluator per panel judge (separate columns in the
    # UI). Key by MODEL (not provider) so same-family judges don't collide.
    def _ckey(m):
        return "correctness_" + re.sub(r"[^a-z0-9]+", "-", m.lower()).strip("-")
    panel_keys = [_ckey(jm) for _, jm in panel]
    evaluators = [_retrieval_recall, _retrieval_any, _make_human_score(human_labels)]
    evaluators += [_make_synthesis_judge(jp, jm, key=_ckey(jm), digests=digests,
                                         human_labels=human_labels) for jp, jm in panel]
    if ragas:
        try:
            import ragas_eval
            evaluators = ragas_eval.make_ragas_evaluators(rjp, rjm) + evaluators
        except Exception:  # noqa: BLE001
            pass  # ragas optional
    tag = "mmr" if mmr else ("rerank" if rerank else "base")
    results = evaluate(
        _make_target(rp, rm, k, rerank, mmr=mmr),
        data=name,
        evaluators=evaluators,
        experiment_prefix=f"crossdoc-{tag}",
        metadata={"eval": "crossdoc", "k": k, "rerank": rerank, "mmr": mmr,
                  "ragas": ragas, "model": f"{rp}/{rm}",
                  "judge_panel": [f"{p}/{m}" for p, m in panel],
                  "ragas_judge": (f"{rjp}/{rjm}" if rjp else None),
                  "human_labeled": sum(1 for v in human_labels.values()
                                       if v.get("human_score") is not None),
                  "dataset_id": ds["id"]},
        client=client,
        max_concurrency=max_concurrency,
        blocking=True,
    )
    rows = list(results)
    agg = rag_experiment._aggregate(rows)
    means = agg.get("means", {})
    panel_scores = [means[k_] for k_ in panel_keys if k_ in means]
    panel_mean = round(sum(panel_scores) / len(panel_scores), 3) if panel_scores else None
    alignment = _judge_alignment(rows, panel_keys, human_labels)
    dataset_url = None
    try:
        dataset_url = getattr(client.read_dataset(dataset_id=ds["id"]), "url", None)
    except Exception:  # noqa: BLE001
        pass
    return {"experiment_name": getattr(results, "experiment_name", None),
            "dataset_id": ds["id"], "dataset_url": dataset_url,
            "metrics": means, "correctness_panel_mean": panel_mean,
            "judge_alignment": alignment, "n": agg.get("n", 0),
            "rerank": rerank, "mmr": mmr, "k": k,
            "generator": f"{rp}/{rm}", "judge_panel": [f"{p}/{m}" for p, m in panel]}
