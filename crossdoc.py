"""Cross-document RAG evaluation.

Builds a reusable evaluation dataset whose questions each require synthesizing
across MULTIPLE documents (so a single chunk/doc is not enough), commits it as a
template, uploads it to LangSmith, and runs an experiment scoring multi-source
**retrieval recall** + RAGAS metrics + an LLM-judged synthesis correctness.

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
from providers import build_chat_model, resolve_provider_model, resolve_judge
from vectorstore import VectorStore

TEMPLATE_PATH = Path(__file__).parent / "eval" / "rag_eval_dataset_crossdoc.json"
DATASET_NAME = "Cross-document RAG eval"


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

Documents:
{docs}

Return ONLY JSON:
{{"questions": [{{"question": "<question>", "expected_urls": ["<url>", "<url>", ...]}}, ...]}}
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
    chat = build_chat_model(rp, rm, max_tokens=2000)
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
        if q and len(exp) >= 2:
            out.append({"question": q, "expected_urls": exp,
                        "titles": [docs[u]["title"] for u in exp]})
    return out


def save_template(eval_set: list[dict], *, dataset_id: str | None = None) -> dict:
    tmpl = {
        "name": DATASET_NAME,
        "dataset_id": dataset_id or load_template().get("dataset_id"),
        "langsmith_project_id": os.environ.get("LANGSMITH_PROJECT_ID", load_template().get("langsmith_project_id")),
        "description": "Reusable CROSS-DOCUMENT RAG eval: each question requires synthesizing "
                       "across 2-3 documents. outputs.expected_urls lists the required sources.",
        "schema": {"inputs": ["question"], "outputs": ["expected_urls", "titles"]},
        "examples": [{"inputs": {"question": e["question"]},
                      "outputs": {"expected_urls": e["expected_urls"], "titles": e.get("titles", [])}}
                     for e in eval_set],
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
                     "outputs": {"expected_urls": e["expected_urls"], "titles": e.get("titles", [])}}
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


# --- multi-expected evaluators ----------------------------------------------
def _retrieval_recall(run, example):
    got = set((run.outputs or {}).get("retrieved_urls") or [])
    exp = set((example.outputs or {}).get("expected_urls") or [])
    return {"key": "retrieval_recall", "score": (len(got & exp) / len(exp)) if exp else 0.0}


def _retrieval_any(run, example):
    got = set((run.outputs or {}).get("retrieved_urls") or [])
    exp = set((example.outputs or {}).get("expected_urls") or [])
    return {"key": "retrieval_any_hit", "score": 1.0 if (got & exp) else 0.0}


_CROSSDOC_JUDGE = """Grade this cross-document RAG answer.

Question: {question}
Documents that should be synthesized: {sources}

Answer under test:
{answer}

Score 0.0-1.0 on whether the answer correctly synthesizes across the relevant
documents and answers the question (1.0 = correct, well-grounded synthesis;
0.0 = wrong or single-source/missing). Return ONLY JSON:
{{"score": <float>, "reason": "<one sentence>"}}"""


def _make_synthesis_judge(provider, model, key="crossdoc_correctness"):
    # Larger budget: gpt-5* spend "reasoning" tokens out of max_completion_tokens.
    judge = build_chat_model(provider, model, max_tokens=2048)

    def crossdoc_correctness(run, example):
        inp, exp = example.inputs or {}, example.outputs or {}
        sources = "; ".join(f"{t} ({u})" for t, u in
                            zip(exp.get("titles", []), exp.get("expected_urls", [])))
        raw = rag._gen(judge, _CROSSDOC_JUDGE.format(
            question=inp.get("question", ""), sources=sources,
            answer=(run.outputs or {}).get("answer", "")))
        score, reason = 0.0, ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                j = json.loads(m.group())
                score = float(j.get("score", 0.0)); reason = j.get("reason", "")
            except Exception:  # noqa: BLE001
                pass
        return {"key": key, "score": score, "comment": reason}

    crossdoc_correctness.__name__ = key
    return crossdoc_correctness


def _make_target(provider, model, k, rerank, graph_rag, mmr=False):
    def target(inputs: dict) -> dict:
        res = rag.answer_with_contexts(inputs["question"], provider=provider, model=model,
                                       k=k, graph_rag=graph_rag, rerank_hits=rerank, mmr=mmr)
        return {"answer": res["answer"], "retrieved_contexts": res["contexts"],
                "retrieved_urls": res["urls"]}
    return target


def run_experiment(*, provider: str = "auto", model: str | None = None,
                   judge_provider: str | None = None, judge_model: str | None = None,
                   k: int = 8, rerank: bool = True, graph_rag: bool = True, ragas: bool = True,
                   mmr: bool = False, n_questions: int = 12, max_concurrency: int = 1) -> dict:
    from langsmith import evaluate
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise rag.RagError("No LLM provider configured for the cross-document experiment.")
    # Panel of judges from DIFFERENT families than the generator (averages out
    # any single judge's idiosyncratic strictness). Honors an explicit override.
    from providers import judge_panel
    if judge_provider:
        jp, jm, _ = resolve_judge(rp, rm, judge_provider, judge_model)
        panel = [(jp, jm)]
    else:
        panel = judge_panel(rp) or [resolve_judge(rp, rm)[:2]]
    config.ensure_tracing_project()

    client = rag_experiment._client()
    # Generate + save the template the first time, so it is committable + reusable.
    if not load_template().get("examples"):
        save_template(build_eval_set(provider=rp, model=rm, n_questions=n_questions))
    ds = sync_dataset(client, provider=rp, model=rm)
    name = client.read_dataset(dataset_id=ds["id"]).name

    # One synthesis-correctness evaluator per panel judge (separate columns in the UI).
    panel_keys = [f"correctness_{jp}" for jp, _ in panel]
    evaluators = [_retrieval_recall, _retrieval_any]
    evaluators += [_make_synthesis_judge(jp, jm, key=f"correctness_{jp}") for jp, jm in panel]
    if ragas:
        try:
            import ragas_eval
            rjp, rjm = panel[0]  # RAGAS judge LLM = first panel member (bounds cost)
            evaluators = ragas_eval.make_ragas_evaluators(rjp, rjm) + evaluators
        except Exception:  # noqa: BLE001
            pass  # ragas optional
    tag = "mmr" if mmr else ("rerank" if rerank else "base")
    results = evaluate(
        _make_target(rp, rm, k, rerank, graph_rag, mmr=mmr),
        data=name,
        evaluators=evaluators,
        experiment_prefix=f"crossdoc-{tag}",
        metadata={"eval": "crossdoc", "k": k, "rerank": rerank, "mmr": mmr,
                  "ragas": ragas, "model": f"{rp}/{rm}",
                  "judge_panel": [f"{p}/{m}" for p, m in panel],
                  "ragas_judge": f"{panel[0][0]}/{panel[0][1]}", "dataset_id": ds["id"]},
        client=client,
        max_concurrency=max_concurrency,
        blocking=True,
    )
    agg = rag_experiment._aggregate(results)
    means = agg.get("means", {})
    panel_scores = [means[k_] for k_ in panel_keys if k_ in means]
    panel_mean = round(sum(panel_scores) / len(panel_scores), 3) if panel_scores else None
    dataset_url = None
    try:
        dataset_url = getattr(client.read_dataset(dataset_id=ds["id"]), "url", None)
    except Exception:  # noqa: BLE001
        pass
    return {"experiment_name": getattr(results, "experiment_name", None),
            "dataset_id": ds["id"], "dataset_url": dataset_url,
            "metrics": means, "correctness_panel_mean": panel_mean, "n": agg.get("n", 0),
            "rerank": rerank, "mmr": mmr, "k": k, "generator": f"{rp}/{rm}",
            "judge_panel": [f"{p}/{m}" for p, m in panel]}
