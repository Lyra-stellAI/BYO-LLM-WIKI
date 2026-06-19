"""Wire the RAG evaluation into a LangSmith dataset + experiment.

Creates (once) a dataset of (question -> expected source) examples from the
ingested library, then runs ``langsmith.evaluate`` with the RAG pipeline as the
target and three evaluators (retrieval hit, reciprocal rank, LLM-judged answer
correctness). The result is a proper LangSmith *experiment* with per-row scores
you can compare in the UI — e.g. base vs. re-ranked runs over the same dataset.
"""

from __future__ import annotations

import json
import re

import rag
from providers import build_chat_model, resolve_provider_model

_DEFAULT_DATASET = "Trend_analysis RAG eval"


def ensure_dataset(client, name: str, eval_set: list[dict]) -> str:
    """Create the dataset + examples if it does not exist yet (idempotent)."""
    if client.has_dataset(dataset_name=name):
        return name
    client.create_dataset(dataset_name=name,
                          description="RAG eval: question -> expected source document.")
    client.create_examples(dataset_name=name, examples=[{
        "inputs": {"question": item["question"]},
        "outputs": {"expected_url": item["expected_url"], "title": item.get("title", "")},
    } for item in eval_set])
    return name


def _make_target(provider, model, k, rerank, graph_rag):
    def target(inputs: dict) -> dict:
        res = rag.answer(inputs["question"], provider=provider, model=model, k=k,
                         graph_rag=graph_rag, rerank_hits=rerank)
        return {"answer": res.get("answer", ""),
                "retrieved_urls": [c.get("url") for c in res.get("citations", [])]}
    return target


def _retrieval_hit(run, example):
    urls = (run.outputs or {}).get("retrieved_urls") or []
    exp = (example.outputs or {}).get("expected_url")
    return {"key": "retrieval_hit", "score": 1.0 if exp in urls else 0.0}


def _reciprocal_rank(run, example):
    urls = (run.outputs or {}).get("retrieved_urls") or []
    exp = (example.outputs or {}).get("expected_url")
    rank = (urls.index(exp) + 1) if exp in urls else 0
    return {"key": "reciprocal_rank", "score": (1.0 / rank) if rank else 0.0}


def _make_answer_judge(provider, model):
    judge = build_chat_model(provider, model, max_tokens=200)

    def answer_correctness(run, example):
        inp, exp = example.inputs or {}, example.outputs or {}
        raw = rag._gen(judge, rag._JUDGE_PROMPT.format(
            question=inp.get("question", ""), title=exp.get("title", ""),
            url=exp.get("expected_url", ""), answer=(run.outputs or {}).get("answer", "")))
        score, reason = 0.0, ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                j = json.loads(m.group())
                score = float(j.get("score", 0.0))
                reason = j.get("reason", "")
            except Exception:  # noqa: BLE001
                pass
        return {"key": "answer_correctness", "score": score, "comment": reason}

    return answer_correctness


def _aggregate(results) -> dict:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n = 0
    for row in results:
        n += 1
        ev = row.get("evaluation_results", {}) if isinstance(row, dict) else {}
        for r in (ev.get("results", []) or []):
            key = getattr(r, "key", None)
            score = getattr(r, "score", None)
            if key is not None and score is not None:
                sums[key] = sums.get(key, 0.0) + float(score)
                counts[key] = counts.get(key, 0) + 1
    return {"n": n, "means": {k: round(sums[k] / counts[k], 3) for k in sums}}


def run_experiment(*, eval_set: list[dict] | None = None, dataset_name: str = _DEFAULT_DATASET,
                   provider: str = "auto", model: str | None = None, k: int = 6,
                   rerank: bool = False, graph_rag: bool = True,
                   max_questions: int = 15, max_concurrency: int = 2) -> dict:
    try:
        from langsmith import Client, evaluate
    except Exception as exc:  # noqa: BLE001
        raise rag.RagError(f"langsmith is required for experiments: {exc}") from exc

    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise rag.RagError("No LLM provider configured for the experiment.")

    client = Client()
    if eval_set is None:
        if client.has_dataset(dataset_name=dataset_name):
            eval_set = _examples_to_set(client, dataset_name)
        else:
            eval_set = rag.build_eval_set(provider=rp, model=rm, max_questions=max_questions)
    ensure_dataset(client, dataset_name, eval_set)

    tag = "rerank" if rerank else "base"
    results = evaluate(
        _make_target(rp, rm, k, rerank, graph_rag),
        data=dataset_name,
        evaluators=[_retrieval_hit, _reciprocal_rank, _make_answer_judge(rp, rm)],
        experiment_prefix=f"rag-{tag}",
        metadata={"k": k, "rerank": rerank, "graph_rag": graph_rag, "model": f"{rp}/{rm}"},
        client=client,
        max_concurrency=max_concurrency,
        blocking=True,
    )
    agg = _aggregate(results)
    experiment_name = getattr(results, "experiment_name", None)
    dataset_url = None
    try:
        dataset_url = getattr(client.read_dataset(dataset_name=dataset_name), "url", None)
    except Exception:  # noqa: BLE001
        pass
    return {"experiment_name": experiment_name, "dataset": dataset_name,
            "dataset_url": dataset_url, "rerank": rerank, "k": k,
            "metrics": agg.get("means", {}), "n": agg.get("n", 0)}


def _examples_to_set(client, dataset_name: str) -> list[dict]:
    out = []
    for ex in client.list_examples(dataset_name=dataset_name):
        inp, outp = ex.inputs or {}, ex.outputs or {}
        if inp.get("question"):
            out.append({"question": inp["question"],
                        "expected_url": outp.get("expected_url"),
                        "title": outp.get("title", "")})
    return out
