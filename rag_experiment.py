"""LangSmith dataset + experiment for the RAG evaluation.

The eval dataset is version-controlled as a reusable template
(``eval/rag_eval_dataset.json``) and referenced by **dataset ID** (rename-proof).
``sync_dataset`` pushes the template into LangSmith (idempotent); ``run_experiment``
resolves the dataset by ID and runs ``langsmith.evaluate`` with the RAG pipeline as
the target and three evaluators (retrieval_hit, reciprocal_rank, LLM-judged
answer_correctness), producing a comparable experiment in the LangSmith UI.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import config
import rag
from providers import build_chat_model, resolve_provider_model, resolve_judge

TEMPLATE_PATH = Path(__file__).parent / "eval" / "rag_eval_dataset.json"
# Canonical dataset reference is the ID (overridable via env); name is resolved at runtime.
DEFAULT_DATASET_ID = "5109d873-476c-477a-b465-a0c57ec959f8"


def load_template() -> dict:
    if TEMPLATE_PATH.exists():
        return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return {}


def dataset_id() -> str | None:
    return os.environ.get("LANGSMITH_RAG_DATASET_ID") or load_template().get("dataset_id") or DEFAULT_DATASET_ID


def _client():
    from langsmith import Client
    return Client()


def sync_dataset(client=None, *, eval_set: list[dict] | None = None) -> dict:
    """Ensure the dataset exists in LangSmith (by ID, else by name from the
    template, else freshly created). Idempotent. Returns {id, name, examples}."""
    client = client or _client()
    tmpl = load_template()
    examples = eval_set or [
        {"inputs": e["inputs"], "outputs": e["outputs"]} for e in tmpl.get("examples", [])]
    if not examples:
        raise rag.RagError("No eval examples available (template empty and none generated).")

    # 1) Reference by stored ID when it still resolves.
    dsid = dataset_id()
    if dsid:
        try:
            ds = client.read_dataset(dataset_id=dsid)
            return {"id": str(ds.id), "name": ds.name, "examples": ds.example_count or len(examples)}
        except Exception:  # noqa: BLE001
            pass  # ID not in this workspace -> create from template below

    # 2) Otherwise create (or reuse by name) and upload the template examples.
    name = tmpl.get("name") or "RAG eval"
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(dataset_name=name,
                              description=tmpl.get("description", "RAG eval dataset."))
        client.create_examples(dataset_name=name, examples=[
            {"inputs": e["inputs"], "outputs": e["outputs"]} for e in examples])
    ds = client.read_dataset(dataset_name=name)
    return {"id": str(ds.id), "name": ds.name, "examples": ds.example_count or len(examples)}


def export_dataset(dataset_id_value: str | None = None, client=None) -> dict:
    """Write the committed template from the current LangSmith dataset."""
    client = client or _client()
    dsid = dataset_id_value or dataset_id()
    ds = client.read_dataset(dataset_id=dsid)
    examples = [{"inputs": ex.inputs, "outputs": ex.outputs}
                for ex in client.list_examples(dataset_id=dsid)]
    tmpl = load_template()
    tmpl.update({
        "name": ds.name, "dataset_id": str(ds.id),
        "langsmith_project_id": os.environ.get("LANGSMITH_PROJECT_ID", tmpl.get("langsmith_project_id")),
        "schema": {"inputs": ["question"], "outputs": ["expected_url", "title"]},
        "examples": sorted(examples, key=lambda e: e["outputs"].get("expected_url", "")),
    })
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_PATH.write_text(json.dumps(tmpl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"id": str(ds.id), "name": ds.name, "examples": len(examples), "path": str(TEMPLATE_PATH)}


# --- evaluation target + evaluators -----------------------------------------
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
            key, score = getattr(r, "key", None), getattr(r, "score", None)
            if key is not None and score is not None:
                sums[key] = sums.get(key, 0.0) + float(score)
                counts[key] = counts.get(key, 0) + 1
    return {"n": n, "means": {k: round(sums[k] / counts[k], 3) for k in sums}}


def run_experiment(*, provider: str = "auto", model: str | None = None,
                   judge_provider: str | None = None, judge_model: str | None = None,
                   k: int = 6, rerank: bool = True, graph_rag: bool = True,
                   max_concurrency: int = 2) -> dict:
    try:
        from langsmith import evaluate
    except Exception as exc:  # noqa: BLE001
        raise rag.RagError(f"langsmith is required for experiments: {exc}") from exc

    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise rag.RagError("No LLM provider configured for the experiment.")
    # Judge from a different model family than the generator (avoid self-bias).
    jp, jm, cross = resolve_judge(rp, rm, judge_provider, judge_model)
    config.ensure_tracing_project()

    client = _client()
    ds = sync_dataset(client)  # ensure present; canonical reference is the ID
    name = client.read_dataset(dataset_id=ds["id"]).name  # evaluate() takes a name

    tag = "rerank" if rerank else "base"
    results = evaluate(
        _make_target(rp, rm, k, rerank, graph_rag),
        data=name,
        evaluators=[_retrieval_hit, _reciprocal_rank, _make_answer_judge(jp, jm)],
        experiment_prefix=f"rag-{tag}",
        metadata={"k": k, "rerank": rerank, "graph_rag": graph_rag,
                  "model": f"{rp}/{rm}", "judge": f"{jp}/{jm}",
                  "judge_cross_family": cross, "dataset_id": ds["id"]},
        client=client,
        max_concurrency=max_concurrency,
        blocking=True,
    )
    agg = _aggregate(results)
    dataset_url = None
    try:
        dataset_url = getattr(client.read_dataset(dataset_id=ds["id"]), "url", None)
    except Exception:  # noqa: BLE001
        pass
    return {"experiment_name": getattr(results, "experiment_name", None),
            "dataset_id": ds["id"], "dataset_url": dataset_url, "rerank": rerank,
            "k": k, "generator": f"{rp}/{rm}", "judge": f"{jp}/{jm}",
            "judge_cross_family": cross,
            "metrics": agg.get("means", {}), "n": agg.get("n", 0)}
