"""RAGAS evaluation wired into a LangSmith experiment.

RAGAS metrics (faithfulness, answer relevancy, context precision) are exposed as
LangSmith evaluators and run via ``langsmith.evaluate`` against the eval dataset,
so each row gets RAGAS scores you can inspect/compare in the LangSmith UI. The
RAG pipeline (retrieval + answer, with full retrieved contexts) is the target.

RAGAS uses our Claude model (LLM judge) + OpenAI embeddings. ragas is an optional
dependency; install with `pip install ragas`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types

# ragas 0.4.x imports a Vertex AI path removed from current langchain-community.
# We never use Vertex AI, so shim the unused symbols before importing ragas
# (keeps the installed langchain 1.x / deepagents stack intact).
for _modname, _attr in (("langchain_community.chat_models.vertexai", "ChatVertexAI"),
                        ("langchain_community.llms.vertexai", "VertexAI")):
    if _modname not in sys.modules:
        _m = types.ModuleType(_modname)
        setattr(_m, _attr, type(_attr, (), {}))
        sys.modules[_modname] = _m

import config
import rag
import rag_experiment
from providers import resolve_provider_model, build_chat_model

_RAGAS_METRICS = ("ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision")
_wrappers_cache: dict = {}


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _wrappers(provider: str, model: str):
    key = (provider, model)
    if key not in _wrappers_cache:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings
        llm = LangchainLLMWrapper(build_chat_model(provider, model, max_tokens=1024))
        emb = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=os.environ.get("KG_EMBED_MODEL", "text-embedding-3-small")))
        _wrappers_cache[key] = (llm, emb)
    return _wrappers_cache[key]


def _score(metric, sample) -> float:
    loop = asyncio.new_event_loop()
    try:
        return float(loop.run_until_complete(metric.single_turn_ascore(sample)))
    finally:
        loop.close()


def _sample(run, example):
    from ragas import SingleTurnSample
    out, inp = run.outputs or {}, example.inputs or {}
    return SingleTurnSample(
        user_input=inp.get("question", ""),
        response=out.get("answer", ""),
        retrieved_contexts=[c for c in (out.get("retrieved_contexts") or []) if c],
    )


def make_ragas_evaluators(provider: str, model: str) -> list:
    import warnings
    warnings.filterwarnings("ignore")
    from ragas.metrics import (Faithfulness, ResponseRelevancy,
                               LLMContextPrecisionWithoutReference)
    llm, emb = _wrappers(provider, model)
    faith = Faithfulness(llm=llm)
    rel = ResponseRelevancy(llm=llm, embeddings=emb)
    cprec = LLMContextPrecisionWithoutReference(llm=llm)

    def _safe(metric, key):
        def _evaluator(run, example):
            try:
                return {"key": key, "score": _score(metric, _sample(run, example))}
            except Exception as exc:  # noqa: BLE001
                return {"key": key, "score": None, "comment": str(exc)[:120]}
        _evaluator.__name__ = key
        return _evaluator

    return [_safe(faith, "ragas_faithfulness"),
            _safe(rel, "ragas_answer_relevancy"),
            _safe(cprec, "ragas_context_precision")]


def _make_target(provider, model, k, rerank, graph_rag):
    def target(inputs: dict) -> dict:
        res = rag.answer_with_contexts(inputs["question"], provider=provider, model=model,
                                       k=k, graph_rag=graph_rag, rerank_hits=rerank)
        return {"answer": res["answer"], "retrieved_contexts": res["contexts"],
                "retrieved_urls": res["urls"]}
    return target


def run_ragas_experiment(*, provider: str = "auto", model: str | None = None, k: int = 6,
                         rerank: bool = True, graph_rag: bool = True,
                         max_concurrency: int = 1) -> dict:
    if not ragas_available():
        raise rag.RagError("ragas is not installed. Run `pip install ragas`.")
    try:
        from langsmith import evaluate
    except Exception as exc:  # noqa: BLE001
        raise rag.RagError(f"langsmith is required: {exc}") from exc

    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise rag.RagError("No LLM provider configured for the RAGAS experiment.")
    config.ensure_tracing_project()

    client = rag_experiment._client()
    ds = rag_experiment.sync_dataset(client)          # ensure dataset is uploaded
    name = client.read_dataset(dataset_id=ds["id"]).name

    evaluators = make_ragas_evaluators(rp, rm) + [rag_experiment._retrieval_hit]
    tag = "rerank" if rerank else "base"
    results = evaluate(
        _make_target(rp, rm, k, rerank, graph_rag),
        data=name,
        evaluators=evaluators,
        experiment_prefix=f"ragas-{tag}",
        metadata={"eval": "ragas", "k": k, "rerank": rerank, "graph_rag": graph_rag,
                  "model": f"{rp}/{rm}", "dataset_id": ds["id"]},
        client=client,
        max_concurrency=max_concurrency,
        blocking=True,
    )
    agg = rag_experiment._aggregate(results)
    dataset_url = None
    try:
        dataset_url = getattr(client.read_dataset(dataset_id=ds["id"]), "url", None)
    except Exception:  # noqa: BLE001
        pass
    return {"experiment_name": getattr(results, "experiment_name", None),
            "dataset_id": ds["id"], "dataset_url": dataset_url,
            "metrics": agg.get("means", {}), "n": agg.get("n", 0),
            "rerank": rerank, "k": k, "provider": rp, "model": rm}
