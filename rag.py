"""Retrieval-augmented Q&A and evaluation over the contextual vector store.

Retrieval is hierarchical (section summaries -> chunks, see vectorstore.py) and
optionally enriched with knowledge-graph context (graph RAG). Answers are grounded
with inline citations. Retrieval and answering are wrapped with LangSmith
``@traceable`` so runs show up in the configured project.
"""

from __future__ import annotations

import json
import re

import embeddings as emb
import knowledge_graph as kg
from providers import ProviderError, build_chat_model, resolve_provider_model, resolve_judge
from vectorstore import VectorStore

try:  # tracing is optional
    from langsmith import traceable
except Exception:  # noqa: BLE001
    def traceable(*dargs, **dkw):  # type: ignore
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]
        def deco(fn):
            return fn
        return deco


class RagError(RuntimeError):
    pass


def _gen(model, prompt: str) -> str:
    msg = model.invoke(prompt)
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _graph_context_for(chunk_id: str) -> dict:
    """Pull 1-hop knowledge-graph context (entities/topics) for a chunk."""
    sub = kg.neighbors(chunk_id, where="overall", depth=1)
    ents, topics = [], []
    for n in sub.get("nodes", []):
        if n.get("type") == "entity" and n.get("name"):
            ents.append(n["name"])
        elif n.get("type") == "topic" and n.get("name"):
            topics.append(n["name"])
    return {"entities": sorted(set(ents))[:8], "topics": sorted(set(topics))[:4]}


_RERANK_PROMPT = """You are a precise search re-ranker. Given a query and numbered
passages, order the passages from MOST to LEAST relevant for answering the query.

Query: {query}

Passages:
{passages}

Return ONLY JSON: {{"ranking": [<passage numbers, best first>]}}.
Include only passages that are actually relevant; omit clearly irrelevant ones."""


@traceable(name="rag.rerank", tags=["rag", "rerank", "knowledge-library"])
def rerank(question: str, hits: list[dict], *, top_k: int,
           provider: str = "auto", model: str | None = None) -> list[dict]:
    """LLM listwise re-rank of candidate passages; returns the top_k reordered."""
    if len(hits) <= 1:
        return hits[:top_k]
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        return hits[:top_k]
    try:
        chat = build_chat_model(rp, rm, max_tokens=400)
    except ProviderError:
        return hits[:top_k]
    passages = "\n".join(
        f"[{i}] {h.get('title')} — {h.get('contextual_summary')}\n{(h.get('text') or '')[:280]}"
        for i, h in enumerate(hits))
    raw = _gen(chat, _RERANK_PROMPT.format(query=question, passages=passages))
    order: list[int] = []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            order = [int(x) for x in json.loads(m.group()).get("ranking", [])
                     if isinstance(x, (int, float))]
        except Exception:  # noqa: BLE001
            order = []
    out, seen = [], set()
    for pos, idx in enumerate(order):
        if 0 <= idx < len(hits) and idx not in seen:
            h = dict(hits[idx]); h["rerank_score"] = round(1.0 / (pos + 1), 4)
            out.append(h); seen.add(idx)
    for i, h in enumerate(hits):  # keep any unranked, after, in vector order
        if i not in seen:
            h = dict(h); h["rerank_score"] = 0.0
            out.append(h); seen.add(i)
    return out[:top_k]


@traceable(name="rag.retrieve", tags=["rag", "retrieval", "knowledge-library"])
def retrieve(question: str, *, k: int = 6, vs_name: str = "library",
             n_sections: int = 6, graph_rag: bool = True, rerank_hits: bool = True,
             mmr: bool = False, mmr_lambda: float = 0.5,
             provider: str = "auto", model: str | None = None) -> list[dict]:
    vs = VectorStore.load(vs_name)
    if not vs.chunks:
        return []
    q = emb.embed_query(question, model=vs.embed_model)
    if mmr:
        # Document-aware MMR selects the final k directly (diversity is the goal).
        hits = vs.search(q, k=k, n_sections=n_sections, mmr=True, mmr_lambda=mmr_lambda)
    elif rerank_hits:
        # Over-fetch candidates, then LLM listwise re-rank down to k.
        hits = vs.search(q, k=max(k * 4, 20), n_sections=n_sections)
        if hits:
            hits = rerank(question, hits, top_k=k, provider=provider, model=model)
    else:
        hits = vs.search(q, k=k, n_sections=n_sections)
    if graph_rag:
        for h in hits:
            h["graph"] = _graph_context_for(h["id"])
    return hits


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        g = h.get("graph") or {}
        extra = ""
        if g.get("entities"):
            extra = f"\nRelated entities: {', '.join(g['entities'])}"
        blocks.append(
            f"[{i}] {h.get('title')} ({h.get('date') or 'n/a'}) — {h.get('url')}\n"
            f"Context: {h.get('contextual_summary')}{extra}\n"
            f"Passage: {h.get('text')}")
    return "\n\n".join(blocks)


_RAG_PROMPT = """Answer the question using ONLY the context passages from the user's
knowledge library. Cite the passages you use inline as [n]. If the answer is not
in the context, say so plainly — do not invent facts.

Question: {question}

Context passages:
{context}

Respond in markdown:
## Answer
<concise, well-structured answer with inline [n] citations>

## Key sources
- [n] <title> — <url>
"""


def _answer_from_hits(question: str, hits: list[dict], rp: str, rm: str) -> str:
    try:
        chat = build_chat_model(rp, rm, max_tokens=1200)
    except ProviderError as exc:
        raise RagError(str(exc)) from exc
    return _gen(chat, _RAG_PROMPT.format(question=question.strip(), context=_format_context(hits)))


def answer_with_contexts(question: str, *, provider: str = "auto", model: str | None = None,
                         k: int = 6, vs_name: str = "library", graph_rag: bool = True,
                         rerank_hits: bool = True, mmr: bool = False,
                         mmr_lambda: float = 0.5) -> dict:
    """Like answer(), but also returns the full retrieved context texts (for RAGAS)."""
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RagError("No LLM provider configured.")
    hits = retrieve(question, k=k, vs_name=vs_name, graph_rag=graph_rag,
                    rerank_hits=rerank_hits, mmr=mmr, mmr_lambda=mmr_lambda, provider=rp, model=rm)
    text = (_answer_from_hits(question, hits, rp, rm) if hits
            else "The knowledge library is empty — ingest some pages first.")
    return {"answer": text,
            "contexts": [h.get("text") or "" for h in hits],
            "urls": [h.get("url") for h in hits],
            "hits": hits, "provider": rp, "model": rm}


@traceable(name="rag.answer", tags=["rag", "qa", "knowledge-library"])
def answer(question: str, *, provider: str = "auto", model: str | None = None,
           k: int = 6, vs_name: str = "library", graph_rag: bool = True,
           rerank_hits: bool = True, mmr: bool = False, mmr_lambda: float = 0.5) -> dict:
    if not question or not question.strip():
        raise RagError("A question is required.")
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RagError("No LLM provider configured. Set an API key to use RAG Q&A.")
    hits = retrieve(question, k=k, vs_name=vs_name, graph_rag=graph_rag,
                    rerank_hits=rerank_hits, mmr=mmr, mmr_lambda=mmr_lambda, provider=rp, model=rm)
    if not hits:
        return {"answer": "The knowledge library is empty — ingest some pages first.",
                "citations": [], "provider": rp, "model": rm}
    text = _answer_from_hits(question, hits, rp, rm)
    citations = [{
        "n": i + 1, "chunk_id": h["id"], "title": h.get("title"), "url": h.get("url"),
        "date": h.get("date"), "section_title": h.get("section_title"),
        "score": h.get("score"), "preview": h.get("preview"),
    } for i, h in enumerate(hits)]
    return {"answer": text, "citations": citations, "provider": rp, "model": rm,
            "k": k, "graph_rag": graph_rag, "reranked": rerank_hits}


# --- Evaluation --------------------------------------------------------------
def _docs_overview(vs: VectorStore) -> dict:
    """Map url -> {title, date, overview, sample_text} from the vector store."""
    docs: dict[str, dict] = {}
    for s in vs.sections:
        u = s.get("url")
        if u and u not in docs:
            docs[u] = {"title": s.get("title"), "date": s.get("date"),
                       "overview": s.get("overview", ""), "sample": s.get("summary", "")}
    return docs


@traceable(name="rag.build_eval_set", tags=["rag", "eval", "knowledge-library"])
def build_eval_set(*, vs_name: str = "library", provider: str = "auto",
                   model: str | None = None, max_questions: int = 10) -> list[dict]:
    """Generate (question, expected_url) pairs grounded in individual documents."""
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RagError("No LLM provider configured for eval-set generation.")
    chat = build_chat_model(rp, rm, max_tokens=300)
    vs = VectorStore.load(vs_name)
    docs = _docs_overview(vs)
    urls = list(docs.keys())[:max_questions]
    out = []
    for u in urls:
        d = docs[u]
        prompt = (
            "Write ONE specific factual question that is clearly answerable from "
            "this document and unlikely to be answered by a generic source. "
            "Return only the question.\n\n"
            f"Title: {d['title']}\nOverview: {d['overview']}\nDetail: {d['sample']}")
        q = _gen(chat, prompt).strip().strip('"').splitlines()[0]
        if q:
            out.append({"question": q, "expected_url": u, "title": d["title"]})
    return out


_JUDGE_PROMPT = """You are grading a RAG answer for an evaluation.

Question: {question}
Expected source document: {title} ({url})

Answer under test:
{answer}

Score the answer from 0.0 to 1.0 on whether it is correct, grounded, and
actually answers the question (1.0 = fully correct and grounded; 0.0 = wrong or
unsupported). Return ONLY JSON: {{"score": <float>, "reason": "<one sentence>"}}"""


@traceable(name="rag.evaluate", tags=["rag", "eval", "knowledge-library"])
def evaluate(eval_set: list[dict], *, provider: str = "auto", model: str | None = None,
             judge_provider: str | None = None, judge_model: str | None = None,
             k: int = 6, vs_name: str = "library", graph_rag: bool = True,
             rerank_hits: bool = True, mmr: bool = False, mmr_lambda: float = 0.5) -> dict:
    """Run retrieval + answer for each eval item; score retrieval and answer quality.

    The answer judge defaults to a DIFFERENT model family than the generator
    (avoids self-preference bias); override with judge_provider/judge_model."""
    rp, rm = resolve_provider_model(provider, model)
    jp, jm, cross = resolve_judge(rp, rm, judge_provider, judge_model) if rp else (None, "", False)
    judge = build_chat_model(jp, jm, max_tokens=200) if jp else None
    rows = []
    hit_at_k = 0
    rr_sum = 0.0
    judge_sum = 0.0
    judged = 0
    for item in eval_set:
        q, exp = item["question"], item["expected_url"]
        hits = retrieve(q, k=k, vs_name=vs_name, graph_rag=graph_rag,
                        rerank_hits=rerank_hits, mmr=mmr, mmr_lambda=mmr_lambda, provider=rp, model=rm)
        urls = [h.get("url") for h in hits]
        hit = exp in urls
        rank = (urls.index(exp) + 1) if hit else 0
        hit_at_k += 1 if hit else 0
        rr_sum += (1.0 / rank) if rank else 0.0

        ans = answer(q, provider=rp, model=rm, k=k, vs_name=vs_name, graph_rag=graph_rag,
                     rerank_hits=rerank_hits, mmr=mmr, mmr_lambda=mmr_lambda)
        score, reason = None, ""
        if judge is not None:
            raw = _gen(judge, _JUDGE_PROMPT.format(
                question=q, title=item.get("title", ""), url=exp, answer=ans["answer"]))
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    j = json.loads(m.group())
                    score = float(j.get("score"))
                    reason = j.get("reason", "")
                except Exception:  # noqa: BLE001
                    pass
            if score is not None:
                judge_sum += score
                judged += 1
        rows.append({"question": q, "expected_url": exp, "retrieval_hit": hit,
                     "rank": rank, "answer_score": score, "reason": reason,
                     "top_url": urls[0] if urls else None})
    n = max(len(eval_set), 1)
    return {
        "n": len(eval_set), "k": k, "graph_rag": graph_rag, "reranked": rerank_hits,
        "mmr": mmr,
        "retrieval_hit_rate": round(hit_at_k / n, 3),
        "mrr": round(rr_sum / n, 3),
        "mean_answer_score": round(judge_sum / judged, 3) if judged else None,
        "generator": f"{rp}/{rm}", "judge": f"{jp}/{jm}", "judge_cross_family": cross,
        "rows": rows,
    }
