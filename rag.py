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
from providers import ProviderError, build_chat_model, resolve_provider_model
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


@traceable(name="rag.retrieve", tags=["rag", "retrieval", "knowledge-library"])
def retrieve(question: str, *, k: int = 6, vs_name: str = "library",
             n_sections: int = 6, graph_rag: bool = True) -> list[dict]:
    vs = VectorStore.load(vs_name)
    if not vs.chunks:
        return []
    q = emb.embed_query(question, model=vs.embed_model)
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


@traceable(name="rag.answer", tags=["rag", "qa", "knowledge-library"])
def answer(question: str, *, provider: str = "auto", model: str | None = None,
           k: int = 6, vs_name: str = "library", graph_rag: bool = True) -> dict:
    if not question or not question.strip():
        raise RagError("A question is required.")
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RagError("No LLM provider configured. Set an API key to use RAG Q&A.")
    hits = retrieve(question, k=k, vs_name=vs_name, graph_rag=graph_rag)
    if not hits:
        return {"answer": "The knowledge library is empty — ingest some pages first.",
                "citations": [], "provider": rp, "model": rm}
    try:
        chat = build_chat_model(rp, rm, max_tokens=1200)
    except ProviderError as exc:
        raise RagError(str(exc)) from exc
    text = _gen(chat, _RAG_PROMPT.format(question=question.strip(), context=_format_context(hits)))
    citations = [{
        "n": i + 1, "chunk_id": h["id"], "title": h.get("title"), "url": h.get("url"),
        "date": h.get("date"), "section_title": h.get("section_title"),
        "score": h.get("score"), "preview": h.get("preview"),
    } for i, h in enumerate(hits)]
    return {"answer": text, "citations": citations, "provider": rp, "model": rm,
            "k": k, "graph_rag": graph_rag}


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
             k: int = 6, vs_name: str = "library", graph_rag: bool = True) -> dict:
    """Run retrieval + answer for each eval item; score retrieval and answer quality."""
    rp, rm = resolve_provider_model(provider, model)
    judge = build_chat_model(rp, rm, max_tokens=200) if rp else None
    rows = []
    hit_at_k = 0
    rr_sum = 0.0
    judge_sum = 0.0
    judged = 0
    for item in eval_set:
        q, exp = item["question"], item["expected_url"]
        hits = retrieve(q, k=k, vs_name=vs_name, graph_rag=graph_rag)
        urls = [h.get("url") for h in hits]
        hit = exp in urls
        rank = (urls.index(exp) + 1) if hit else 0
        hit_at_k += 1 if hit else 0
        rr_sum += (1.0 / rank) if rank else 0.0

        ans = answer(q, provider=rp, model=rm, k=k, vs_name=vs_name, graph_rag=graph_rag)
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
        "n": len(eval_set), "k": k, "graph_rag": graph_rag,
        "retrieval_hit_rate": round(hit_at_k / n, 3),
        "mrr": round(rr_sum / n, 3),
        "mean_answer_score": round(judge_sum / judged, 3) if judged else None,
        "provider": rp, "model": rm,
        "rows": rows,
    }
