"""Retrieval-augmented Q&A and evaluation over the contextual vector store.

Retrieval is hierarchical (section summaries -> chunks, see vectorstore.py) with
an optional LLM re-ranker or document-aware MMR for cross-document diversity.
Answers are grounded with inline citations. Retrieval and answering are wrapped
with LangSmith ``@traceable`` so runs show up in the configured project.
"""

from __future__ import annotations

import json
import os
import re

import embeddings as emb
import memory
from providers import ProviderError, build_chat_model, resolve_provider_model, resolve_judge
from vectorstore import VectorStore

# Token budget that separates the read regimes for a scoped "focus" (see
# answer()): a focus that fits gets full-document ICL; a larger focus falls to
# coarse-retrieve-sections-then-ICL. A rough chars/4 estimate is fine for a gate.
ICL_BUDGET_TOKENS = int(os.environ.get("RAG_ICL_BUDGET_TOKENS", "100000"))


def _est_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)

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
             n_sections: int = 6, rerank_hits: bool = True,
             mmr: bool = False, mmr_lambda: float = 0.5, hybrid: bool = True,
             provider: str = "auto", model: str | None = None) -> list[dict]:
    vs = VectorStore.load(vs_name)
    if not vs.chunks:
        return []
    q = emb.embed_query(question, model=vs.embed_model)

    def _candidates(n: int) -> list[dict]:
        # Hybrid (dense + BM25, RRF-fused) by default; dense-only when hybrid off.
        if hybrid:
            return vs.search_hybrid(q, question, k=n, n_sections=n_sections)
        return vs.search(q, k=n, n_sections=n_sections)

    if mmr:
        # Document-aware MMR selects the final k directly (diversity is the goal;
        # the dense path owns MMR, so hybrid is not applied here).
        hits = vs.search(q, k=k, n_sections=n_sections, mmr=True, mmr_lambda=mmr_lambda)
    elif rerank_hits:
        # Over-fetch hybrid candidates, then LLM listwise re-rank down to k.
        hits = _candidates(max(k * 4, 20))
        if hits:
            hits = rerank(question, hits, top_k=k, provider=provider, model=model)
    else:
        hits = _candidates(k)
    return hits


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] {h.get('title')} ({h.get('date') or 'n/a'}) — {h.get('url')}\n"
            f"Context: {h.get('contextual_summary')}\n"
            f"Passage: {h.get('text')}")
    return "\n\n".join(blocks)


_RAG_PROMPT = """Answer the question using the context passages from the user's
knowledge library. Ground every claim in the passages and cite them inline as [n].

The question may require SYNTHESIS — connecting, comparing, or combining facts
spread across several passages that no single passage states together. That is
expected: reason over the passages and draw the connections they jointly support,
as long as each step stays grounded in their content. Do not invent facts or rely
on outside knowledge. Only say the answer is not in the context when the passages
genuinely lack the information needed — not merely because no single passage
states the connection explicitly.
{memory}
Question: {question}

Context passages:
{context}

Respond in markdown:
## Answer
<concise, well-structured answer with inline [n] citations>

## Key sources
- [n] <title> — <url>
"""


def _answer_gist(text: str, n: int = 400) -> str:
    """Leading prose of an answer (headings, blank lines and the trailing
    sources/citations section stripped) — used as the stored memory text."""
    body = re.split(r"\n#{1,6}\s*(?:Key sources|Citations|Sources)\b",
                    text or "", maxsplit=1, flags=re.I)[0]
    lines = [ln.strip() for ln in body.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    return " ".join(lines)[:n].strip()


def _answer_from_hits(question: str, hits: list[dict], rp: str, rm: str,
                      memory_section: str = "") -> str:
    try:
        chat = build_chat_model(rp, rm, max_tokens=1200)
    except ProviderError as exc:
        raise RagError(str(exc)) from exc
    return _gen(chat, _RAG_PROMPT.format(question=question.strip(),
                                         context=_format_context(hits),
                                         memory=memory_section))


# --- In-context (ICL) answering over a scoped "focus" ------------------------
_ICL_SYSTEM = (
    "You answer questions using ONLY the documents provided below. Ground every "
    "claim in them and cite inline as [n] by the document/section number. "
    "Synthesis across the documents is expected — connect and compare facts they "
    "jointly support — but do not use outside knowledge or invent facts. If the "
    "documents genuinely lack the answer, say so."
)
_ICL_FORMAT = ("Respond in markdown:\n## Answer\n<concise answer with inline [n] "
               "citations>\n\n## Key sources\n- [n] <title> — <url>")


def _load_focus(focus_ids: list[str] | None) -> list[dict]:
    """Load cached-store records (with raw_text) for a focus selection."""
    if not focus_ids:
        return []
    import cached_store
    store = cached_store.get_store()
    out = []
    for fid in focus_ids:
        rec = store.get(fid, with_text=True)
        if rec and (rec.get("raw_text") or "").strip():
            out.append(rec)
    return out


def _icl_generate(question: str, blocks: list[dict], rp: str, rm: str,
                  memory_section: str = "") -> str:
    """Answer from full document/section text held in context. For Anthropic, the
    document prefix is sent with prompt caching so repeat questions on the same
    focus are ~0.1x cost + lower latency; other providers stuff it uncached."""
    docs_text = "\n\n".join(
        f"[{i + 1}] {b['title']}" + (f" — {b['section']}" if b.get("section") else "")
        + f" ({b['url']})\n{b['text']}"
        for i, b in enumerate(blocks))
    user = ((memory_section + "\n") if memory_section else "") + \
        f"Question: {question.strip()}\n\n{_ICL_FORMAT}"

    if rp == "anthropic":
        try:
            import anthropic
            import config
            client = config.traced_anthropic(anthropic.Anthropic())
            msg = client.messages.create(
                model=rm, max_tokens=1500,
                system=[
                    {"type": "text", "text": _ICL_SYSTEM},
                    # Stable, large prefix → cache it; the volatile question is in
                    # the user turn after it, so the prefix stays byte-identical.
                    {"type": "text", "text": docs_text,
                     "cache_control": {"type": "ephemeral"}},
                ],
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        except Exception:  # noqa: BLE001  (fall back to the generic provider path)
            pass
    # Generic provider path (no prompt caching): one stuffed prompt.
    try:
        chat = build_chat_model(rp, rm, max_tokens=1500)
    except ProviderError as exc:
        raise RagError(str(exc)) from exc
    return _gen(chat, f"{_ICL_SYSTEM}\n\n{docs_text}\n\n{user}")


def _icl_citations(blocks: list[dict]) -> list[dict]:
    return [{"n": i + 1, "title": b["title"], "url": b["url"],
             "section_title": b.get("section"), "preview": (b["text"][:240])}
            for i, b in enumerate(blocks)]


def _answer_icl_docs(question: str, focus: list[dict], rp: str, rm: str,
                     memory_section: str) -> tuple[str, list[dict], dict]:
    """Regime 1: the focus fits the budget — ICL the FULL text of each doc."""
    blocks = [{"title": it.get("source_title") or it.get("source_url") or it["id"],
               "url": it.get("source_url") or f"cache:{it['id']}",
               "text": it.get("raw_text") or ""} for it in focus]
    text = _icl_generate(question, blocks, rp, rm, memory_section)
    scope = {"mode": "icl-documents", "documents": len(blocks),
             "tokens": sum(_est_tokens(b["text"]) for b in blocks)}
    return text, _icl_citations(blocks), scope


def _answer_icl_sections(question: str, focus: list[dict], rp: str, rm: str,
                         memory_section: str, vs_name: str) -> tuple[str, list[dict], dict]:
    """Regime 2: the focus is too big to stuff whole — coarse-retrieve the most
    relevant SECTIONS within the focus docs, reconstruct each section's full
    text, fill the budget best-first, and ICL those sections."""
    vs = VectorStore.load(vs_name)
    focus_urls = {(it.get("source_url") or f"cache:{it['id']}") for it in focus}
    ranked = vs.rank_sections(emb.embed_query(question, model=vs.embed_model),
                              restrict_urls=focus_urls) if vs.sections else []
    blocks, used = [], 0
    for s in ranked:
        txt = vs.section_full_text(s["id"])
        if not txt:
            continue
        t = _est_tokens(txt)
        if blocks and used + t > ICL_BUDGET_TOKENS:
            break
        blocks.append({"title": s.get("title") or s.get("url"),
                       "url": s.get("url"),
                       "section": s.get("section_title") or s.get("name"),
                       "text": txt})
        used += t
        if used >= ICL_BUDGET_TOKENS:
            break

    if not blocks:
        # Focus docs aren't vectorized (no sections) — degrade to full-doc ICL
        # truncated to the budget, in selection order.
        for it in focus:
            raw = it.get("raw_text") or ""
            room = ICL_BUDGET_TOKENS - used
            if room <= 0:
                break
            snippet = raw[: room * 4]
            blocks.append({"title": it.get("source_title") or it["id"],
                           "url": it.get("source_url") or f"cache:{it['id']}",
                           "text": snippet})
            used += _est_tokens(snippet)
        text = _icl_generate(question, blocks, rp, rm, memory_section)
        return text, _icl_citations(blocks), {
            "mode": "icl-truncated", "documents": len(blocks), "tokens": used,
            "note": "focus not vectorized; truncated to budget"}

    text = _icl_generate(question, blocks, rp, rm, memory_section)
    scope = {"mode": "icl-sections", "sections": len(blocks),
             "documents": len({b["url"] for b in blocks}),
             "available_sections": len(ranked), "tokens": used}
    return text, _icl_citations(blocks), scope


def answer_with_contexts(question: str, *, provider: str = "auto", model: str | None = None,
                         k: int = 6, vs_name: str = "library",
                         rerank_hits: bool = True, mmr: bool = False,
                         mmr_lambda: float = 0.5) -> dict:
    """Like answer(), but also returns the full retrieved context texts (for RAGAS)."""
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RagError("No LLM provider configured.")
    hits = retrieve(question, k=k, vs_name=vs_name,
                    rerank_hits=rerank_hits, mmr=mmr, mmr_lambda=mmr_lambda,
                    provider=rp, model=rm)
    text = (_answer_from_hits(question, hits, rp, rm) if hits
            else "The knowledge library is empty — ingest some pages first.")
    return {"answer": text,
            "contexts": [h.get("text") or "" for h in hits],
            "urls": [h.get("url") for h in hits],
            "hits": hits, "provider": rp, "model": rm}


@traceable(name="rag.answer", tags=["rag", "qa", "knowledge-library"])
def answer(question: str, *, provider: str = "auto", model: str | None = None,
           k: int = 6, vs_name: str = "library",
           rerank_hits: bool = True, mmr: bool = False, mmr_lambda: float = 0.5,
           hybrid: bool = True, focus_ids: list[str] | None = None,
           use_memory: bool = True, write_back: bool = True) -> dict:
    """Answer a question with the regime that fits the scope:

    - **no focus** → hybrid RAG over the whole library (retrieve top passages).
    - **focus that fits the budget** → ICL the full text of the focus documents.
    - **focus too big** → coarse-retrieve the most relevant sections within the
      focus, then ICL those whole sections.

    ``focus_ids`` are cached-store item ids (the user's selection)."""
    if not question or not question.strip():
        raise RagError("A question is required.")
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RagError("No LLM provider configured. Set an API key to use RAG Q&A.")

    # Memory recall: fold prior learnings into the prompt as background context.
    mems, memory_section = [], ""
    if use_memory:
        try:
            mems = memory.recall(question, k=4)
        except Exception:  # noqa: BLE001
            mems = []
        if mems:
            memory_section = ("\nPrior library memory (background only — not "
                              "citable; verify against the passages below):\n"
                              + memory.format_memories(mems) + "\n")

    # --- regime router ------------------------------------------------------
    focus = _load_focus(focus_ids)
    if focus:
        scoped = sum(_est_tokens(it.get("raw_text") or "") for it in focus)
        if scoped <= ICL_BUDGET_TOKENS:
            regime = "icl-documents"
            text, citations, scope = _answer_icl_docs(question, focus, rp, rm, memory_section)
        else:
            regime = "icl-sections"
            text, citations, scope = _answer_icl_sections(
                question, focus, rp, rm, memory_section, vs_name)
    else:
        regime = "rag"
        hits = retrieve(question, k=k, vs_name=vs_name, rerank_hits=rerank_hits,
                        mmr=mmr, mmr_lambda=mmr_lambda, hybrid=hybrid, provider=rp, model=rm)
        if not hits:
            return {"answer": "The knowledge library is empty — ingest or vectorize "
                    "some pages first.", "citations": [], "provider": rp, "model": rm,
                    "memories_used": mems, "regime": regime,
                    "scope": {"mode": "rag", "passages": 0}}
        text = _answer_from_hits(question, hits, rp, rm, memory_section=memory_section)
        citations = [{
            "n": i + 1, "chunk_id": h["id"], "title": h.get("title"), "url": h.get("url"),
            "date": h.get("date"), "section_title": h.get("section_title"),
            "score": h.get("score"), "preview": h.get("preview"),
        } for i, h in enumerate(hits)]
        scope = {"mode": "rag", "passages": len(hits),
                 "documents": len({h.get("url") for h in hits}),
                 "reranked": rerank_hits, "mmr": mmr, "hybrid": hybrid}

    # Write-back: reinforce used memories and file this answer for next time.
    if write_back:
        try:
            if mems:
                memory.bump_use([m["id"] for m in mems])
            gist = _answer_gist(text)
            if gist and "not in the context" not in text.lower():
                memory.remember(f"Q: {question.strip()}\nA: {gist}", kind="answer",
                                salience=2, origin="rag_ask", confidence="EXTRACTED",
                                source_url=(citations[0]["url"] if citations else ""),
                                tags=["rag"])
        except Exception:  # noqa: BLE001  (write-back must never break an answer)
            pass

    return {"answer": text, "citations": citations, "provider": rp, "model": rm,
            "k": k, "reranked": rerank_hits, "mmr": mmr, "memories_used": mems,
            "regime": regime, "scope": scope}


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
             k: int = 6, vs_name: str = "library",
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
        hits = retrieve(q, k=k, vs_name=vs_name,
                        rerank_hits=rerank_hits, mmr=mmr, mmr_lambda=mmr_lambda, provider=rp, model=rm)
        urls = [h.get("url") for h in hits]
        hit = exp in urls
        rank = (urls.index(exp) + 1) if hit else 0
        hit_at_k += 1 if hit else 0
        rr_sum += (1.0 / rank) if rank else 0.0

        ans = answer(q, provider=rp, model=rm, k=k, vs_name=vs_name,
                     rerank_hits=rerank_hits, mmr=mmr, mmr_lambda=mmr_lambda,
                     use_memory=False, write_back=False)
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
        "n": len(eval_set), "k": k, "reranked": rerank_hits, "mmr": mmr,
        "retrieval_hit_rate": round(hit_at_k / n, 3),
        "mrr": round(rr_sum / n, 3),
        "mean_answer_score": round(judge_sum / judged, 3) if judged else None,
        "generator": f"{rp}/{rm}", "judge": f"{jp}/{jm}", "judge_cross_family": cross,
        "rows": rows,
    }
