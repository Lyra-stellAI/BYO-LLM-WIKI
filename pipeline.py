"""Contextual-retrieval ingestion pipeline.

For each URL: fetch and clean the page, split it into sections, and make ONE LLM
call that reads the whole document and returns a doc overview plus a short
contextual summary per section (so each summary is situated in the full doc, in
the spirit of Anthropic's Contextual Retrieval). Sections are split into chunks;
each chunk is embedded with its title/date/section-context prepended. Records go
into the two-layer vector store and the doc -> section -> chunk hierarchy is
mirrored into the knowledge graph.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

import embeddings as emb
import ingestion
import knowledge_graph as kg
from providers import ProviderError, build_chat_model, resolve_provider_model
from vectorstore import VectorStore

_UA = "Mozilla/5.0 (X11; Linux x86_64) knowledge-library/1.0"
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b")
_MAX_DOC_CHARS_FOR_SUMMARY = 60000


class PipelineError(RuntimeError):
    pass


def _slug(s: str) -> str:
    s = re.sub(r"^https?://", "", (s or "").lower())
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80] or "doc"


def _fetch(url: str) -> dict:
    resp = requests.get(url, headers={"User-Agent": _UA,
                        "Accept": "text/html,application/xhtml+xml"}, timeout=25)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    date = ""
    for attrs in ({"property": "article:published_time"}, {"property": "og:updated_time"},
                  {"name": "datePublished"}, {"itemprop": "datePublished"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            date = tag["content"][:10]
            break
    t = soup.find("time")
    if not date and t and (t.get("datetime") or t.get_text(strip=True)):
        date = (t.get("datetime") or t.get_text(strip=True))[:10]

    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header",
                     "footer", "nav", "aside", "form"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    if not date:
        m = _DATE_RE.search(text[:1200])  # publish date usually near the byline
        if m:
            date = m.group(0)
    return {"title": title, "url": url, "text": text, "date": date}


def _gen(model, prompt: str) -> str:
    msg = model.invoke(prompt)
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _parse_obj(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", (text or "").strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        out = json.loads(m.group())
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _contextualize(model, title: str, date: str, sections: list[str]) -> tuple[str, list[str]]:
    """One LLM call: doc overview + a contextual summary for each section."""
    blocks = []
    total = 0
    for i, s in enumerate(sections):
        snippet = s[:1800]
        total += len(snippet)
        blocks.append(f"[SECTION {i}]\n{snippet}")
        if total > _MAX_DOC_CHARS_FOR_SUMMARY:
            break
    joined = "\n\n".join(blocks)
    prompt = (
        "You are indexing a web document for retrieval. Read the whole document, "
        "then write a concise 1-2 sentence CONTEXTUAL SUMMARY for each numbered "
        "section that situates it within the overall document (what it covers and "
        "how it fits the whole). These summaries are used as retrieval keys.\n\n"
        f"Document title: {title}\nDate: {date or 'n/a'}\n\n"
        f"{joined}\n\n"
        "Return ONLY JSON of this exact shape:\n"
        '{"overview": "<2-3 sentence overview of the whole document>", '
        '"sections": [{"index": 0, "summary": "<contextual summary>"}, ...]}\n'
        "Include one entry per section index shown above."
    )
    data = _parse_obj(_gen(model, prompt))
    overview = (data.get("overview") or "").strip()
    summaries = [""] * len(sections)
    for item in data.get("sections", []) or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(sections):
            summaries[idx] = (item.get("summary") or "").strip()
    # Fallback for any missing summary: use the overview or a text snippet.
    for i in range(len(sections)):
        if not summaries[i]:
            summaries[i] = overview or sections[i][:160]
    return overview, summaries


def ingest_url(url: str, model, vs: VectorStore, *, section_size: int = 2500,
               chunk_size: int = 800, overlap: int = 120) -> dict:
    page = _fetch(url)
    title, date, text = page["title"], page["date"], page["text"]
    if len(text) < 200:
        raise PipelineError(f"Too little text extracted from {url}")

    doc_id = _slug(url)
    section_texts = ingestion.chunk_text(text, chunk_size=section_size, overlap=0)
    overview, summaries = _contextualize(model, title, date, section_texts)

    sec_records, chunk_records = [], []
    chunk_embed_inputs, sec_embed_inputs = [], []
    for si, (sec_text, summary) in enumerate(zip(section_texts, summaries)):
        sec_id = f"section_{doc_id}_{si}"
        sec_title = f"{title} — part {si + 1}"
        sec_records.append({"id": sec_id, "doc_id": doc_id, "url": url, "title": title,
                            "date": date, "section_index": si, "section_title": sec_title,
                            "summary": summary, "overview": overview})
        sec_embed_inputs.append(f"{title} ({date or 'n/a'}) — {sec_title}: {summary}")
        for ci, ch_text in enumerate(ingestion.chunk_text(sec_text, chunk_size=chunk_size, overlap=overlap)):
            ch_id = f"chunk_{doc_id}_{si}_{ci}"
            chunk_records.append({
                "id": ch_id, "doc_id": doc_id, "url": url, "title": title, "date": date,
                "section_id": sec_id, "section_title": sec_title,
                "contextual_summary": summary, "text": ch_text,
                "preview": (ch_text[:240] + "…") if len(ch_text) > 240 else ch_text,
                "position": ci,
            })
            # Contextual retrieval: prepend title/date/section context before embedding.
            chunk_embed_inputs.append(
                f"{title} ({date or 'n/a'})\nSection: {sec_title}\n"
                f"Context: {summary}\n\n{ch_text}")

    chunk_emb = emb.embed_texts(chunk_embed_inputs, model=vs.embed_model)
    sec_emb = emb.embed_texts(sec_embed_inputs, model=vs.embed_model)

    vs.remove_doc(url)
    vs.add(chunk_records, chunk_emb, sec_records, sec_emb)

    # Mirror the hierarchy into the knowledge graph (doc -> section -> chunk).
    kg.add_rag_document(url, title, date,
                        sections=[{"id": s["id"], "title": s["section_title"], "summary": s["summary"]}
                                  for s in sec_records],
                        chunks=[{"id": c["id"], "section_id": c["section_id"],
                                 "text": c["text"], "preview": c["preview"]} for c in chunk_records])
    return {"url": url, "title": title, "date": date or "n/a",
            "sections": len(sec_records), "chunks": len(chunk_records)}


def ingest_urls(urls: list[str], *, provider: str = "auto", model: str | None = None,
                vs_name: str = "library", section_size: int = 2500,
                chunk_size: int = 800, overlap: int = 120, on_progress=None) -> dict:
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise PipelineError("No LLM provider configured for contextual summaries.")
    if not emb.embeddings_available():
        raise PipelineError("OPENAI_API_KEY is required for embeddings.")
    try:
        chat = build_chat_model(rp, rm, max_tokens=2000)
    except ProviderError as exc:
        raise PipelineError(str(exc)) from exc

    vs = VectorStore.load(vs_name)
    results, errors = [], []
    for url in urls:
        try:
            r = ingest_url(url, chat, vs, section_size=section_size,
                           chunk_size=chunk_size, overlap=overlap)
            results.append(r)
            if on_progress:
                on_progress(r)
        except Exception as exc:  # noqa: BLE001
            errors.append({"url": url, "error": str(exc)})
            if on_progress:
                on_progress({"url": url, "error": str(exc)})
    vs.persist()
    return {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "provider": rp, "model": rm,
        "results": results, "errors": errors,
        "vector_stats": vs.stats(), "graph_stats": kg.stats()["overall"],
    }
