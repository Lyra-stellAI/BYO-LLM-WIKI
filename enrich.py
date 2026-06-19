"""Populate the entity + topic layers over an already-ingested library.

The RAG ingest builds source -> section -> chunk. This adds the semantic layers
on top: extract entities + typed relations per section (concurrently), wire them
into the graph as canonical entities with chunk->entity mentions, then cluster the
top entities into named topics. This is what makes graph-RAG and the document map
"live".
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re

import extraction
import knowledge_graph as kg
from providers import build_chat_model, resolve_provider_model


def _gen(model, prompt: str) -> str:
    msg = model.invoke(prompt)
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def extract_entities(provider: str = "auto", model: str | None = None, *,
                     concurrency: int = 4, limit: int | None = None,
                     on_progress=None) -> dict:
    """Extract entities + relations per section and wire them into the graph."""
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RuntimeError("No LLM provider configured for entity extraction.")
    g = kg.get_graph("overall")
    chunks_by_section: dict[str, list] = {}
    sections = []
    for n in g["nodes"]:
        if n.get("type") == "section":
            sections.append(n)
        elif n.get("type") == "chunk" and n.get("section_id"):
            chunks_by_section.setdefault(n["section_id"], []).append(n)
    units = []
    for s in sections:
        chs = chunks_by_section.get(s["id"], [])
        text = "\n\n".join(c.get("text", "") for c in chs)[:6000]
        if text.strip():
            units.append((s["id"], [c["id"] for c in chs], text))
    if limit:
        units = units[:limit]

    items, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(extraction.extract_kg_llm, t, rp, rm): (sid, cids)
                for sid, cids, t in units}
        for fut in cf.as_completed(futs):
            sid, cids = futs[fut]
            try:
                ext = fut.result() or {}
            except Exception:  # noqa: BLE001
                ext = {}
            items.append({"chunk_ids": cids, "entities": ext.get("entities", []),
                          "relations": ext.get("relations", [])})
            done += 1
            if on_progress and done % 20 == 0:
                on_progress(done, len(units))
    summary = kg.add_extractions(items)
    summary["sections_processed"] = len(units)
    summary["provider"] = f"{rp}/{rm}"
    summary["stats"] = kg.stats()["overall"]
    return summary


_TOPIC_PROMPT = """Group these entities from a knowledge library into at most {n}
coherent, non-overlapping THEMES (topics). Use short theme names. Every theme must
list the entity names (verbatim) that belong to it.

Entities:
{entities}

Return ONLY JSON: {{"topics": [{{"name": "<theme>", "entities": ["<entity>", ...]}}, ...]}}"""


def build_topics(provider: str = "auto", model: str | None = None, *,
                 top_n: int = 90, n_topics: int = 12) -> dict:
    """Cluster the most-mentioned entities into named topics and assign them."""
    rp, rm = resolve_provider_model(provider, model)
    if not rp:
        raise RuntimeError("No LLM provider configured for topic building.")
    ents = kg.list_entities(limit=top_n)
    if not ents:
        return {"topics": 0, "assigned": 0}
    names = {e["name"].lower(): e["name"] for e in ents}
    listing = "\n".join(f"- {e['name']} ({e.get('kind') or 'concept'})" for e in ents)
    chat = build_chat_model(rp, rm, max_tokens=2000)
    raw = _gen(chat, _TOPIC_PROMPT.format(n=n_topics, entities=listing))
    topics = []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            topics = json.loads(m.group()).get("topics", []) or []
        except Exception:  # noqa: BLE001
            topics = []
    created, assigned = 0, 0
    for t in topics:
        tname = (t.get("name") or "").strip()
        if not tname:
            continue
        kg.upsert_topic(tname, summary="")
        created += 1
        for en in t.get("entities", []):
            canonical = names.get(str(en).strip().lower())
            if canonical and kg.assign_entity_to_topic(canonical, tname):
                assigned += 1
    return {"topics": created, "assigned": assigned, "stats": kg.stats()["overall"]}
