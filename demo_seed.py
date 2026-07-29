"""One-time pre-seed for the public demo.

Populates a read-only library (cache + vectors + knowledge graph) from a curated
URL list so the demo's Q&A and KG work instantly, with no per-visitor ingestion
cost. Run ONCE by the owner, pointing KG_DATA_DIR at the demo's data directory,
BEFORE launching the demo process.

Run with DEMO_MODE OFF (seeding must spend freely; the demo's budget meter would
otherwise gate it). Uses the cheap general model for contextualization to keep
seeding cheap. Override the list with DEMO_SEED_URLS (comma/newline separated).

    KG_DATA_DIR=./demo_data \
    DEMO_SEED_PROVIDER=gemini DEMO_SEED_MODEL=gemini-3.5-flash \
    python demo_seed.py
"""

from __future__ import annotations

import os
import re
import sys

import config

config.load_env()

# Curated, on-theme library (LangChain agent/eval/observability + Anthropic
# research). Override via DEMO_SEED_URLS.
DEFAULT_SEED_URLS = [
    "https://www.langchain.com/blog/the-art-of-loop-engineering",
    "https://www.langchain.com/blog/designing-efficient-verifiers-for-legal-agents",
    "https://www.langchain.com/blog/introducing-rubrics-for-deepagents",
    "https://www.langchain.com/blog/how-to-build-a-custom-agent-harness",
    "https://www.langchain.com/blog/interpreter-skills",
    "https://www.langchain.com/blog/human-judgment-in-the-agent-improvement-loop",
    "https://www.langchain.com/blog/better-harness-a-recipe-for-harness-hill-climbing-with-evals",
    "https://www.langchain.com/blog/traces-start-agent-improvement-loop",
    "https://www.langchain.com/resources/llm-evaluation-benchmarks",
    "https://www.langchain.com/resources/llm-evaluation-framework",
    "https://www.langchain.com/resources/agent-observability",
    "https://www.langchain.com/resources/llm-evals",
    "https://www.langchain.com/resources/llm-monitoring-observability",
    "https://www.langchain.com/resources/ai-observability",
    "https://www.langchain.com/resources/llm-evaluation-metrics",
    "https://www.anthropic.com/research/project-fetch-phase-two",
    "https://www.anthropic.com/research/claude-code-expertise",
    "https://www.anthropic.com/research/agents-in-biology",
    "https://www.anthropic.com/research/n-days",
    "https://www.anthropic.com/research/making-claude-a-chemist",
    "https://www.anthropic.com/research/attack-navigator",
    "https://www.anthropic.com/news/AI-enabled-cyber-threats-mitre-attack",
    "https://www.anthropic.com/research/coding-agents-social-sciences",
    "https://www.anthropic.com/research/exploit-evals",
    "https://www.anthropic.com/research/trustworthy-agents",
    "https://www.anthropic.com/research/zero-days",
    "https://www.anthropic.com/research/shade-arena-sabotage-monitoring",
    "https://www.anthropic.com/research/agentic-misalignment",
]


def _seed_urls() -> list[str]:
    raw = os.environ.get("DEMO_SEED_URLS", "")
    if raw.strip():
        return [u for u in re.split(r"[\s,]+", raw) if u.strip()]
    return list(DEFAULT_SEED_URLS)


def main() -> int:
    if os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
        print("WARNING: DEMO_MODE is set — seeding will be budget-gated and may stop "
              "early. Run the seed with DEMO_MODE unset.", file=sys.stderr)

    urls = _seed_urls()
    provider = os.environ.get("DEMO_SEED_PROVIDER", "gemini")
    model = os.environ.get("DEMO_SEED_MODEL", "gemini-3.5-flash")
    # Vectorizing already mirrors a rich structural graph (sources/sections/
    # chunks/edges) into the KG, which the KG view renders fully. LLM entity/
    # topic extraction is an OPTIONAL enrichment that's slow (sequential calls
    # over every chunk) — off by default. Set DEMO_SEED_KG=1 to run it.
    do_kg = os.environ.get("DEMO_SEED_KG", "0").strip().lower() in ("1", "true", "yes", "on")
    print(f"Seeding {len(urls)} URLs into {os.environ.get('KG_DATA_DIR', 'data')} "
          f"(model {provider}/{model}, kg={do_kg})")

    # Imported lazily so config/env are set first.
    import app
    import cached_store
    import pipeline
    import knowledge_graph as kg

    store = cached_store.get_store()
    # Reuse already-cached URLs (idempotent, cheap re-runs); only fetch new ones.
    by_url = {r.get("source_url"): r for r in store.list() if r.get("source_url")}
    records, ok, failed, reused = [], 0, 0, 0
    for i, url in enumerate(urls, 1):
        try:
            if url in by_url:
                records.append(by_url[url]); reused += 1
                print(f"  [{i}/{len(urls)}] reuse cached: {url[:70]}")
                continue
            page = app.fetch_page(url)  # PDF-aware fetch + boilerplate-stripped text
            text = (page.get("text") or "").strip()
            if not text:
                raise RuntimeError("no text extracted")
            rec = cached_store.ingest(kind="url", source_url=url,
                                      source_title=page.get("title") or url, raw_text=text)
            records.append(rec)
            ok += 1
            print(f"  [{i}/{len(urls)}] cached: {(page.get('title') or url)[:70]} "
                  f"({len(text):,} chars)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [{i}/{len(urls)}] FAILED {url}: {e}", file=sys.stderr)

    if not records:
        print("Nothing cached — aborting.", file=sys.stderr)
        return 1

    # ingest_cached_items needs the body (raw_text); public records omit it.
    full_records = []
    for r in records:
        full = store.get(r["id"], with_text=True)
        if full and (full.get("raw_text") or "").strip():
            full_records.append(full)

    print(f"Vectorizing {len(full_records)} docs (contextual retrieval) …")
    vres = pipeline.ingest_cached_items(full_records, provider=provider, model=model,
                                        vs_name="library")
    print(f"  vectorized: {vres}")
    for rec in full_records:
        try:
            store.set_flags(rec["id"], vectorized=True,
                            vector_projected_hash=rec.get("content_hash", ""))
        except Exception:  # noqa: BLE001
            pass

    if do_kg:
        print("Building knowledge graph (entity/relation extraction) …")
        import extraction
        for rec in full_records:
            try:
                kg.add_chunk(rec.get("raw_text") or "",
                             source_url=rec.get("source_url") or "",
                             source_title=rec.get("source_title") or "")
            except Exception as e:  # noqa: BLE001
                print(f"  kg add failed for {rec.get('source_url')}: {e}", file=sys.stderr)
        try:
            res = kg.integrate(extract_fn=lambda t: extraction.extract_kg_llm(t, provider, model))
            print(f"  kg integrated: {res}")
        except Exception as e:  # noqa: BLE001
            print(f"  kg integrate failed: {e}", file=sys.stderr)

    print(f"\nDone. cached={ok} failed={failed}. Library is ready for the demo.")
    print("Launch: DEMO_MODE=1 CACHED_STORE_BACKEND=local MEMORY_BACKEND=local "
          f"KG_DATA_DIR={os.environ.get('KG_DATA_DIR', 'data')} python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
