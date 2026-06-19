#!/usr/bin/env python3
"""CLI for the local knowledge library agent (init / ingest / query / lint).

Mirrors the LangChain ``llm-wiki`` runner, but builds a personal, on-disk
knowledge library instead of syncing to a hub. Modes:

    init    scaffold the local /wiki workspace and catalog
    ingest  stage sources, build the graph, then let the agent organize them
    query   answer a question grounded in the library, with citations
    lint    whole-library maintenance pass (dedupe, topics, syntheses)

Examples:
    python runner.py --mode init
    python runner.py --mode ingest --source notes/ada.md --source notes/refs/
    python runner.py --mode ingest --url https://example.com/article
    python runner.py --mode query --question "What did Ada contribute?"
    python runner.py --mode lint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
config.load_env()  # pick up LangSmith + model keys from a local .env if present

import agent
import extraction
import ingestion
import knowledge_graph as kg
from providers import resolve_provider_model

_ALLOWED_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".csv", ".html", ".htm", ".pdf"}


def _read_source_file(path: Path) -> str:
    return ingestion.parse_file(path.name, path.read_bytes())


def _fetch_url(url: str) -> tuple[str, str]:
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0 (knowledge-library-cli)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return title, main.get_text(separator="\n", strip=True)


def _gather_sources(args) -> list[tuple[str, str, str]]:
    """Return (title, url, text) tuples from --source/--url/--text."""
    out: list[tuple[str, str, str]] = []
    for raw in args.source or []:
        p = Path(raw).expanduser()
        if not p.exists():
            print(f"warning: source not found: {p}", file=sys.stderr)
            continue
        files = [p] if p.is_file() else [f for f in sorted(p.rglob("*")) if f.is_file()]
        for f in files:
            if f.suffix.lower() not in _ALLOWED_SUFFIXES:
                continue
            try:
                out.append((f.name, "", _read_source_file(f)))
            except Exception as exc:  # noqa: BLE001
                print(f"warning: could not read {f}: {exc}", file=sys.stderr)
    for url in args.url or []:
        try:
            title, text = _fetch_url(url)
            out.append((title, url, text))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not fetch {url}: {exc}", file=sys.stderr)
    if args.text:
        out.append((args.title or "Pasted text", "", args.text))
    return out


def _stage_and_integrate(sources, provider, model, *, use_ai: bool, chunk_size: int, overlap: int):
    total_chunks = 0
    for title, url, text in sources:
        for i, c in enumerate(ingestion.chunk_text(text, chunk_size=chunk_size, overlap=overlap), 1):
            kg.add_chunk(c, source_url=url, source_title=title)
            total_chunks += 1
    extract_fn = None
    used_provider = "heuristic"
    if use_ai:
        rp, rm = resolve_provider_model(provider, model)
        if rp:
            used_provider = f"{rp}/{rm}"
            extract_fn = lambda t: extraction.extract_kg_llm(t, rp, rm)  # noqa: E731
    summary = kg.integrate(extract_fn=extract_fn)
    return total_chunks, used_provider, summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local knowledge library agent (deepagents harness).")
    p.add_argument("--mode", required=True, choices=[
        "init", "ingest", "query", "lint",
        "rag-ingest", "rag-ask", "rag-eval", "rag-experiment"])
    p.add_argument("--rerank", action="store_true", help="Enable the LLM re-ranker for RAG retrieval")
    p.add_argument("--topic", default="Knowledge", help="Display name for the library")
    p.add_argument("--workspace", default=None, help="Workspace dir (default: $KG_DATA_DIR/library)")
    p.add_argument("--source", action="append", default=[], help="File or directory to ingest (repeatable)")
    p.add_argument("--url", action="append", default=[], help="URL to fetch and ingest (repeatable)")
    p.add_argument("--text", default=None, help="Raw text to ingest")
    p.add_argument("--title", default=None, help="Source title for --text")
    p.add_argument("--question", default=None, help="Question for query mode")
    p.add_argument("--provider", default="auto", help="auto|anthropic|openai|qwen|deepseek")
    p.add_argument("--model", default=None, help="Model id override")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--no-agent", action="store_true", help="Skip the agent organize/maintain step")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    workspace = Path(args.workspace) if args.workspace else None

    ts = config.tracing_status()
    if ts["enabled"]:
        print(f"[tracing] LangSmith → project '{ts['project']}' ({ts['endpoint']})", file=sys.stderr)

    if args.mode == "init":
        res = agent.run_init(topic=args.topic, workspace=workspace)
        print(f"Initialized library at {res['workspace']}")
        print(f"Stats: {res['stats']}")
        return 0

    if args.mode == "ingest":
        sources = _gather_sources(args)
        if not sources:
            print("error: provide --source / --url / --text to ingest", file=sys.stderr)
            return 2
        n, used, summary = _stage_and_integrate(
            sources, args.provider, args.model, use_ai=not args.no_agent,
            chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Staged {n} chunk(s) from {len(sources)} source(s); extractor: {used}")
        print(f"Integrated: +{summary['entities_added']} entities, "
              f"+{summary['relation_edges_added']} relations, "
              f"{summary['sources_added']} sources")
        if not args.no_agent:
            try:
                res = agent.run_ingest(provider=args.provider, model=args.model,
                                       topic=args.topic, workspace=workspace)
                print("\n--- Agent organize report ---\n" + res["report"])
                print(f"\nLibrary now: {res['stats']}")
            except agent.AgentError as exc:
                print(f"\n(agent organize skipped: {exc})", file=sys.stderr)
        return 0

    if args.mode == "query":
        if not args.question:
            print("error: --question is required for query mode", file=sys.stderr)
            return 2
        try:
            res = agent.run_query(args.question, provider=args.provider, model=args.model,
                                  topic=args.topic, workspace=workspace)
        except agent.AgentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(res["answer"])
        if res["citations"]:
            print("\n--- Citations ---")
            for c in res["citations"]:
                print(f"- {c['chunk_id']} · {c['source_title'] or c['source_url'] or 'source'}")
        if res.get("filed"):
            print(f"\n(Filed to wiki/{res['filed']})")
        return 0

    if args.mode == "lint":
        try:
            res = agent.run_lint(provider=args.provider, model=args.model,
                                 topic=args.topic, workspace=workspace)
        except agent.AgentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(res["report"])
        print(f"\nEntities: {res['before']['entities']} -> {res['after']['entities']}; "
              f"topics: {res['after']['topics']}")
        return 0

    if args.mode == "rag-ingest":
        import pipeline
        urls = list(args.url or [])
        for raw in args.source or []:
            p = Path(raw).expanduser()
            if p.is_file():
                urls += [ln.strip() for ln in p.read_text().splitlines()
                         if ln.strip() and ln.strip().startswith("http")]
        if not urls:
            print("error: provide --url (repeatable) or --source <file-of-urls>", file=sys.stderr)
            return 2
        def _prog(r):
            if "error" in r:
                print(f"  ERROR {r['url']}: {r['error']}", file=sys.stderr)
            else:
                print(f"  ok [{r['date']:>14}] {r['sections']:>2} sec / {r['chunks']:>3} chunks  {r['title'][:48]}")
        try:
            res = pipeline.ingest_urls(urls, provider=args.provider, model=args.model,
                                       on_progress=_prog)
        except pipeline.PipelineError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"\nVector store: {res['vector_stats']}")
        print(f"Graph: {res['graph_stats']}")
        return 0

    if args.mode == "rag-ask":
        if not args.question:
            print("error: --question is required for rag-ask", file=sys.stderr)
            return 2
        import rag
        try:
            res = rag.answer(args.question, provider=args.provider, model=args.model,
                             rerank_hits=args.rerank)
        except rag.RagError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(res["answer"])
        if res.get("citations"):
            print(f"\n--- Retrieved passages{' (re-ranked)' if args.rerank else ''} ---")
            for c in res["citations"]:
                print(f"  [{c['n']}] score={c.get('score')} {c['date'] or 'n/a'} · {c['title'][:48]} · {c['url']}")
        return 0

    if args.mode == "rag-eval":
        import rag, json as _json
        try:
            eval_set = rag.build_eval_set(provider=args.provider, model=args.model)
            report = rag.evaluate(eval_set, provider=args.provider, model=args.model,
                                  rerank_hits=args.rerank)
        except rag.RagError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
        print("\n--- per-question ---")
        for r in report["rows"]:
            print(f"  hit={r['retrieval_hit']} rank={r['rank']} score={r['answer_score']} :: {r['question'][:70]}")
        return 0

    if args.mode == "rag-experiment":
        import rag_experiment, json as _json
        try:
            res = rag_experiment.run_experiment(provider=args.provider, model=args.model,
                                                rerank=args.rerank)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_json.dumps(res, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
