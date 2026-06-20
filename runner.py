#!/usr/bin/env python3
"""CLI for the local knowledge library agent (init / ingest / query / lint).

Mirrors the LangChain ``llm-wiki`` runner, but builds a personal, on-disk
knowledge library instead of syncing to a hub. Modes:

    init    scaffold the local /wiki workspace and catalog
    ingest  stage sources, build the graph, then let the agent organize them
    query   answer a question grounded in the library, with citations
    lint    whole-library maintenance pass (dedupe, topics, syntheses)

    memory-list / memory-recall / memory-add / memory-forget
            inspect and edit the cross-session memory layer (what the library
            has learned, durable answers, preferences, gaps, corrections)

    skill-build / skill-list / skill-show / skill-eval / skill-pending /
    skill-review / skill-rebuild / skill-export / skill-forget
            build a reusable agent skill (layer 7) from selected context, grade it
            (deterministic checks + LLM rubric), gate it, and align on it with a
            human before it joins the skill library

Examples:
    python runner.py --mode init
    python runner.py --mode ingest --source notes/ada.md --source notes/refs/
    python runner.py --mode ingest --url https://example.com/article
    python runner.py --mode query --question "What did Ada contribute?"
    python runner.py --mode lint
    python runner.py --mode memory-add --text "User is researching agent memory." --kind preference
    python runner.py --mode memory-recall --question "agent memory"
    python runner.py --mode memory-list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
config.load_env()  # pick up LangSmith + model keys from a local .env if present
config.ensure_tracing_project()  # resolve LANGSMITH_PROJECT_ID -> name for tracing

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
        "init", "ingest", "query", "lint", "rag-ingest", "rag-ask", "rag-eval",
        "rag-experiment", "rag-dataset", "rag-ragas", "rag-crossdoc",
        "rag-crossdoc-labels", "kg-extract",
        "memory-list", "memory-recall", "memory-add", "memory-forget",
        "skill-build", "skill-list", "skill-show", "skill-eval", "skill-pending",
        "skill-review", "skill-rebuild", "skill-refine", "skill-export", "skill-forget",
        "skill-runs", "skill-observability", "skill-backends", "skill-trace-init",
        "skill-graph-build", "skill-graph-resume", "skill-graph-status"])
    p.add_argument("--overwrite", action="store_true",
                   help="rag-crossdoc-labels: redraft key_points for already-labeled questions too")
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable the LLM re-ranker for RAG retrieval (default: on; use --no-rerank to disable)")
    p.add_argument("--mmr", action=argparse.BooleanOptionalAction, default=False,
                   help="Use document-aware MMR retrieval (diversifies top-k across docs; lifts multi-doc recall)")
    p.add_argument("--export", action="store_true",
                   help="rag-dataset: export LangSmith dataset to the template file (instead of syncing up)")
    p.add_argument("--topic", default="Knowledge", help="Display name for the library")
    p.add_argument("--workspace", default=None, help="Workspace dir (default: $KG_DATA_DIR/library)")
    p.add_argument("--source", action="append", default=[], help="File or directory to ingest (repeatable)")
    p.add_argument("--url", action="append", default=[], help="URL to fetch and ingest (repeatable)")
    p.add_argument("--text", default=None, help="Raw text to ingest")
    p.add_argument("--title", default=None, help="Source title for --text")
    p.add_argument("--question", default=None, help="Question for query / memory-recall mode")
    p.add_argument("--kind", default=None,
                   help="memory kind: fact|answer|preference|gap|correction|observation")
    p.add_argument("--salience", type=int, default=3, help="memory salience 1-5 (memory-add)")
    p.add_argument("--id", dest="mem_id", default=None, help="memory id (memory-forget)")
    p.add_argument("--provider", default="auto",
                   help="auto|anthropic|openai|qwen|deepseek|gemini|mistral")
    p.add_argument("--model", default=None, help="Model id override")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--no-agent", action="store_true", help="Skip the agent organize/maintain step")
    # Agent-skill layer (skill-*)
    p.add_argument("--goal", default=None, help="skill-build: what skill to build from the context")
    p.add_argument("--query", default=None, help="skill-build: pull matching chunks as context")
    p.add_argument("--tags", default=None, help="skill-build: comma-separated tags to pull chunks by")
    p.add_argument("--status", default=None, help="skill-list: filter by status")
    p.add_argument("--skill-id", dest="skill_id", default=None, help="skill id for show/eval/review/...")
    p.add_argument("--decision", default=None, help="skill-review: accept | reject | revise")
    p.add_argument("--score", type=float, default=None, help="skill-review: human score 0-1")
    p.add_argument("--notes", default=None, help="skill-review: reviewer notes / revision guidance")
    p.add_argument("--rubric", action=argparse.BooleanOptionalAction, default=True,
                   help="skill-build/eval: run the LLM rubric panel (default: on; --no-rubric for deterministic-only)")
    p.add_argument("--tools", action=argparse.BooleanOptionalAction, default=True,
                   help="skill-build/refine: let the author use tools to check+refine its draft (default: on)")
    p.add_argument("--backend", default=None, choices=["pipeline", "claude_code"],
                   help="skill-build/refine generator: in-process pipeline or the Claude Code subprocess agent")
    p.add_argument("--thread-id", dest="thread_id", default=None,
                   help="skill-graph-resume/status: the LangGraph thread to resume/inspect")
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
                             rerank_hits=args.rerank, mmr=args.mmr)
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
                                  rerank_hits=args.rerank, mmr=args.mmr)
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

    if args.mode == "kg-extract":
        import enrich, json as _json
        try:
            ext = enrich.extract_entities(provider=args.provider, model=args.model,
                                          on_progress=lambda d, t: print(f"  extracted {d}/{t} sections", flush=True))
            print("entities/relations:", _json.dumps({k: ext[k] for k in
                  ("entities_added", "relations_added", "mentions_added", "sections_processed")}))
            top = enrich.build_topics(provider=args.provider, model=args.model)
            print("topics:", _json.dumps({k: top.get(k) for k in ("topics", "assigned")}))
            print("library:", _json.dumps(top.get("stats", ext.get("stats"))))
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.mode == "rag-crossdoc":
        import crossdoc, json as _json
        try:
            res = crossdoc.run_experiment(provider=args.provider, model=args.model,
                                          rerank=args.rerank, mmr=args.mmr)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_json.dumps(res, indent=2))
        return 0

    if args.mode == "rag-crossdoc-labels":
        import crossdoc, json as _json
        try:
            res = crossdoc.scaffold_human_labels(provider=args.provider, model=args.model,
                                                 overwrite=args.overwrite)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Seeded key_points into {res['path']}")
        print(f"  labeled={res['labeled']}  added={res['added']}  needs_review={res['needs_review']}")
        print("Next: open the file and fill in human_score (0-1) + reviewed_answer for each entry,")
        print("then re-run `--mode rag-crossdoc` to see judge↔human alignment.")
        return 0

    if args.mode == "rag-ragas":
        import ragas_eval, json as _json
        try:
            res = ragas_eval.run_ragas_experiment(provider=args.provider, model=args.model,
                                                  rerank=args.rerank)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_json.dumps(res, indent=2))
        return 0

    if args.mode == "rag-dataset":
        import rag_experiment
        try:
            if args.export:
                res = rag_experiment.export_dataset()
                print(f"Exported dataset to {res['path']}: {res['examples']} examples (id {res['id']})")
            else:
                res = rag_experiment.sync_dataset()
                print(f"Synced dataset to LangSmith: '{res['name']}' "
                      f"id={res['id']} ({res['examples']} examples)")
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.mode == "memory-list":
        import memory, json as _json
        print(_json.dumps(memory.stats(), indent=2))
        rows = memory.list_memories(kind=args.kind, limit=200)
        print(f"\n--- {len(rows)} memories ---")
        for r in rows:
            print(f"  [{r['kind']:<11}] s{r['salience']} ×{r.get('use_count', 0):<2} "
                  f"{r['id']}  {r['preview']}")
        return 0

    if args.mode == "memory-recall":
        if not args.question:
            print("error: --question is required for memory-recall", file=sys.stderr)
            return 2
        import memory
        rows = memory.recall(args.question, k=10,
                             kinds=[args.kind] if args.kind else None)
        if not rows:
            print("(no relevant memories)")
            return 0
        for r in rows:
            print(f"  [{r['kind']}] score={r['score']} sim={r['similarity']}  {r['text']}")
        return 0

    if args.mode == "memory-add":
        if not args.text:
            print("error: --text is required for memory-add", file=sys.stderr)
            return 2
        import memory
        rec = memory.remember(args.text, kind=(args.kind or "fact"),
                              salience=args.salience, confidence="USER", origin="cli")
        if not rec:
            print("error: nothing stored", file=sys.stderr)
            return 1
        print(f"Stored {rec['id']} [{rec['kind']}] salience={rec['salience']}")
        return 0

    if args.mode == "memory-forget":
        if not args.mem_id:
            print("error: --id is required for memory-forget", file=sys.stderr)
            return 2
        import memory
        print("removed" if memory.forget(args.mem_id) else "not found")
        return 0

    # --- Agent-skill layer (layer 7): build / eval / human review / library ---
    if args.mode == "skill-build":
        tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] or None
        if not (args.query or args.text or tags):
            print("error: provide --query, --text, or --tags as context for skill-build",
                  file=sys.stderr)
            return 2
        import skill_agent, skill_library as sk, json as _json
        try:
            res = skill_agent.build_skill(
                query=args.query, text=args.text, tags=tags, goal=args.goal or "",
                provider=args.provider, model=args.model, run_rubric=args.rubric,
                use_tools=args.tools, backend=args.backend)
        except skill_agent.SkillError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        s, ev = res["skill"], res["eval"]
        obs = res.get("observability", {})
        print(f"Drafted skill '{s['name']}' ({s['id']})  →  gate={res['gate']}  status={s['status']}")
        print(f"  backend: {res.get('backend')}  ·  author: {res['provider']}/{res['model']}"
              f"{' +tools (%d calls)' % obs.get('tools_used', 0) if obs.get('tool_mode') else ''}"
              f"  ·  {obs.get('duration_ms', '?')} ms  ·  {obs.get('tokens', '?')} tokens")
        det = ev["deterministic"]
        print(f"  deterministic: {det['passed']}/{det['total']} passed" +
              (f"  (failures: {', '.join(det['failures'])})" if det['failures'] else ""))
        if ev.get("rubric") and ev["rubric"].get("mean") is not None:
            print(f"  rubric mean: {ev['rubric']['mean']}  per-dim: "
                  f"{_json.dumps(ev['rubric'].get('per_dimension', {}))}")
        if ev.get("triggering"):
            t = ev["triggering"]
            print(f"  triggering: precision={t.get('precision')} recall={t.get('recall')} f1={t.get('f1')}")
        print(f"  gate reasons: {'; '.join(ev.get('gate_reasons', []))}")
        if s["status"] == sk.PENDING_REVIEW:
            print("\nNext: review it with "
                  f"`--mode skill-review --skill-id {s['id']} --decision accept|reject|revise`")
        return 0

    if args.mode == "skill-list":
        import skill_library as sk, json as _json
        print(_json.dumps(sk.stats(), indent=2))
        rows = sk.list_skills(status=args.status, limit=200)
        print(f"\n--- {len(rows)} skill(s) ---")
        for r in rows:
            print(f"  [{r['status']:<14}] {r['id']}  {r['name']}  —  {r.get('preview', '')}")
        return 0

    if args.mode == "skill-pending":
        import skill_library as sk
        rows = sk.pending_review()
        if not rows:
            print("(no skills awaiting review)")
            return 0
        print(f"--- {len(rows)} skill(s) awaiting human alignment ---")
        for r in rows:
            ev = r.get("eval") or {}
            print(f"  {r['id']}  {r['name']}  (gate={ev.get('gate')}, rubric={ev.get('rubric_mean')})")
            print(f"     {r.get('description', '')}")
        return 0

    if args.mode == "skill-show":
        if not args.skill_id:
            print("error: --skill-id is required for skill-show", file=sys.stderr)
            return 2
        import skill_library as sk
        s = sk.get_skill(args.skill_id)
        if not s:
            print("not found", file=sys.stderr)
            return 1
        print(sk.to_skill_md(s))
        return 0

    if args.mode == "skill-eval":
        if not args.skill_id:
            print("error: --skill-id is required for skill-eval", file=sys.stderr)
            return 2
        import skill_library as sk, skill_eval, json as _json
        s = sk.get_skill(args.skill_id)
        if not s:
            print("not found", file=sys.stderr)
            return 1
        report = skill_eval.run_eval(s, provider=args.provider, model=args.model,
                                     run_rubric=args.rubric)
        sk.record_eval(s["id"], report)
        print(_json.dumps({k: v for k, v in report.items() if k != "rubric"}, indent=2))
        if report.get("rubric"):
            print("rubric:", _json.dumps({"mean": report["rubric"].get("mean"),
                                          "per_dimension": report["rubric"].get("per_dimension")}))
        return 0

    if args.mode == "skill-review":
        if not args.skill_id or not args.decision:
            print("error: --skill-id and --decision (accept|reject|revise) are required",
                  file=sys.stderr)
            return 2
        import skill_agent
        try:
            res = skill_agent.review_skill(
                args.skill_id, decision=args.decision, score=args.score,
                notes=args.notes or "", rebuild_on_revise=False)
        except skill_agent.SkillError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{res['skill']['name']} → status={res['status']}")
        return 0

    if args.mode == "skill-rebuild":
        if not args.skill_id:
            print("error: --skill-id is required for skill-rebuild", file=sys.stderr)
            return 2
        import skill_agent
        try:
            res = skill_agent.rebuild_skill(args.skill_id, provider=args.provider,
                                            model=args.model, extra_guidance=args.notes or "",
                                            run_rubric=args.rubric)
        except skill_agent.SkillError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Rebuilt '{res['skill']['name']}' → gate={res['gate']} status={res['status']}")
        return 0

    if args.mode == "skill-refine":
        if not args.skill_id:
            print("error: --skill-id is required for skill-refine", file=sys.stderr)
            return 2
        import skill_agent
        try:
            res = skill_agent.refine_skill(args.skill_id, provider=args.provider,
                                           model=args.model, use_tools=args.tools,
                                           run_rubric=args.rubric, backend=args.backend)
        except skill_agent.SkillError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        obs = res.get("observability", {})
        print(f"Refined '{res['skill']['name']}' → gate={res['gate']} status={res['status']} "
              f"({obs.get('duration_ms','?')} ms, {obs.get('tokens','?')} tokens)")
        return 0

    if args.mode == "skill-runs":
        import skill_runs
        rows = skill_runs.list_runs(skill_id=args.skill_id, limit=50)
        print(f"--- {len(rows)} run(s) ---")
        for r in rows:
            m = r.get("metrics") or {}
            print(f"  [{r['kind']:<7}] {r.get('at','')[:19]}  {r.get('model','')}  "
                  f"gate={r.get('gate')}  {r.get('duration_ms','?')}ms  {r.get('tokens','?')}tok  "
                  f"det={m.get('deterministic_ratio')} rubric={m.get('rubric_mean')}  {r.get('skill_name','')}")
        return 0

    if args.mode == "skill-observability":
        import skill_runs, skill_tracing, json as _json
        print(_json.dumps(skill_runs.benchmark(skill_id=args.skill_id), indent=2))
        print("tracing:", _json.dumps(skill_tracing.status()))
        return 0

    if args.mode == "skill-graph-build":
        tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] or None
        if not (args.query or args.text or tags):
            print("error: provide --query, --text, or --tags as context", file=sys.stderr)
            return 2
        import skill_graph
        try:
            res = skill_graph.run_build(query=args.query, text=args.text, tags=tags,
                                        goal=args.goal or "", provider=args.provider, model=args.model,
                                        backend=args.backend, use_tools=args.tools,
                                        run_rubric=args.rubric)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if res.get("awaiting_review"):
            s = res.get("skill") or {}
            print(f"Build paused for review (checkpoint={skill_graph.checkpoint_backend()}).")
            print(f"  thread_id: {res['thread_id']}")
            print(f"  skill: {s.get('name')} ({s.get('id')})  gate={res.get('gate')}")
            print(f"\nResume: --mode skill-graph-resume --thread-id {res['thread_id']} "
                  f"--decision accept|reject|revise [--notes '...']")
        else:
            print(f"Build finished: status={res.get('status')}")
        return 0

    if args.mode == "skill-graph-resume":
        if not args.thread_id or not args.decision:
            print("error: --thread-id and --decision (accept|reject|revise) are required",
                  file=sys.stderr)
            return 2
        import skill_graph
        try:
            res = skill_graph.resume_review(args.thread_id, decision=args.decision,
                                            score=args.score, notes=args.notes or "")
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if res.get("awaiting_review"):
            print(f"Revised and paused again — thread_id {res['thread_id']} "
                  f"(skill v{(res.get('skill') or {}).get('version')}). Resume again to decide.")
        else:
            print(f"Done: status={res.get('status')}")
        return 0

    if args.mode == "skill-graph-status":
        if not args.thread_id:
            print("error: --thread-id is required", file=sys.stderr)
            return 2
        import skill_graph, json as _json
        print(_json.dumps(skill_graph.get_status(args.thread_id), indent=2))
        return 0

    if args.mode == "skill-trace-init":
        import skill_tracing, json as _json
        res = skill_tracing.ensure_project()
        print(_json.dumps(res, indent=2))
        print("status:", _json.dumps(skill_tracing.status()))
        if not res.get("ok"):
            print("\nSet LANGSMITH_API_KEY and SKILL_TRACING=true (and optionally "
                  "LANGSMITH_SKILL_PROJECT) to enable skill tracing.", file=sys.stderr)
        return 0

    if args.mode == "skill-backends":
        import skill_agent, skill_claude_agent, json as _json
        cc = skill_claude_agent.status()
        print(f"default backend: {skill_agent.DEFAULT_BACKEND}")
        print(f"  pipeline     : in-process LLM phases")
        print(f"  claude_code  : {'available' if cc['available'] else 'NOT FOUND'} "
              f"(bin={cc['bin']}, model={cc['model']}, sdk={'yes' if cc['sdk_installed'] else 'no'})")
        if not cc["available"]:
            print("  → install the Claude Code CLI (npm i -g @anthropic-ai/claude-code) and authenticate it,")
            print("    or set CLAUDE_CODE_BIN, to use --backend claude_code.")
        return 0

    if args.mode == "skill-export":
        if not args.skill_id:
            print("error: --skill-id is required for skill-export", file=sys.stderr)
            return 2
        import skill_library as sk
        res = sk.export_skill(args.skill_id)
        print(f"Wrote {res['path']}" if res.get("ok") else f"error: {res.get('error')}")
        return 0 if res.get("ok") else 1

    if args.mode == "skill-forget":
        if not args.skill_id:
            print("error: --skill-id is required for skill-forget", file=sys.stderr)
            return 2
        import skill_library as sk
        print("removed" if sk.forget(args.skill_id) else "not found")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
