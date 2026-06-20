# Build Your Own WIKI

<img width="1695" height="928" alt="ChatGPT Image Jun 19, 2026, 10_56_01 PM" src="https://github.com/user-attachments/assets/0b73a45f-65e8-4170-ab05-474049fc6a77" />

Build **your own wiki** from the web, files, and text: search and summarize pages
(or paste a link to extract its context), ingest them into a contextual vector
index for grounded, cited Q&A, and grow a layered knowledge graph an LLM agent
keeps coherent. One Flask app — web UI, JSON API, and a `runner.py` CLI. Inspired
by the LangChain *llm-wiki* deep-agents example, but it builds a personal, on-disk
library instead of syncing to a hub.

Four cooperating parts:

- **Agentic knowledge graph** — save passages, extract entities + typed relations,
  and let a [`deepagents`](https://pypi.org/project/deepagents/) agent (local
  filesystem backend, no cloud sandbox) de-duplicate entities, build a topic
  hierarchy, and write synthesis pages.
- **Contextual RAG library** — a two-layer HNSW index built with Anthropic-style
  *contextual retrieval*; answers are grounded with hierarchical retrieval plus an
  LLM re-ranker or document-aware MMR.
- **Memory layer** — a cross-session store the library recalls *before* and writes
  back *after* every answer, so it improves from use, not just from ingestion.
- **Agent-skill library** — on request, a sub-agent pipeline turns the context you
  select into a reusable, *evaluated* agent skill: it understands → analyzes →
  authors (codeact, with tool-use) → evaluates → gates the skill, and a human signs
  off before it joins the library. Skills are authored by the latest Claude — in
  process or via the **Claude Code CLI as a tool-using subprocess** — written to
  `SKILL.md`, measured in an observability store, and self-improved via `refine`.
  So the library grows what it can *do*, not just what it knows.

## Providers

Every AI feature reads one shared provider table, so a single key lights up the
app. Pick provider/model in the UI, or pass `--provider`/`--model` on the CLI.

| Provider | API-key env | Default model | OpenAI-compatible |
| --- | --- | --- | --- |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | no |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | yes |
| Qwen (DashScope) | `DASHSCOPE_API_KEY` | `qwen-plus` | yes |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` | yes |
| Google (Gemini) | `GEMINI_API_KEY` | `gemini-3.5-flash` | yes |
| Mistral | `MISTRAL_API_KEY` | `mistral-large-latest` | yes |

With no key, summaries fall back to a local extractive summarizer. **RAG features
need `OPENAI_API_KEY`** regardless of chat provider — embeddings use OpenAI
`text-embedding-3-small` (Anthropic has no embeddings API).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add `ragas` for the RAGAS metrics
cp .env.example .env                      # set at least one provider key
python app.py                            # → http://localhost:5000
```

Both `app.py` and `runner.py` load `.env` automatically.

## How it works

**Layered knowledge model.** The library is a multi-layer property graph, not a
flat entity bag:

| Layer | Node | What it is |
| --- | --- | --- |
| 0 | `source` | a document / URL / file |
| 1 | `section` | a contextual-summary span within a source |
| 2 | `chunk` | an immutable passage of evidence |
| 3 | `entity` | a canonical, de-duplicated thing |
| 4 | `topic` | a theme grouping entities (can nest) |
| 5 | `synthesis` | an agent-written note unifying evidence |
| 6 | `memory` | a durable, cross-session learning |
| 7 | `agent_skill` | a reusable, evaluated competence built from selected context |

Typed edges (`from_source`, `in_section`, `mentions`, `relation`, `belongs_to`,
`subtopic_of`, `shares`, …) connect layers. The KG tab renders three
**granularities** — document / section / chunk — via `GET /api/kg/graph`. Stores
are plain JSON under `data/`.

**Contextual retrieval.** Each page is fetched, cleaned, and given one
LLM-written **contextual summary per section**; chunks are embedded with their
`title · date · section` prepended. `vectorstore.py` keeps a two-layer **HNSW**
index (section summaries + chunks). Retrieval (`rag.py`) ranks sections, drills
into chunks (blended score), then either an **LLM re-ranker** (default, precision)
or **document-aware MMR** (`--mmr`, multi-doc recall) selects the top-k.

**What the graph is for.** Retrieval is purely vector-based — entity-graph
traversal was evaluated and **removed** (MMR beat it on recall *and* judged
synthesis). The graph earns its keep in **construction** (canonicalization,
topics, provenance) and **presentation** (document map of shared-entity links,
zoomable views, browse-by-concept).

**Memory layer** (`memory.py`, layer 6). A separate JSON store that **recalls**
relevant memories into the prompt before each pass and **writes back** after
(answers, observations, 👍/👎 feedback, corrections that *supersede* stale notes).
Memories reinforce with reuse and de-duplicate on re-assertion. Evaluation runs
disable memory so metrics stay deterministic.

**Agent-skill layer** (`skill_*.py`, layer 7). A skill isn't knowledge quoted from
a source (a chunk) or a learning recalled across sessions (a memory) — it's an
*executable competence*: a name, a description (the signal an agent reads to decide
whether to invoke it), step-by-step instructions, declared tools, positive *and*
negative trigger cases, success criteria, and a small test set. When you feed the
library selective context and ask for a skill, a sub-agent pipeline
(`skill_agent.py`) runs explicit phases — **understand** (state what the context
is) → **analyze** (design the spec + success criteria) → **codeact** (author the
skill) → **eval** → **gate** — modeled on OpenAI's
[*Evaluating skills*](https://developers.openai.com/blog/eval-skills) loop.
Evaluation (`skill_eval.py`) is two-layered: fast **deterministic checks** (name,
usable description, ≥2 explicit steps, triggers + negative controls, declared
tools, positive/negative tests, no placeholders, grounded in its source context)
plus a cross-family **rubric judge panel** that scores *outcome / process / style /
efficiency* and groundedness. The **accept/reject gate** combines the two but never
finalizes on its own: a passing skill lands in `pending_review`, and only a
**human** (`review_skill`) moves it to `accepted` (or `revise`, which folds notes
back into a fresh `codeact` pass). The store (`skill_library.py`) reports how often
the human's call matched the gate, and accepted skills are recallable — the
curating agent calls `skill_recall` to find a proven skill for a repeatable task.

The author is **tool-using and measured**, following Anthropic's improved
skill-creator and the Hermes *act → verify → save* loop. Two interchangeable
generator **backends** produce the skill; both feed the *same* evaluation/gate, so
the eval step is backend-agnostic:

- `pipeline` (default) — in-process LLM phases. Skills are generated by the
  strongest available Claude (`claude-opus-4-8`, overridable) through a native
  tool-calling loop (`skill_runtime.py`) that can read the source context, recall
  memory, list existing skills (avoid duplicates), and **check its own draft** with
  the deterministic checks before finalizing — test → measure → refine in one pass.
- `claude_code` — the **Claude Code CLI invoked as a subprocess**
  (`skill_claude_agent.py`): a real tool-using agent that reads the context from a
  scratch workspace and writes `skill.json` + `SKILL.md`, recovered from disk or its
  final message. Requires the `claude` CLI installed and authenticated; it incurs
  network latency and cost (captured in observability).

Either backend writes the portable `SKILL.md` to disk. Every build / eval / refine / review
is logged to an **observability store** (`skill_runs.py`) with per-phase timings,
token usage, the gate decision, and triggering **precision/recall** (does the
description fire on the positive tests and stay quiet on the negative controls?);
`benchmark()` aggregates those into a gate pass-rate and average latency/tokens.
Each run is also mirrored to **LangSmith** (and optional **OTel** spans) in its own
dedicated project (`skill_tracing.py`, `LANGSMITH_SKILL_PROJECT`, default
"BYO-WIKI Agent Skills") — separate from the agent-layer project — with the metrics
attached as numeric feedback so they chart; the Claude Code subprocess runs in its
own process, so its tokens/cost/turns come from the CLI's JSON envelope.
`refine_skill` closes the loop: it reads a skill's measured weaknesses (failed
checks, low triggering precision/recall, weakest rubric dimension) and rebuilds it
to fix them. Accepted skills are also saved to memory, so the agent auto-loads them
later — the memory↔skill partnership.

**LangGraph orchestration** (`skill_graph.py`, optional). The same pipeline can run
as a LangGraph `StateGraph` (understand → analyze → codeact → evaluate nodes, the
gate as a conditional edge, the refine loop as a real cycle), where **human review
is a durable `interrupt()`**: the build *pauses* at the gate, its state is
checkpointed, and it *resumes* — even in a separate process — when a decision
arrives (`accept` finalizes, `reject` ends, `revise` loops back through `codeact`
with the notes). Checkpoints persist to SQLite by default (`data/skill_graph.sqlite`),
swappable to Postgres (e.g. Supabase) or in-memory via `SKILL_GRAPH_CHECKPOINT`.

For durable, shared, cross-machine resume, point it at Postgres:

```bash
pip install "langgraph-checkpoint-postgres" "psycopg[binary]" psycopg-pool
export SKILL_GRAPH_CHECKPOINT=postgres
export SKILL_GRAPH_DB_URL="$SUPABASE_DB_URL"   # or any Postgres URL
# build on one machine (pauses), resume on another — same DB:
python runner.py --mode skill-graph-build --text "…" --goal "…"
python runner.py --mode skill-graph-resume --thread-id <tid> --decision accept
```

Checkpoint tables go in their own schema (`SKILL_GRAPH_PG_SCHEMA`, default
`skill_graph`) over a connection pool; prepared statements are disabled and
connections run autocommit, so the **Supabase transaction pooler (port 6543)**
works as well as a direct/session connection. If the DB is unreachable or the deps
are missing, it logs a note and **falls back to SQLite** — defaults stay local-first.
`runner.py --mode skill-backends` prints the effective checkpoint backend.

## Web UI

Tabs for **Read** (web search, or paste a link to fetch + extract its context;
summarize a URL or raw text; highlight to save a chunk to the KG), **Knowledge
Graph** (ingest Files/URLs/Text, browse staging + the layer-colored graph,
*Integrate*, *Ask*, *Maintain*), **Library Q&A** (cited answers with re-ranker /
MMR toggles), **Memory** (recall, add, reinforce/forget), and **Agent Skills**
(build a skill from context, watch the eval/gate report, then accept / revise /
reject from the review queue; toggle **durable ⛓** to run it as a checkpointed
LangGraph build that runs in the background, pauses for review, and can be resumed
later — the queue auto-refreshes and the tab badge pulses as builds finish or get
resumed).

## Command line

```bash
# Knowledge-graph workspace
python runner.py --mode init
python runner.py --mode ingest --source notes/ --url https://example.com/article
python runner.py --mode query --question "What did Ada contribute?"
python runner.py --mode lint                       # whole-library maintenance

# Memory
python runner.py --mode memory-add --text "…" --kind preference
python runner.py --mode memory-recall --question "agent memory"
python runner.py --mode memory-list | memory-forget --id memory_xxx

# Agent skills (layer 7): build from context → eval/gate → human review → library
python runner.py --mode skill-build --query "deep agents" --goal "draft a design brief"
python runner.py --mode skill-build --text "…" --backend claude_code   # Claude Code CLI subprocess
python runner.py --mode skill-backends                      # which generators are available
python runner.py --mode skill-pending                       # the human-review queue
python runner.py --mode skill-review --skill-id skill_xxx --decision accept --score 0.9
python runner.py --mode skill-refine --skill-id skill_xxx    # fix measured weaknesses
python runner.py --mode skill-observability                 # benchmark: pass-rate, latency, tokens
python runner.py --mode skill-trace-init                    # create the LangSmith project for skill runs
# LangGraph orchestrator: a checkpointed build that pauses at review and resumes later
python runner.py --mode skill-graph-build --text "…" --goal "…"   # prints a thread_id, pauses
python runner.py --mode skill-graph-resume --thread-id <tid> --decision accept   # (any later process)
python runner.py --mode skill-list | skill-runs | skill-show --skill-id skill_xxx

# Contextual RAG + evaluation
python runner.py --mode rag-ingest --source eval/corpus_urls.txt   # 28-doc corpus
python runner.py --mode rag-ask --question "How do rubrics help agents self-correct?"
python runner.py --mode rag-eval                   # local metrics (--no-rerank to A/B)
python runner.py --mode rag-experiment --rerank    # single-doc LangSmith experiment
python runner.py --mode rag-ragas                  # RAGAS metrics (needs `ragas`)
python runner.py --mode rag-crossdoc-labels        # draft key_points for human review
python runner.py --mode rag-crossdoc --mmr         # cross-document experiment
```

Common flags: `--provider`, `--model`, `--rerank/--no-rerank`, `--mmr`,
`--chunk-size`, `--overlap`, `--no-agent`.

## Evaluation

Everything runs on the version-controlled corpus (`eval/corpus_urls.txt`, 28
LangChain + Anthropic articles) with reusable dataset templates under `eval/`.

- **Local** (`rag-eval`): retrieval hit-rate@k, MRR, LLM-judge answer score.
- **Single-doc LangSmith** (`rag-experiment`): retrieval_hit, reciprocal_rank,
  judged answer_correctness — base vs. re-ranked over identical inputs.
- **RAGAS** (`rag-ragas`): faithfulness, answer relevancy, context precision.
- **Cross-document** (`rag-crossdoc`): questions that each require synthesizing
  **2–3 documents**, scored on retrieval_recall / any_hit, RAGAS, and a judged
  **synthesis correctness**.

**Grounded, calibrated judging.** The cross-doc synthesis judge sees real
evidence — excerpts of the gold documents **and** a gold `key_points` reference —
so "well-grounded" is checkable, not guessed from a title/URL. Human labels
(`eval/crossdoc_human_labels.json`) supply that reference plus a `human_score`;
each run reports **`judge_alignment`** (how closely each judge tracks the human,
`1 − MAE`) over *fresh* pairs, flagging answers that drifted from their reviewed
version as **stale → re-review**. A **judge panel** of five families avoids
self-preference bias and averages out any one model's strictness:

`gpt-5.2` · `qwen3.7-plus` · `deepseek-v4-flash` · `gemini-3.5-flash` ·
`mistral-large-2512`

Each gets a `correctness_<model>` column plus a panel mean; RAGAS is judged by the
first non-reasoning panel model. Absolute LLM-judge scores are judge-dependent, so
the **deterministic retrieval metrics are the most comparable across runs**.

### Results (28-doc corpus)

| Setting | Metric | Result |
| --- | --- | --- |
| Single-doc, re-ranked | hit@6 / MRR | 0.87 / 0.81 |
| Re-ranker off → on | RAGAS context precision | 0.51 → 0.67 |
| Cross-doc, base → MMR | retrieval recall | 0.59 → **0.91** |
| Cross-doc, MMR vs. graph-RAG | retrieval recall | **0.909** vs. 0.788 |
| Cross-doc, synthesis-prompt fix | RAGAS answer relevancy | 0.65 → **0.90** |
| Cross-doc, MMR (5-judge panel) | panel correctness | ≈ 0.68 |

MMR winning on recall *and* judged synthesis is why entity-graph retrieval was
removed. The answer-relevancy jump came from letting the generator synthesize
across passages instead of refusing when no single chunk states the connection.

## HTTP API

Grouped under `/api`: **read** (`/search` — web results, or extracted context when
the query is a link; `/summarize`; `/providers`), **knowledge graph** (`/kg/stats`,
`/kg/graph`, `/kg/add`, `/kg/integrate`, `/kg/query`, `/kg/ingest/{text,urls,
files}`), **agent** (`/agent/{status,ask,maintain}`), **memory** (`/memory/{stats,
list,recall,add,feedback}`), **RAG** (`/rag/{stats,ingest,search,ask,eval,
experiment,dataset,ragas,crossdoc,crossdoc/labels}`), and **skills**
(`/skill/{stats,list,pending,build,observability,runs,backends,tracing/init}`, the
LangGraph endpoints `/skill/graph/{build,resume,status}`, plus `/skill/<id>` and
`/skill/<id>/{eval,review,rebuild,refine,export}`). Most accept `{provider?,
model?}`; cross-doc returns `judge_alignment` when human labels exist, and
`/skill/build` (accepts `use_tools` and `backend` = `pipeline` | `claude_code`)
returns every phase plus the eval/gate report and an observability summary
(timings, tokens, tool calls, cost).

## Configuration

Set provider keys (above) plus, for tracing/eval:

| Variable | Purpose |
| --- | --- |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | enable + authenticate LangSmith. |
| `LANGSMITH_PROJECT_ID` | project by **ID** (rename-proof; resolved at runtime). |
| `LANGSMITH_RAG_DATASET_ID` / `LANGSMITH_CROSSDOC_DATASET_ID` | eval dataset IDs. |
| `*_BASE_URL` | override an OpenAI-compatible endpoint (proxy/Azure/region). |
| `KG_DATA_DIR` | store directory (default `data/`). |
| `KG_AGENT_RECURSION_LIMIT` | max agent steps for Maintain/`lint` (default 150). |
| `SKILL_GATE_ACCEPT` / `SKILL_GATE_REJECT` | skill-gate rubric thresholds (0.70 / 0.40). |
| `SKILL_DET_PASS` | deterministic pass-ratio a skill must clear (default 0.80). |
| `SKILL_GENERATOR_MODEL` | Claude model that authors skills (default `claude-opus-4-8`). |
| `SKILL_RUNS_MAX` | observability runs to keep in `data/skill_runs.json` (default 500). |
| `SKILL_BACKEND` | default generator: `pipeline` or `claude_code`. |
| `CLAUDE_CODE_BIN` / `CLAUDE_CODE_MODEL` | Claude Code CLI path + model for the `claude_code` backend. |
| `SKILL_TRACING` / `LANGSMITH_SKILL_PROJECT` | export skill runs to LangSmith + OTel, in their own project. |
| `SKILL_GRAPH_CHECKPOINT` | LangGraph checkpoint store: `sqlite` (default) / `postgres` / `memory`. |
| `PORT` | bind port (default 5000). |

`.env` is gitignored — never commit real keys. See `.env.example` for the full list.

## Project layout

```
app.py / runner.py      Flask app (UI + API) / CLI
agent.py kg_tools.py    deepagents harness + graph tools
memory.py memory_tools.py   cross-session memory (layer 6)
skill_library.py skill_eval.py skill_agent.py skill_tools.py   agent-skill build loop (layer 7)
skill_runtime.py skill_runs.py skill_claude_agent.py skill_tracing.py   tool-use loop · observability · Claude Code subprocess · LangSmith/OTel export
skill_graph.py          LangGraph StateGraph: durable human-in-the-loop + refine cycle
knowledge_graph.py      multi-layer graph store + queries
ingestion.py extraction.py enrich.py   chunking, entity/relation/topic extraction
pipeline.py embeddings.py vectorstore.py   contextual ingest + HNSW index + MMR
rag.py rag_experiment.py ragas_eval.py crossdoc.py   retrieval, answers, eval suite
providers.py config.py   provider table + judge selection; .env / LangSmith
eval/ static/ templates/ data/   corpus + dataset templates; UI; local stores
```

## Credits

Inspired by the LangChain *llm-wiki* deep-agents example. Built with Flask,
`deepagents`/LangChain, `hnswlib`, OpenAI embeddings, RAGAS, and LangSmith.
