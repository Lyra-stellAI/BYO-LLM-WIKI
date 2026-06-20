# Build Your Own WIKI

<img width="1695" height="928" alt="ChatGPT Image Jun 19, 2026, 10_56_01 PM" src="https://github.com/user-attachments/assets/0b73a45f-65e8-4170-ab05-474049fc6a77" />


Build **your own wiki** from the web, files, text. Search and summarize pages,
ingest them into a contextual vector index for grounded Q&A, and grow a layered
knowledge graph that an LLM agent helps keep coherent. It runs as a single Flask
app with a web UI, a JSON API, and a `runner.py` CLI.

The project began as a web-browsing summarizer and grew into two cooperating
halves:

- **An agentic knowledge graph** — save passages, extract entities and typed
  relations, and let a [`deepagents`](https://pypi.org/project/deepagents/) agent
  (running **locally** on a filesystem backend, no cloud sandbox) de-duplicate
  entities, build a topic hierarchy, and write canonical synthesis pages.
- **A contextual RAG library** — ingest pages into a two-layer HNSW vector index
  built with Anthropic-style *contextual retrieval*, then answer cited questions
  with hierarchical retrieval plus an LLM re-ranker or document-aware MMR.
- **A memory layer** — a cross-session memory the library *recalls before* and
  *writes back after* every answer and maintenance pass, so it keeps improving
  from its own use instead of only growing when you ingest documents.

It is inspired by the LangChain *llm-wiki* deep-agents example: it reuses that
harness and its `init` / `ingest` / `query` / `lint` orchestration, but builds a
personal, on-disk library instead of syncing to a hub.

## Providers

Every AI feature — summaries, the agent, RAG answers, eval judges — reads one
shared provider table, so a single API key lights up the whole app. Pick a
provider/model from the UI dropdowns (the model field is free-form, so any model
ID the provider supports works), or set `--provider`/`--model` on the CLI.

| Provider | API-key env var | Default model | OpenAI-compatible |
| --- | --- | --- | --- |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | no |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | yes |
| Qwen (DashScope) | `DASHSCOPE_API_KEY` | `qwen-plus` | yes |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` | yes |
| Google (Gemini) | `GEMINI_API_KEY` | `gemini-3.5-flash` | yes (compat endpoint) |
| Mistral | `MISTRAL_API_KEY` | `mistral-large-latest` | yes |

If no key is set, summaries fall back to a local extractive summarizer that needs
no key. **The RAG vector features need an `OPENAI_API_KEY`** regardless of the
chat provider, because embeddings use OpenAI `text-embedding-3-small` (Anthropic
has no embeddings API).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy the env template and fill in at least one provider key.
cp .env.example .env
$EDITOR .env            # set ANTHROPIC_API_KEY / OPENAI_API_KEY / ...

python app.py
```

Then open <http://localhost:5000>. Both `app.py` and `runner.py` load `.env`
automatically on startup. To run the RAGAS metrics, also `pip install ragas`
(a heavy, optional dependency that is imported lazily).

## How it works

### Layered knowledge model

Instead of a flat bag of entities, the library is a hierarchical, multi-layer
property graph that unifies disparate sources into reusable understanding:

| Layer | Node | What it is |
| --- | --- | --- |
| 0 | `source` | a document / URL / file the knowledge came from |
| 1 | `section` | a contextual-summary span within a source |
| 2 | `chunk` | an immutable passage of evidence |
| 3 | `entity` | a canonical, de-duplicated thing (person, org, concept, …) |
| 4 | `topic` | a theme that groups related entities (can nest) |
| 5 | `synthesis` | an agent-written canonical note unifying the evidence |
| 6 | `memory` | a durable, cross-session learning (see [Memory layer](#memory-layer-keeping-the-library-dynamic)) |

Typed edges connect the layers: `from_source`, `in_section`, `mentions`,
`relation` (entity→entity predicate), `belongs_to` (entity→topic), `subtopic_of`,
`covers`, `cites`, and `shares` (sources that share entities).

The graph can be viewed at three **granularities** (a selector in the KG tab, or
`GET /api/kg/graph?granularity=…`): **document** (the high-level map of
sources/topics/syntheses, plus shared-entity links), **section** (adds the
contextual-summary layer), or **chunk** (the full fine-grained graph). Coarse
levels summarize; fine levels expose evidence.

Stores are plain JSON: `data/current.json` (staging) and `data/overall.json`
(integrated). The agent's human-readable wiki lives under `data/library/wiki/`
(`index.md` catalog, `topics/`, `synthesis/`, `query/`) with an append-only
`data/library/log.md` timeline — the same shape as the llm-wiki workspace.

### What the knowledge graph is for

Retrieval is **purely vector-based** (hierarchical search + re-ranker / MMR).
Entity-graph traversal was evaluated as a retrieval path and **removed** —
document-aware MMR beat it on cross-document recall *and* judged synthesis
quality, because rare entities can't bridge documents while generic "hub"
entities over-connect them. The knowledge graph instead earns its keep in:

- **Construction** — the `source → section → chunk → entity → topic` hierarchy is
  the organizing skeleton the vector index mirrors. Entity canonicalization
  (`groom` / the agent's *Maintain* pass) merges the same concept across many
  documents into one node — this is what "unifies disparate data." Topics give a
  controlled vocabulary, and every node keeps its provenance (source URL, date).
- **Presentation** — the **document map** (`granularity=document`) links sources
  that share ≥2 entities into a navigable thematic graph; **zoomable views** show
  the corpus at three altitudes; entity/topic browsing lets you navigate by
  concept instead of keyword.

In short: embeddings + MMR/re-rank *answer the question*; the knowledge graph
*organizes the library and lets you see and navigate it.*

### Contextual retrieval & vector library

- **Indexing** (`pipeline.py`): each page is fetched and cleaned; one LLM call
  reads the whole document and writes a **contextual summary per section** (so
  each summary is situated in the full document). Sections are split into chunks,
  and every chunk is embedded with its `title · date · section context`
  prepended. Each record carries a clear retrieval index: **date, source URL,
  title, and contextual summary**.
- **Storage** (`vectorstore.py`): a two-layer **HNSW** index (via `hnswlib`, with
  a numpy cosine fallback) — a coarse *section-summary* layer and a fine *chunk*
  layer — persisted under `data/vectors/`.
- **Retrieval** (`rag.py`) is **hierarchical**: rank section summaries first, then
  drill into chunks, scoring each chunk by a blend of its own similarity and its
  parent section's. A **re-ranker** (on by default) over-fetches ≈4×k candidates
  and has an LLM re-order them listwise for precision. For multi-document
  questions, **document-aware MMR** (`--mmr` / `"mmr": true`) instead selects the
  top-k to spread across distinct documents (relevance − redundancy, with a
  same-document penalty), lifting multi-doc recall.

### Memory layer: keeping the library dynamic

The graph and vector index grow when you *ingest documents*. The **memory layer**
(`memory.py`, layer 6) adds the other half of "dynamic": the library learns from
its own use and remembers across sessions. It is a separate plain-JSON store
(`data/memory.json`), keeping the local-first design — semantic recall reuses the
same OpenAI embeddings as RAG (and falls back to keyword overlap when no
`OPENAI_API_KEY` is set, so it always works).

- **Recall (before).** Every agent `query`/`ingest`/`lint` pass and every RAG
  `ask` first recalls the most relevant memories and folds them into the prompt
  as background (clearly marked *not citable*, so RAG still cites real passages).
- **Write-back (after).** Answering files a durable `answer` memory; a Maintain
  pass records an `observation`; the agent can record `fact`/`gap` memories with
  its `memory_write` tool. Memories **reinforce with reuse** (`use_count`,
  `salience`) and **de-duplicate** on re-assertion — the same mechanics that make
  entities strengthen in the graph.
- **Feedback loop.** 👍/👎 ratings and written corrections feed back as memories;
  a correction **supersedes** the memory it fixes so stale knowledge drops out of
  recall.

Memory kinds: `fact`, `answer`, `preference`, `gap`, `correction`, `observation`.
Manage them in the **Memory** tab, the `memory-*` CLI modes, or the
`/api/memory/*` endpoints. Evaluation runs (`rag-eval`/experiments) disable
memory recall and write-back so deterministic metrics stay comparable.

## Using the web UI

### Read tab
- **Search** the web (DuckDuckGo) from the input bar.
- **Summarize a URL** — paste a URL and press *Summarize*; or paste **raw text**
  to summarize that instead. Every search result has a one-click *Summarize this*.
- **Save a chunk to KG** — paste text and click `+ KG`, or highlight any text in a
  summary to surface a floating *Save to KG* button. A modal adds tags, a note,
  and source metadata, landing the chunk in staging.

### Knowledge Graph tab
- **Ingest a corpus** in three modes: **Files** (drag-drop `.txt`/`.md`/`.html`/
  `.pdf`, up to 50 MB), **URLs** (one per line), or **Text** (paste a document).
  All share a **chunk size** (default 800 chars) and **overlap** (default 120);
  the chunker splits on paragraph then sentence boundaries and carries a
  word-aligned overlap, tagging chunks with their source and `part:i/N`.
- Browse **staging** (recently saved chunks) and the **integrated** graph.
- **Integrate →** moves staged chunks into the graph: the selected provider/model
  extracts entities + typed relations and links them (heuristic fallback with no
  key). Enable the agentic organize pass to also have the agent reshape the graph.
- **Ask your library** — the agent reads the catalog, searches the graph, follows
  relationships (multi-hop), reads the underlying evidence, and answers with
  **citations**; durable answers are filed as `wiki/query/*.md`.
- **Maintain ✨** — a whole-library pass: merge duplicate entities, build the
  **topic hierarchy**, add missing relations, and write **synthesis** pages.
- The graph is **layer-aware**: nodes are colored by layer with legend-chip
  filters; topics are diamonds, syntheses are stars, entity size reflects
  mentions. Click a node for its profile; **search** matches names, aliases,
  summaries, chunk text, tags, and source titles.

### Library Q&A tab
Ask questions grounded in the **contextual vector library** (separate from the
agent's KG). Toggle the **re-ranker** and **MMR (diversify)**, choose a provider/
model, and get an answer with cited passages, dates, and scores.

### Memory tab
Inspect and grow the library's [memory layer](#memory-layer-keeping-the-library-dynamic).
**Recall** memories by query; **add** a memory (pick a kind and salience); browse
**stored memories** filtered by kind, with 👍/👎 to reinforce or demote and a
*forget* button. The header shows how many memories are embedded and whether
recall is semantic (an `OPENAI_API_KEY` is set) or keyword-only.

## Command line (`runner.py`)

```bash
# Agentic knowledge-graph workspace (llm-wiki-style)
python runner.py --mode init
python runner.py --mode ingest --source notes/ada.md --url https://example.com/article
python runner.py --mode query --question "What did Ada contribute to computing?"
python runner.py --mode lint                       # whole-library maintenance pass

# Memory layer (cross-session learnings)
python runner.py --mode memory-add --text "User is researching agent memory." --kind preference
python runner.py --mode memory-recall --question "agent memory"   # what does it remember?
python runner.py --mode memory-list                # all memories + stats
python runner.py --mode memory-forget --id memory_xxxxxxxx

# Contextual RAG library
python runner.py --mode rag-ingest --source eval/corpus_urls.txt   # bundled 28-doc corpus
python runner.py --mode rag-ask --question "How do rubrics help agents self-correct?"
python runner.py --mode kg-extract                 # add entity + topic layers over the library

# Evaluation (see below)
python runner.py --mode rag-eval                   # local metrics; --no-rerank to A/B
python runner.py --mode rag-experiment --rerank    # LangSmith experiment
python runner.py --mode rag-dataset                # sync the eval template up to LangSmith
python runner.py --mode rag-ragas                  # RAGAS metrics (needs `pip install ragas`)
python runner.py --mode rag-crossdoc --mmr         # cross-document experiment, MMR retrieval
```

Common flags: `--provider auto|anthropic|openai|qwen|deepseek|gemini|mistral`,
`--model <id>`, `--rerank/--no-rerank` (default on), `--mmr`, `--chunk-size`,
`--overlap`, `--no-agent` (skip the LLM organize step), `--export` (rag-dataset).

> **Model choice matters for *Maintain* / `lint`.** It is the most open-ended,
> multi-step task. Capable models converge and write a full report; very small
> models may loop and hit the step budget (`KG_AGENT_RECURSION_LIMIT`, default
> 150). That is handled gracefully — work done so far is saved and you can re-run
> to continue — but prefer a capable model for one-shot results.

## Evaluation

Everything is grounded in the bundled, version-controlled corpus
(`eval/corpus_urls.txt`, 28 LangChain + Anthropic articles) and reusable dataset
templates committed under `eval/`.

- **Local eval** (`rag.py`, `rag-eval`): generates a `(question, expected source)`
  set from the ingested docs and scores **retrieval hit-rate@k**, **MRR**, and an
  **LLM-judge answer score**.
- **LangSmith experiments** (`rag_experiment.py`, `rag-experiment`): runs
  `langsmith.evaluate` with the RAG pipeline as the target and three evaluators —
  **retrieval_hit**, **reciprocal_rank**, and LLM-judged **answer_correctness** —
  producing a comparable experiment in the LangSmith UI (diff base vs. re-ranked
  over identical inputs). The dataset is referenced by **ID** (rename-proof) via
  `LANGSMITH_RAG_DATASET_ID`; `rag-dataset` syncs the committed template
  (`eval/rag_eval_dataset.json`) up, and `--export` pulls it back.
- **RAGAS** (`ragas_eval.py`, `rag-ragas`): **faithfulness**, **answer relevancy**,
  and **context precision** (reference-free) wired as LangSmith evaluators over
  the same dataset.
- **Cross-document** (`crossdoc.py`, `rag-crossdoc`): a separate dataset
  (`eval/rag_eval_dataset_crossdoc.json`) whose questions each **require
  synthesizing across 2-3 documents** (`outputs.expected_urls` lists the required
  sources), scored on **retrieval_recall** / **retrieval_any_hit**, RAGAS, and an
  LLM-judged **synthesis correctness**. The synthesis judge is graded against
  **evidence, not its own priors**: each judge sees (a) excerpts of the gold
  source documents and (b) a gold **`key_points`** reference for the question, so
  "well-grounded" becomes checkable instead of a guess from the title/URL alone.

### Human-in-the-loop calibration

An LLM judge with no reference silently grades on its own world-knowledge and
style, so the same answer can swing ~0.4 across judge families. The cross-doc eval
closes the loop with **human labels** (`eval/crossdoc_human_labels.json`): each
entry holds the gold `key_points` (the reference the judge grades against), a
human `human_score` for a `reviewed_answer`, and notes. The run reports
**`judge_alignment`** — how closely each LLM judge tracks the human (`alignment =
1 − MAE`, plus `within_0.2`), over *fresh* pairs only; an answer that no longer
matches its `reviewed_answer` is flagged **stale → re-review**. This surfaces
which judges to trust and turns reviewer corrections into reusable references —
the human-judgment step of the agent-improvement loop.

```bash
python runner.py --mode rag-crossdoc-labels   # draft key_points; human fills human_score
python runner.py --mode rag-crossdoc          # runs the eval + prints judge_alignment
```

### Judge ≠ generator family

LLM-judged metrics never use the generator's own family (self-preference bias).
The single-judge path (`providers.resolve_judge`) picks a different-family model;
the cross-document eval goes further with a **judge panel** of five distinct
families (`providers.judge_panel`):

`gpt-5.2` · `qwen3.7-max` · `deepseekV3-chat` · `gemini-3.5-flash` ·
`mistral-large-latest`

Each scores its own `correctness_<model>` column plus a panel mean, averaging out
any single model's strictness. Only configured providers join the panel, so a
family without an API key (or a quota-limited one) is skipped automatically.
Absolute LLM-judge scores are judge-dependent, so the **deterministic** retrieval
metrics are the most comparable across runs.

### Results we measured (28-doc corpus)

| Setting | Metric | Result |
| --- | --- | --- |
| Single-doc, re-ranked | hit@6 / MRR | ≈ 0.87 / 0.81 |
| Re-ranker on vs. off | RAGAS context precision | ≈ 0.51 → 0.67 (+31%) |
| Cross-doc, base → MMR | retrieval recall | 0.59 → 0.91 |
| Cross-doc, MMR vs. graph-RAG | retrieval recall | **0.909** vs. 0.788 |
| Cross-doc, MMR vs. graph-RAG | panel correctness (common judges) | **0.667** vs. 0.598 |

MMR winning on both recall and judged synthesis is exactly why entity-graph
retrieval was removed.

## Tracing with LangSmith

Because the agent and the RAG pipeline run on LangChain, tracing is automatic
once the env vars are set — no code changes needed. Runs get readable names
(`ingest · …`, `query · …`, `maintain · …`) and are tagged `knowledge-library`.
`LANGSMITH_PROJECT_ID` is resolved to the project's current name at runtime
(`config.ensure_tracing_project`), so renaming the project in LangSmith does not
break tracing. The agent-status line in the UI (and `GET /api/agent/status`)
shows the active project when enabled.

## HTTP API

**Read / providers**
- `GET /api/providers` — configured providers, suggested models, and whether the
  agent stack is installed (`agent_available`).
- `POST /api/search` — `{ "query" }` → DuckDuckGo results.
- `POST /api/summarize` — `{ "input": "<url-or-text>", "provider?", "model?" }` →
  page/text summary (`provider` may be `auto` … `mistral`, or `extractive`).

**Knowledge graph**
- `GET /api/kg/stats` — per-store counts for `current` and `overall`.
- `GET /api/kg/graph?where=current|overall[&granularity=document|section|chunk]` —
  nodes + edges (with `layer`); `granularity` returns a zoom-level subgraph.
- `POST /api/kg/add` — `{ "text", "source_title?", "source_url?", "tags?", "note?" }`
  adds a chunk to staging.
- `POST /api/kg/integrate` — `{ "provider?", "model?", "use_ai?", "use_agent?" }`
  moves staged chunks into the graph; `use_agent` also runs the organize pass.
- `POST /api/kg/query` — `{ "query", "where?" }` returns matching nodes.
- `DELETE /api/kg/node/<id>?where=current|overall` — remove a node.
- `POST /api/kg/ingest/text` — chunk a pasted document into staging
  (`{ "text", "source_title?", "source_url?", "tags?", "chunk_size?", "overlap?" }`).
- `POST /api/kg/ingest/urls` — fetch, parse, and chunk each URL into staging.
- `POST /api/kg/ingest/files` — multipart upload of `.txt`/`.md`/`.html`/`.pdf`.

**Agent**
- `GET /api/agent/status` — whether the agent is available and which provider is ready.
- `POST /api/agent/ask` — `{ "question", "provider?", "model?", "file_answer?" }`
  → grounded answer + citations.
- `POST /api/agent/maintain` — `{ "provider?", "model?" }` runs the maintenance pass.

**Memory layer**
- `GET /api/memory/stats` — totals, counts by kind, how many are embedded.
- `GET /api/memory/list?kind=&limit=` — stored memories (most salient first).
- `POST /api/memory/recall` — `{ "query", "k?", "kinds?" }` → most relevant memories.
- `POST /api/memory/add` — `{ "text", "kind?", "salience?", "tags?" }` stores a memory.
- `POST /api/memory/feedback` — `{ "memory_id?", "rating?", "correction?", "question?" }`
  reinforces/demotes a memory or files a superseding correction.
- `DELETE /api/memory/<id>` — forget a memory.

**RAG library**
- `GET /api/rag/stats` — vector library stats (documents, sections, chunks, index).
- `POST /api/rag/ingest` — `{ "urls": [...], "provider?", "model?" }` builds the
  contextual vector index for the given pages.
- `POST /api/rag/search` — `{ "query", "k?", "rerank?", "mmr?" }` hierarchical
  retrieval only (no LLM); returns ranked passages with scores.
- `POST /api/rag/ask` — `{ "question", "provider?", "model?", "k?", "rerank?", "mmr?" }`
  → grounded answer with citations.
- `POST /api/rag/eval` — `{ "max_questions?", "k?", "rerank?", "mmr?", "provider?", "model?", "judge_provider?", "judge_model?" }`
  → retrieval + answer-quality metrics.
- `POST /api/rag/experiment` — `{ "rerank?", "k?", "provider?", "model?" }` runs a
  LangSmith evaluation experiment over the dataset (resolved by ID).
- `POST /api/rag/dataset` — sync the committed eval template to LangSmith
  (`{ "export": true }` writes the template from the dataset instead).
- `POST /api/rag/ragas` — `{ "rerank?", "k?", "provider?", "model?" }` runs RAGAS
  metrics as a LangSmith experiment.
- `POST /api/rag/crossdoc` — `{ "rerank?", "mmr?", "k?", "ragas?", "provider?", "model?", "judge_provider?", "judge_model?" }`
  builds/syncs the cross-document dataset and runs a multi-source experiment
  (returns `judge_alignment` when human labels are present).
- `POST /api/rag/crossdoc/labels` — `{ "provider?", "model?", "overwrite?" }`
  drafts `key_points` into `eval/crossdoc_human_labels.json` for a human to score.

## Configuration

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Claude chat / summaries / agent. |
| `OPENAI_API_KEY` | OpenAI chat **and embeddings** (required for RAG). |
| `DASHSCOPE_API_KEY` | Qwen (DashScope). |
| `DEEPSEEK_API_KEY` | DeepSeek. |
| `GEMINI_API_KEY` | Google Gemini (OpenAI-compatible endpoint). |
| `MISTRAL_API_KEY` | Mistral. |
| `OPENAI_BASE_URL` / `QWEN_BASE_URL` / `DEEPSEEK_BASE_URL` / `GEMINI_BASE_URL` / `MISTRAL_BASE_URL` | Override an OpenAI-compatible base URL (proxy, Azure, regional endpoint). |
| `LANGSMITH_TRACING` | `true` to trace agent + RAG runs. |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint. |
| `LANGSMITH_API_KEY` | LangSmith key. |
| `LANGSMITH_PROJECT_ID` | Project referenced by **ID** (rename-proof); resolved to its name at runtime. |
| `LANGSMITH_PROJECT` | Project **name** fallback if no ID is set. |
| `LANGSMITH_RAG_DATASET_ID` | Single-doc eval dataset ID. |
| `LANGSMITH_CROSSDOC_DATASET_ID` | Cross-document eval dataset ID. |
| `KG_DATA_DIR` | Directory for the graph + vector + memory stores (default `data/`). |
| `KG_AGENT_RECURSION_LIMIT` | Max agent steps for Maintain / `lint` (default 150). |
| `KG_MEMORY_DEDUP_THRESHOLD` | Cosine ≥ this reinforces an existing memory instead of adding a new one (default `0.92`). |
| `KG_MEMORY_RECALL_FLOOR` | Min semantic similarity for a memory to be recalled (default `0.20`; keyword path uses `KG_MEMORY_RECALL_FLOOR_KEYWORD`, `0.05`). |
| `PORT` | Port to bind (default `5000`). |

`.env` is gitignored; never commit real keys. See `.env.example` for the full
template.

## Project layout

```
app.py             Flask app: web UI + JSON API
runner.py          CLI: init/ingest/query/lint + rag-* + kg-extract
agent.py           deepagents harness (local filesystem backend)
kg_tools.py        agent tools over the knowledge graph
memory.py          cross-session memory store (recall + write-back, layer 6)
memory_tools.py    agent tools over the memory layer (recall / write)
knowledge_graph.py multi-layer graph store + queries (granularity, dedup, map)
ingestion.py       chunking + fetch/parse for the KG staging area
extraction.py      entity + relation extraction
enrich.py          entity/topic layers over an ingested RAG library
pipeline.py        contextual-retrieval ingest (fetch → sections → chunks)
embeddings.py      OpenAI embedding helpers
vectorstore.py     two-layer HNSW index + hierarchical search + MMR
rag.py             retrieval, re-ranker, grounded answers, local eval
rag_experiment.py  LangSmith dataset + single-doc experiment
ragas_eval.py      RAGAS metrics as LangSmith evaluators
crossdoc.py        cross-document dataset + experiment + judge panel
providers.py       provider table, chat-model factory, judge selection
config.py          .env loading + LangSmith project resolution
eval/              corpus_urls.txt + committed dataset templates
static/ templates/ web UI assets (incl. the Memory tab)
data/              graph JSON, vectors, memory.json, and the agent's wiki workspace
```

## Credits

Inspired by the LangChain *llm-wiki* deep-agents example. Built with Flask,
`deepagents`/LangChain, `hnswlib`, OpenAI embeddings, RAGAS, and LangSmith.
