# Build Your Own Knowledge Library Agent
<img width="1408" height="768" alt="Knowledge Library" src="https://github.com/user-attachments/assets/f63e43bb-b635-4ecc-8ddf-78102f99975c" />

Build **your own knowledge library** from the web: search and summarize pages,
ingest them into a contextual vector index for grounded, cited Q&A, and grow a
layered knowledge graph an LLM agent keeps coherent. One Flask app — web UI, JSON
API, and a `runner.py` CLI. Inspired by the LangChain *llm-wiki* deep-agents
example, but it builds a personal, on-disk library instead of syncing to a hub.

Three cooperating parts:

- **Agentic knowledge graph** — save passages, extract entities + typed relations,
  and let a [`deepagents`](https://pypi.org/project/deepagents/) agent (local
  filesystem backend, no cloud sandbox) de-duplicate entities, build a topic
  hierarchy, and write synthesis pages.
- **Contextual RAG library** — a two-layer HNSW index built with Anthropic-style
  *contextual retrieval*; answers are grounded with hierarchical retrieval plus an
  LLM re-ranker or document-aware MMR.
- **Memory layer** — a cross-session store the library recalls *before* and writes
  back *after* every answer, so it improves from use, not just from ingestion.

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

Grouped under `/api`: **read** (`/search`, `/summarize`, `/providers`),
**knowledge graph** (`/kg/stats`, `/kg/graph`, `/kg/add`, `/kg/integrate`,
`/kg/query`, `/kg/ingest/{text,urls,files}`), **agent** (`/agent/{status,ask,
maintain}`), **memory** (`/memory/{stats,list,recall,add,feedback}`), and **RAG**
(`/rag/{stats,ingest,search,ask,eval,experiment,dataset,ragas,crossdoc,
crossdoc/labels}`). Most accept `{provider?, model?}`; cross-doc returns
`judge_alignment` when human labels exist.

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
| `PORT` | bind port (default 5000). |

`.env` is gitignored — never commit real keys. See `.env.example` for the full list.

## Project layout

```
app.py / runner.py      Flask app (UI + API) / CLI
agent.py kg_tools.py    deepagents harness + graph tools
memory.py memory_tools.py   cross-session memory (layer 6)
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
