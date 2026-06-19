import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

import config
import knowledge_graph as kg
import ingestion
import extraction
from providers import (
    PROVIDERS,
    agent_dependencies_available,
    first_available_provider,
    provider_configured,
    resolve_provider_model,
)

# Load a local .env (LangSmith + model keys) if present, before reading env.
config.load_env()
# Resolve LANGSMITH_PROJECT_ID -> current project name for tracing (best-effort).
config.ensure_tracing_project()

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
MAX_CHARS_FOR_MODEL = 16000


def is_valid_url(text: str) -> bool:
    try:
        parsed = urlparse(text.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def fetch_page(url: str) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else url)

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return {"title": title, "url": url, "text": text}


def extractive_summary(text: str, max_sentences: int = 6) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    if not sentences:
        return text[:600]

    word_counts: dict[str, int] = {}
    for s in sentences:
        for w in re.findall(r"[A-Za-z]{4,}", s.lower()):
            word_counts[w] = word_counts.get(w, 0) + 1

    stopwords = {
        "this", "that", "with", "from", "have", "they", "their", "there", "which",
        "would", "could", "should", "about", "these", "those", "been", "were", "what",
        "when", "where", "your", "will", "more", "than", "into", "also", "such",
        "some", "many", "most", "other", "even", "only", "very", "much", "like",
    }
    scores = []
    for s in sentences:
        words = re.findall(r"[A-Za-z]{4,}", s.lower())
        score = sum(word_counts.get(w, 0) for w in words if w not in stopwords)
        scores.append((score / max(len(words), 1), s))

    ranked = sorted(enumerate(scores), key=lambda x: x[1][0], reverse=True)[:max_sentences]
    ranked.sort(key=lambda x: x[0])
    return " ".join(s for _, (_, s) in ranked)


def build_prompt(title: str, url: str, text: str) -> str:
    snippet = text[:MAX_CHARS_FOR_MODEL]
    return (
        f"Summarize the following web page in clear, concise prose. "
        f"Start with a one-sentence TL;DR, then 3-6 bullet points of key takeaways.\n\n"
        f"Title: {title}\nURL: {url}\n\nContent:\n{snippet}"
    )


def anthropic_summary(model: str, title: str, url: str, text: str) -> str:
    if Anthropic is None:
        raise RuntimeError("anthropic package is not installed.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(title, url, text)}],
    )
    return "".join(block.text for block in msg.content if hasattr(block, "text"))


def openai_compatible_summary(provider: str, model: str, title: str, url: str, text: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"{config['env_key']} is not set.")
    base_url = os.environ.get(config.get("base_url_env", ""), config.get("base_url"))
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(title, url, text)}],
    )
    return (resp.choices[0].message.content or "").strip()


def generate_ai_summary(provider: str, model: str, title: str, url: str, text: str) -> str:
    if provider == "anthropic":
        return anthropic_summary(model, title, url, text)
    if provider in ("openai", "qwen", "deepseek"):
        return openai_compatible_summary(provider, model, title, url, text)
    raise ValueError(f"Unknown provider: {provider}")


def web_search(query: str, max_results: int = 8) -> list[dict]:
    if DDGS is None:
        return []
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body", ""),
            })
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/providers")
def api_providers():
    payload = {}
    for name, cfg in PROVIDERS.items():
        payload[name] = {
            "label": cfg["label"],
            "default_model": cfg["default_model"],
            "models": cfg["models"],
            "configured": provider_configured(name),
            "env_key": cfg["env_key"],
        }
    return jsonify({
        "providers": payload,
        "auto": first_available_provider(),
        "agent_available": agent_dependencies_available(),
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is required."}), 400
    try:
        results = web_search(query)
        return jsonify({"query": query, "results": results})
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json(silent=True) or {}
    target = (data.get("input") or "").strip()
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()

    if not target:
        return jsonify({"error": "Provide a URL or paste text to summarize."}), 400

    try:
        if is_valid_url(target):
            page = fetch_page(target)
            title, url, text = page["title"], page["url"], page["text"]
        else:
            title, url, text = "Pasted text", "", target

        if len(text) < 100:
            return jsonify({"error": "Not enough content to summarize."}), 400

        if provider == "extractive":
            return jsonify({
                "title": title, "url": url,
                "summary": extractive_summary(text),
                "engine": "extractive", "provider": "extractive", "model": "",
                "chars": len(text),
            })

        if provider == "auto":
            provider = first_available_provider() or ""

        if provider in PROVIDERS:
            chosen_model = model or PROVIDERS[provider]["default_model"]
            try:
                summary = generate_ai_summary(provider, chosen_model, title, url, text)
                return jsonify({
                    "title": title, "url": url,
                    "summary": summary,
                    "engine": "ai", "provider": provider, "model": chosen_model,
                    "chars": len(text),
                })
            except RuntimeError as e:
                return jsonify({"error": str(e)}), 400

        return jsonify({
            "title": title, "url": url,
            "summary": extractive_summary(text),
            "engine": "extractive", "provider": "extractive", "model": "",
            "chars": len(text),
        })
    except requests.HTTPError as e:
        return jsonify({"error": f"Could not fetch page: HTTP {e.response.status_code}"}), 502
    except requests.RequestException as e:
        return jsonify({"error": f"Could not fetch page: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {e}"}), 500


@app.route("/api/kg/stats")
def api_kg_stats():
    return jsonify(kg.stats())


@app.route("/api/kg/graph")
def api_kg_graph():
    where = request.args.get("where", "current")
    granularity = request.args.get("granularity")
    if granularity:
        return jsonify(kg.granularity_view(where, granularity))
    return jsonify(kg.get_graph(where))


@app.route("/api/kg/add", methods=["POST"])
def api_kg_add():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text) < 5:
        return jsonify({"error": "Text is required."}), 400
    chunk = kg.add_chunk(
        text,
        source_url=(data.get("source_url") or "").strip(),
        source_title=(data.get("source_title") or "").strip(),
        tags=data.get("tags") if isinstance(data.get("tags"), list) else None,
        note=(data.get("note") or "").strip(),
    )
    return jsonify({"chunk": chunk, "stats": kg.stats()})


@app.route("/api/kg/integrate", methods=["POST"])
def api_kg_integrate():
    data = request.get_json(silent=True) or {}
    use_ai = bool(data.get("use_ai", True))
    use_agent = bool(data.get("use_agent", False))
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()

    extract_fn = None
    used_provider = "heuristic"
    used_model = ""
    if use_ai:
        actual, actual_model = resolve_provider_model(provider, model)
        if actual:
            used_provider, used_model = actual, actual_model
            extract_fn = lambda t: extraction.extract_kg_llm(t, actual, actual_model)  # noqa: E731

    result = kg.integrate(extract_fn=extract_fn)
    result["provider_used"] = used_provider
    result["model_used"] = used_model

    # Optional agentic organize pass: dedupe, build topics, summarize, synthesize.
    if use_agent:
        try:
            import agent
            organize = agent.run_ingest(provider=provider, model=model)
            result["agent_report"] = organize.get("report")
            result["agent_used"] = f"{organize.get('provider')}/{organize.get('model')}"
        except Exception as e:  # noqa: BLE001
            result["agent_error"] = str(e)

    result["stats"] = kg.stats()
    return jsonify(result)


@app.route("/api/kg/node/<node_id>", methods=["DELETE"])
def api_kg_delete(node_id):
    where = request.args.get("where", "current")
    removed = kg.remove_node(node_id, where=where)
    return jsonify({"removed": removed, "stats": kg.stats()})


@app.route("/api/kg/query", methods=["POST"])
def api_kg_query():
    data = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    where = data.get("where", "overall")
    nodes = kg.query(q, where=where)
    return jsonify({"results": nodes, "query": q, "where": where})


# --- Agent layer endpoints ---------------------------------------------------
def _agent_or_error():
    """Import the agent module, returning (module, None) or (None, error_json)."""
    if not agent_dependencies_available():
        return None, (jsonify({
            "error": "The agent layer requires deepagents. Install with "
                     "`pip install deepagents langchain-anthropic langchain-openai`."
        }), 503)
    try:
        import agent
        return agent, None
    except Exception as e:  # noqa: BLE001
        return None, (jsonify({"error": f"Agent unavailable: {e}"}), 503)


@app.route("/api/agent/status")
def api_agent_status():
    return jsonify({
        "agent_available": agent_dependencies_available(),
        "provider_ready": first_available_provider(),
        "configured": {name: provider_configured(name) for name in PROVIDERS},
        "tracing": config.tracing_status(),
    })


@app.route("/api/agent/ask", methods=["POST"])
def api_agent_ask():
    mod, err = _agent_or_error()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "A question is required."}), 400
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    file_answer = bool(data.get("file_answer", True))
    try:
        res = mod.run_query(question, provider=provider, model=model, file_answer=file_answer)
        return jsonify(res)
    except mod.AgentError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Query failed: {e}"}), 500


@app.route("/api/agent/maintain", methods=["POST"])
def api_agent_maintain():
    mod, err = _agent_or_error()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    try:
        res = mod.run_lint(provider=provider, model=model)
        res["stats"] = kg.stats()
        return jsonify(res)
    except mod.AgentError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Maintenance failed: {e}"}), 500


# --- RAG / contextual vector library endpoints ------------------------------
@app.route("/api/rag/stats")
def api_rag_stats():
    try:
        from vectorstore import VectorStore
        return jsonify(VectorStore.load("library").stats())
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/ingest", methods=["POST"])
def api_rag_ingest():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls") or []
    if isinstance(urls, str):
        urls = [u.strip() for u in urls.splitlines() if u.strip()]
    urls = [u for u in urls if is_valid_url(u)]
    if not urls:
        return jsonify({"error": "Provide one or more valid URLs."}), 400
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    try:
        import pipeline
        return jsonify(pipeline.ingest_urls(urls, provider=provider, model=model))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/search", methods=["POST"])
def api_rag_search():
    """Hierarchical retrieval only (no LLM) — inspect what the retriever returns."""
    data = request.get_json(silent=True) or {}
    q = (data.get("query") or data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "A query is required."}), 400
    k = int(data.get("k") or 8)
    graph_rag = bool(data.get("graph_rag", True))
    rerank = bool(data.get("rerank", True))
    mmr = bool(data.get("mmr", False))
    try:
        import rag
        hits = rag.retrieve(q, k=k, graph_rag=graph_rag, rerank_hits=rerank, mmr=mmr)
        return jsonify({"query": q, "reranked": rerank, "mmr": mmr, "hits": hits})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/ask", methods=["POST"])
def api_rag_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "A question is required."}), 400
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    k = int(data.get("k") or 6)
    graph_rag = bool(data.get("graph_rag", True))
    rerank = bool(data.get("rerank", True))
    mmr = bool(data.get("mmr", False))
    try:
        import rag
        return jsonify(rag.answer(question, provider=provider, model=model,
                                  k=k, graph_rag=graph_rag, rerank_hits=rerank, mmr=mmr))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/eval", methods=["POST"])
def api_rag_eval():
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    k = int(data.get("k") or 6)
    max_q = int(data.get("max_questions") or 10)
    graph_rag = bool(data.get("graph_rag", True))
    rerank = bool(data.get("rerank", True))
    mmr = bool(data.get("mmr", False))
    judge_provider = (data.get("judge_provider") or "").strip().lower() or None
    judge_model = (data.get("judge_model") or "").strip() or None
    try:
        import rag
        eval_set = data.get("eval_set") or rag.build_eval_set(
            provider=provider, model=model, max_questions=max_q)
        report = rag.evaluate(eval_set, provider=provider, model=model,
                              judge_provider=judge_provider, judge_model=judge_model,
                              k=k, graph_rag=graph_rag, rerank_hits=rerank, mmr=mmr)
        return jsonify(report)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/crossdoc", methods=["POST"])
def api_rag_crossdoc():
    """Build/sync the cross-document eval dataset and run a multi-source experiment."""
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    rerank = bool(data.get("rerank", True))
    mmr = bool(data.get("mmr", False))
    k = int(data.get("k") or 8)
    ragas = bool(data.get("ragas", True))
    judge_provider = (data.get("judge_provider") or "").strip().lower() or None
    judge_model = (data.get("judge_model") or "").strip() or None
    try:
        import crossdoc
        return jsonify(crossdoc.run_experiment(provider=provider, model=model,
                                               judge_provider=judge_provider, judge_model=judge_model,
                                               rerank=rerank, mmr=mmr, k=k, ragas=ragas))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/ragas", methods=["POST"])
def api_rag_ragas():
    """Run a RAGAS evaluation (faithfulness/answer-relevancy/context-precision) as a
    LangSmith experiment over the eval dataset."""
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    rerank = bool(data.get("rerank", True))
    k = int(data.get("k") or 6)
    judge_provider = (data.get("judge_provider") or "").strip().lower() or None
    judge_model = (data.get("judge_model") or "").strip() or None
    try:
        import ragas_eval
        return jsonify(ragas_eval.run_ragas_experiment(
            provider=provider, model=model, judge_provider=judge_provider,
            judge_model=judge_model, rerank=rerank, k=k))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/dataset", methods=["POST"])
def api_rag_dataset():
    """Sync the committed eval template to LangSmith (or ?export=1 to pull it down)."""
    data = request.get_json(silent=True) or {}
    try:
        import rag_experiment
        if data.get("export"):
            return jsonify(rag_experiment.export_dataset())
        return jsonify(rag_experiment.sync_dataset())
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/experiment", methods=["POST"])
def api_rag_experiment():
    """Create/refresh a LangSmith dataset and run an evaluation experiment."""
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    rerank = bool(data.get("rerank", True))
    k = int(data.get("k") or 6)
    max_q = int(data.get("max_questions") or 15)
    judge_provider = (data.get("judge_provider") or "").strip().lower() or None
    judge_model = (data.get("judge_model") or "").strip() or None
    try:
        import rag_experiment
        return jsonify(rag_experiment.run_experiment(
            provider=provider, model=model, judge_provider=judge_provider,
            judge_model=judge_model, rerank=rerank, k=k, max_questions=max_q))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


def _ingest_chunks(text, *, source_title, source_url, tags, chunk_size, overlap):
    chunks = ingestion.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    total = len(chunks)
    if total == 0:
        return 0
    for i, c in enumerate(chunks, start=1):
        title = source_title if total == 1 else f"{source_title} [{i}/{total}]"
        chunk_tags = list(tags) + ([f"part:{i}/{total}"] if total > 1 else [])
        kg.add_chunk(c, source_url=source_url, source_title=title, tags=chunk_tags)
    return total


def _parse_options(data):
    try:
        chunk_size = int(data.get("chunk_size") or 800)
    except (TypeError, ValueError):
        chunk_size = 800
    try:
        overlap = int(data.get("overlap") or 120)
    except (TypeError, ValueError):
        overlap = 120
    chunk_size = max(200, min(chunk_size, 4000))
    overlap = max(0, min(overlap, chunk_size // 2))
    tags_in = data.get("tags") or []
    if isinstance(tags_in, str):
        tags = [t.strip() for t in tags_in.split(",") if t.strip()]
    elif isinstance(tags_in, list):
        tags = [str(t).strip() for t in tags_in if str(t).strip()]
    else:
        tags = []
    return chunk_size, overlap, tags


@app.route("/api/kg/ingest/text", methods=["POST"])
def api_kg_ingest_text():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text) < 50:
        return jsonify({"error": "Provide at least 50 characters of text."}), 400
    chunk_size, overlap, tags = _parse_options(data)
    source_title = (data.get("source_title") or "Pasted document").strip()
    source_url = (data.get("source_url") or "").strip()
    total = _ingest_chunks(
        text, source_title=source_title, source_url=source_url,
        tags=tags, chunk_size=chunk_size, overlap=overlap,
    )
    return jsonify({
        "ok": True,
        "source": source_title,
        "chunks_created": total,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "stats": kg.stats(),
    })


@app.route("/api/kg/ingest/urls", methods=["POST"])
def api_kg_ingest_urls():
    data = request.get_json(silent=True) or {}
    raw_urls = data.get("urls") or []
    if isinstance(raw_urls, str):
        raw_urls = [u.strip() for u in raw_urls.splitlines() if u.strip()]
    urls = [u.strip() for u in raw_urls if u and u.strip()]
    if not urls:
        return jsonify({"error": "Provide at least one URL."}), 400
    chunk_size, overlap, tags = _parse_options(data)

    results = []
    total_chunks = 0
    for url in urls:
        if not is_valid_url(url):
            results.append({"url": url, "error": "Invalid URL", "chunks": 0})
            continue
        try:
            page = fetch_page(url)
            n = _ingest_chunks(
                page["text"], source_title=page["title"], source_url=url,
                tags=tags, chunk_size=chunk_size, overlap=overlap,
            )
            results.append({"url": url, "title": page["title"], "chunks": n})
            total_chunks += n
        except Exception as e:
            results.append({"url": url, "error": str(e), "chunks": 0})
    return jsonify({
        "ok": True,
        "total_chunks": total_chunks,
        "results": results,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "stats": kg.stats(),
    })


@app.route("/api/kg/ingest/files", methods=["POST"])
def api_kg_ingest_files():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400
    chunk_size, overlap, tags = _parse_options({
        "chunk_size": request.form.get("chunk_size"),
        "overlap": request.form.get("overlap"),
        "tags": request.form.get("tags") or "",
    })

    results = []
    total_chunks = 0
    for f in files:
        name = f.filename or "untitled"
        try:
            content = f.read()
            text = ingestion.parse_file(name, content)
            n = _ingest_chunks(
                text, source_title=name, source_url="",
                tags=tags, chunk_size=chunk_size, overlap=overlap,
            )
            results.append({"filename": name, "chunks": n, "chars": len(text)})
            total_chunks += n
        except Exception as e:
            results.append({"filename": name, "error": str(e), "chunks": 0})
    return jsonify({
        "ok": True,
        "total_chunks": total_chunks,
        "results": results,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "stats": kg.stats(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
