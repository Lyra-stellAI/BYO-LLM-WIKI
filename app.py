import os
import re
from io import BytesIO
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, url_for

import config
import knowledge_graph as kg
import cached_store
import ingestion
import extraction
import memory
import skill_library as skills
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

try:
    from langsmith import traceable
except ImportError:  # tracing is optional
    def traceable(*dargs, **dkw):  # type: ignore
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]
        def _wrap(fn):
            return fn
        return _wrap

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


def _looks_like_pdf(url: str, resp) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return ("application/pdf" in ctype
            or url.split("?")[0].split("#")[0].lower().endswith(".pdf")
            or resp.content[:5] == b"%PDF-")


def _pdf_title(content: bytes, url: str) -> str:
    """PDF metadata title if present, else the URL's file name."""
    try:
        import pypdf
        meta = pypdf.PdfReader(BytesIO(content)).metadata
        title = ((meta.title if meta else "") or "").strip()
        if title:
            return title
    except Exception:  # noqa: BLE001
        pass
    name = url.split("?")[0].split("#")[0].rstrip("/").rsplit("/", 1)[-1]
    return name or url


def fetch_page(url: str) -> dict:
    headers = {"User-Agent": USER_AGENT,
               "Accept": "text/html,application/xhtml+xml,application/pdf,*/*"}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    # PDFs (e.g. arXiv /pdf/ links) are not HTML — extract their text with pypdf
    # rather than feeding the binary to the HTML parser (which would return the
    # raw "%PDF-..." bytes as bogus "content").
    if _looks_like_pdf(url, resp):
        text = ingestion.parse_file("download.pdf", resp.content)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"title": _pdf_title(resp.content, url), "url": url, "text": text}

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
    client = config.traced_anthropic(Anthropic())
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(title, url, text)}],
    )
    return "".join(block.text for block in msg.content if hasattr(block, "text"))


def openai_compatible_summary(provider: str, model: str, title: str, url: str, text: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    pcfg = PROVIDERS[provider]
    api_key = os.environ.get(pcfg["env_key"])
    if not api_key:
        raise RuntimeError(f"{pcfg['env_key']} is not set.")
    base_url = os.environ.get(pcfg.get("base_url_env", ""), pcfg.get("base_url"))
    client = config.traced_openai(OpenAI(api_key=api_key, base_url=base_url))
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(title, url, text)}],
    )
    return (resp.choices[0].message.content or "").strip()


@traceable(name="summarize", tags=["summarize", "read-tab"])
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


_LOGO_NAMES = ("owl.png", "owl.webp", "owl.jpg", "owl.jpeg", "owl.svg",
               "logo.png", "logo.webp", "logo.jpg", "logo.svg")


def _find_logo():
    """URL of a custom brand image in static/ (owl.png, logo.png, …), or None.

    Drop your own logo into static/ and it replaces the built-in SVG owl with no
    code changes; until then the header falls back to the inline owl mark."""
    folder = app.static_folder or "static"
    for name in _LOGO_NAMES:
        if os.path.exists(os.path.join(folder, name)):
            return url_for("static", filename=name)
    return None


@app.route("/")
def index():
    return render_template("index.html", logo_url=_find_logo())


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


def extract_link_context(url: str) -> dict:
    """Fetch a link and return its extracted readable context.

    Shaped like a single search result (``title``/``url``/``snippet``) so the
    front-end can render it with the same card, plus the full ``context`` text
    and a ``chars`` count for downstream summarize / + KG actions.
    """
    page = fetch_page(url)
    text = page["text"]
    lead = text[:280].strip()
    if len(text) > len(lead):
        lead = lead.rsplit(" ", 1)[0] + "…"
    return {
        "title": page["title"],
        "url": page["url"],
        "snippet": lead,
        "context": text,
        "chars": len(text),
    }


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is required."}), 400

    # If the input is a link, fetch it and extract its context instead of running
    # a keyword web search (a URL makes a poor search term — it returns unrelated
    # hits rather than the content of the page the user actually pointed at).
    if is_valid_url(query):
        try:
            result = extract_link_context(query)
            return jsonify({"query": query, "kind": "link", "results": [result]})
        except requests.HTTPError as e:
            return jsonify({"error": f"Could not fetch link: HTTP {e.response.status_code}"}), 502
        except requests.RequestException as e:
            return jsonify({"error": f"Could not fetch link: {e}"}), 502
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"Could not extract context from link: {e}"}), 500

    try:
        results = web_search(query)
        return jsonify({"query": query, "kind": "web", "results": results})
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


# --- Cached content store ----------------------------------------------------
# The cache is the single source of truth for raw extracted content; KG nodes
# and vector records are derived projections (see cached_store.py). These routes
# are additive — the existing /api/search, /api/kg/* and /api/rag/* flows are
# unchanged.
def _qs_bool(value):
    if value is None:
        return None
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@app.route("/api/cache/extract", methods=["POST"])
def api_cache_extract():
    """Read tab step 1: extract MULTIPLE URLs at once. Fetches each page and
    returns previews + full text (so ingest needn't re-fetch). Commits nothing;
    flags items whose content is already cached."""
    data = request.get_json(silent=True) or {}
    raw = data.get("urls") or ""
    if isinstance(raw, str):
        urls = [u.strip() for u in re.split(r"[\n,]+", raw) if u.strip()]
    else:
        urls = [str(u).strip() for u in raw if str(u).strip()]
    if not urls:
        return jsonify({"error": "Provide at least one URL."}), 400

    store = cached_store.get_store()
    results, errors = [], []
    for url in urls:
        if not is_valid_url(url):
            errors.append({"url": url, "error": "Invalid URL"})
            continue
        try:
            ctx = extract_link_context(url)
            content_hash = cached_store._sha256(ctx["context"])
            # url-kind items dedupe on the URL (not the body hash), so check the
            # same id ingest() will use — matches even if the page text drifted.
            rid = cached_store.url_cache_id(ctx["url"])
            existing = store.get(rid)
            results.append({
                "url": ctx["url"], "title": ctx["title"], "snippet": ctx["snippet"],
                "text": ctx["context"], "chars": ctx["chars"],
                "content_hash": content_hash,
                "already_cached": existing is not None,
                "cache_id": rid if existing else None,
            })
        except requests.HTTPError as e:
            errors.append({"url": url, "error": f"HTTP {e.response.status_code}"})
        except requests.RequestException as e:
            errors.append({"url": url, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            errors.append({"url": url, "error": str(e)})
    return jsonify({"results": results, "errors": errors})


@app.route("/api/cache/ingest", methods=["POST"])
def api_cache_ingest():
    """Persist items into the cached store. Each item supplies already-extracted
    ``text`` (from /api/cache/extract — no re-fetch), or just a ``url`` (fetched
    once here), or pasted ``text`` with no url. Idempotent by content hash."""
    data = request.get_json(silent=True) or {}
    items = data.get("items")
    if isinstance(items, dict):
        items = [items]
    if not items:
        return jsonify({"error": "Provide one or more items to ingest."}), 400

    store = cached_store.get_store()
    out, errors, created, reused = [], [], 0, 0
    for it in items:
        url = (it.get("url") or "").strip()
        text = it.get("text")
        title = (it.get("source_title") or it.get("title") or "").strip()
        tags = it.get("tags") if isinstance(it.get("tags"), list) else None
        note = (it.get("note") or "").strip()
        kind = (it.get("kind") or ("url" if url else "text")).strip()
        try:
            if text is None and url:
                if not is_valid_url(url):
                    errors.append({"url": url, "error": "Invalid URL"})
                    continue
                page = fetch_page(url)
                text = page["text"]
                title = title or page["title"]
            text = text or ""
            if len((text or "").strip()) < 1:
                errors.append({"url": url or title, "error": "Empty content"})
                continue
            rec = cached_store.ingest(
                kind=kind, source_url=url, source_title=title, raw_text=text,
                tags=tags, note=note, origin=(it.get("origin") or "read_page"))
            if rec.get("created"):
                created += 1
            else:
                reused += 1
            out.append({
                "id": rec["id"], "source_title": rec["source_title"],
                "source_url": rec["source_url"], "chars": rec["chars"],
                "in_kg": rec["in_kg"], "vectorized": rec["vectorized"],
                "created": rec.get("created", False),
            })
        except requests.HTTPError as e:
            errors.append({"url": url, "error": f"HTTP {e.response.status_code}"})
        except requests.RequestException as e:
            errors.append({"url": url, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            errors.append({"url": url or title, "error": str(e)})
    return jsonify({"ok": True, "items": out, "created": created, "reused": reused,
                    "errors": errors, "store_stats": store.stats()})


@app.route("/api/cache/items")
def api_cache_items():
    """List cached items (no raw_text) for the KG/Q&A selectors and status badges.

    ``needs_kg`` / ``needs_vectors`` are drift-aware views: an item qualifies if
    it has NOT been projected yet OR its content changed since it was (so a
    re-ingested page reappears in the picker for re-projection)."""
    store = cached_store.get_store()
    try:
        limit = int(request.args.get("limit") or 0)
    except (TypeError, ValueError):
        limit = 0
    q = (request.args.get("q") or "").strip()
    needs_kg = _qs_bool(request.args.get("needs_kg"))
    needs_vectors = _qs_bool(request.args.get("needs_vectors"))
    if needs_kg or needs_vectors:
        # staleness is a cross-field condition; fetch without the plain flag filters.
        items = store.list(q=q)
    else:
        items = store.list(
            in_kg=_qs_bool(request.args.get("in_kg")),
            vectorized=_qs_bool(request.args.get("vectorized")), q=q)
    for it in items:
        it["kg_stale"] = cached_store.kg_stale(it)
        it["vector_stale"] = cached_store.vector_stale(it)
    if needs_kg:
        items = [it for it in items if (not it["in_kg"]) or it["kg_stale"]]
    if needs_vectors:
        items = [it for it in items if (not it["vectorized"]) or it["vector_stale"]]
    if limit and limit > 0:
        items = items[:limit]
    st = store.stats()
    return jsonify({"items": items,
                    "counts": {"total": st["items"], "in_kg": st["in_kg"],
                               "vectorized": st["vectorized"]},
                    "backend": st["backend"]})


@app.route("/api/cache/item/<item_id>", methods=["GET", "DELETE"])
def api_cache_item(item_id):
    store = cached_store.get_store()
    if request.method == "DELETE":
        # The cache is the source of truth, so deleting an item also removes the
        # KG chunk nodes and vector-store document it projected into.
        rec = store.get(item_id)
        cleanup = {"kg_chunks_removed": 0, "vectors_removed": 0}
        if rec is not None:
            for cid in rec.get("kg_chunk_ids") or []:
                for where in ("current", "overall"):
                    if kg.remove_node(cid, where=where):
                        cleanup["kg_chunks_removed"] += 1
            if rec.get("vectorized"):
                try:
                    from vectorstore import VectorStore
                    vs = VectorStore.load("library")
                    cleanup["vectors_removed"] = vs.remove_doc(cached_store.projection_url(rec))
                    vs.persist()
                except Exception:  # noqa: BLE001
                    pass
        removed = store.delete(item_id)
        return jsonify({"ok": removed, "cleanup": cleanup, "store_stats": store.stats()})
    rec = store.get(item_id, with_text=True)
    if rec is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(rec)


@app.route("/api/cache/to-kg", methods=["POST"])
def api_cache_to_kg():
    """KG tab step 2: project SELECTED cached items into the graph WITHOUT
    re-fetching. Stages chunks via the existing _ingest_chunks; integration
    stays on the unchanged /api/kg/integrate flow."""
    data = request.get_json(silent=True) or {}
    ids = data.get("item_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    if not ids:
        return jsonify({"error": "Select at least one cached item."}), 400
    chunk_size, overlap, tags = _parse_options(data)
    store = cached_store.get_store()
    staged, errors, total = [], [], 0
    for iid in ids:
        rec = store.get(iid, with_text=True)
        if rec is None:
            errors.append({"item_id": iid, "error": "Not found"})
            continue
        text = rec.get("raw_text") or ""
        if len(text.strip()) < 5:
            errors.append({"item_id": iid, "error": "No cached text"})
            continue
        # Idempotent re-projection: drop any chunk nodes a prior projection left
        # behind (in staging or already integrated) before re-staging, so a
        # second convert / a drifted re-convert replaces rather than duplicates.
        for old_id in rec.get("kg_chunk_ids") or []:
            kg.remove_node(old_id, where="current")
            kg.remove_node(old_id, where="overall")
        url = cached_store.projection_url(rec)
        item_tags = tags or rec.get("tags") or []
        n, chunk_ids = _ingest_chunks(
            text, source_title=rec["source_title"], source_url=url,
            tags=item_tags, chunk_size=chunk_size, overlap=overlap)
        store.set_flags(iid, in_kg=True, kg_chunk_ids=chunk_ids,
                        kg_projected_hash=rec.get("content_hash", ""))
        staged.append({"item_id": iid, "chunks": n})
        total += n
    return jsonify({"ok": True, "staged": staged, "errors": errors,
                    "total_chunks": total, "stats": kg.stats()})


@app.route("/api/cache/to-vectors", methods=["POST"])
def api_cache_to_vectors():
    """Q&A step 3: vectorize SELECTED cached items into the RAG library WITHOUT
    re-fetching. Response shape matches /api/rag/ingest so rag.js reuses it."""
    data = request.get_json(silent=True) or {}
    ids = data.get("item_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    if not ids:
        return jsonify({"error": "Select at least one cached item."}), 400
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    store = cached_store.get_store()
    records, missing = [], []
    for iid in ids:
        rec = store.get(iid, with_text=True)
        if rec is None or not (rec.get("raw_text") or "").strip():
            missing.append(iid)
            continue
        records.append(rec)
    if not records:
        return jsonify({"error": "No cached text found for the selected items."}), 400
    try:
        import pipeline
        result = pipeline.ingest_cached_items(records, provider=provider, model=model)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    hash_by_id = {rec["id"]: rec.get("content_hash", "") for rec in records}
    for r in result.get("results", []):
        cid = r.get("cache_id")
        if cid:
            store.set_flags(cid, vectorized=True,
                            vector_doc_id=pipeline._slug(r.get("url") or ""),
                            vector_projected_hash=hash_by_id.get(cid, ""))
    if missing:
        result.setdefault("errors", []).extend(
            {"cache_id": m, "error": "No cached text"} for m in missing)
    result["store_stats"] = store.stats()
    return jsonify(result)


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


# --- Memory layer endpoints (cross-session recall + write-back) -------------
@app.route("/api/memory/stats")
def api_memory_stats():
    return jsonify(memory.stats())


@app.route("/api/memory/list")
def api_memory_list():
    kind = (request.args.get("kind") or "").strip() or None
    try:
        limit = int(request.args.get("limit") or 100)
    except (TypeError, ValueError):
        limit = 100
    return jsonify({"memories": memory.list_memories(kind=kind, limit=limit),
                    "stats": memory.stats()})


@app.route("/api/memory/recall", methods=["POST"])
def api_memory_recall():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or data.get("question") or "").strip()
    if not query:
        return jsonify({"error": "A query is required."}), 400
    try:
        k = int(data.get("k") or 6)
    except (TypeError, ValueError):
        k = 6
    kinds_in = data.get("kinds")
    kinds = ([k.strip() for k in kinds_in.split(",") if k.strip()] if isinstance(kinds_in, str)
             else kinds_in if isinstance(kinds_in, list) else None)
    return jsonify({"query": query, "memories": memory.recall(query, k=k, kinds=kinds or None)})


@app.route("/api/memory/add", methods=["POST"])
def api_memory_add():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text) < 3:
        return jsonify({"error": "Memory text is required."}), 400
    kind = (data.get("kind") or "fact").strip().lower()
    try:
        salience = int(data.get("salience") or 3)
    except (TypeError, ValueError):
        salience = 3
    tags_in = data.get("tags")
    tags = ([t.strip() for t in tags_in.split(",") if t.strip()] if isinstance(tags_in, str)
            else [str(t).strip() for t in tags_in if str(t).strip()] if isinstance(tags_in, list)
            else None)
    rec = memory.remember(text, kind=kind, salience=salience, tags=tags,
                          confidence="USER", origin="user",
                          source_url=(data.get("source_url") or "").strip())
    return jsonify({"ok": bool(rec), "memory": rec, "stats": memory.stats()})


@app.route("/api/memory/feedback", methods=["POST"])
def api_memory_feedback():
    data = request.get_json(silent=True) or {}
    rec = memory.record_feedback(
        question=(data.get("question") or "").strip(),
        answer=(data.get("answer") or "").strip(),
        rating=(data.get("rating") or "").strip(),
        correction=(data.get("correction") or "").strip(),
        memory_id=(data.get("memory_id") or "").strip(),
    )
    return jsonify({"ok": bool(rec), "memory": rec, "stats": memory.stats()})


@app.route("/api/memory/<mem_id>", methods=["DELETE"])
def api_memory_delete(mem_id):
    return jsonify({"removed": memory.forget(mem_id), "stats": memory.stats()})


# --- Agent-skill layer endpoints (build / eval / human review / library) ----
@app.route("/api/skill/stats")
def api_skill_stats():
    return jsonify(skills.stats())


@app.route("/api/skill/list")
def api_skill_list():
    status = (request.args.get("status") or "").strip() or None
    try:
        limit = int(request.args.get("limit") or 100)
    except (TypeError, ValueError):
        limit = 100
    return jsonify({"skills": skills.list_skills(status=status, limit=limit),
                    "stats": skills.stats()})


@app.route("/api/skill/pending")
def api_skill_pending():
    """The human-review queue: skills that passed the gate and await alignment."""
    return jsonify({"skills": skills.pending_review(), "stats": skills.stats()})


@app.route("/api/skill/observability")
def api_skill_observability():
    """Benchmark readout over logged runs: gate pass-rate, avg latency/tokens, etc."""
    import skill_runs, skill_tracing
    skill_id = (request.args.get("skill_id") or "").strip() or None
    return jsonify({"benchmark": skill_runs.benchmark(skill_id=skill_id),
                    "recent": skill_runs.list_runs(skill_id=skill_id, limit=20),
                    "tracing": skill_tracing.status()})


@app.route("/api/skill/changes")
def api_skill_changes():
    """A cheap change token for polling: counts + store/run timestamps. Lets the UI
    refresh the review queue only when something actually changed."""
    import skill_runs
    st = skills.stats()
    return jsonify({"pending_review": st["pending_review"], "total": st["total"],
                    "accepted": st["accepted"], "store_updated_at": skills.store_updated_at(),
                    "last_run_at": skill_runs.last_run_at()})


@app.route("/api/skill/tracing/init", methods=["POST"])
def api_skill_tracing_init():
    """Create the dedicated LangSmith project for skill runs (idempotent)."""
    import skill_tracing
    return jsonify({"ensure": skill_tracing.ensure_project(), "status": skill_tracing.status()})


@app.route("/api/skill/backends")
def api_skill_backends():
    """Which skill-generation backends are available: the in-process pipeline and
    the Claude Code CLI subprocess agent."""
    import skill_agent, skill_claude_agent
    payload = {
        "default": skill_agent.DEFAULT_BACKEND,
        "backends": {
            "pipeline": {"name": "pipeline", "available": bool(first_available_provider()),
                         "label": "In-process pipeline (LLM phases)"},
            "claude_code": {**skill_claude_agent.status(),
                            "label": "Claude Code CLI (tool-using subprocess agent)"},
        },
    }
    try:
        import skill_graph
        payload["checkpoint"] = skill_graph.checkpoint_status()
    except Exception:  # noqa: BLE001
        payload["checkpoint"] = None
    return jsonify(payload)


@app.route("/api/skill/runs")
def api_skill_runs():
    """Raw observability runs (build / eval / refine / review) with per-run metrics."""
    import skill_runs
    skill_id = (request.args.get("skill_id") or "").strip() or None
    kind = (request.args.get("kind") or "").strip() or None
    try:
        limit = int(request.args.get("limit") or 50)
    except (TypeError, ValueError):
        limit = 50
    return jsonify({"runs": skill_runs.list_runs(skill_id=skill_id, kind=kind, limit=limit)})


@app.route("/api/skill/<skill_id>")
def api_skill_get(skill_id):
    s = skills.get_skill(skill_id)
    if not s:
        return jsonify({"error": "Skill not found."}), 404
    return jsonify({"skill": s, "markdown": skills.to_skill_md(s)})


@app.route("/api/skill/build", methods=["POST"])
def api_skill_build():
    """Run the build pipeline (understand → analyze → codeact → eval → gate) over
    selected context, producing a drafted, evaluated, gated skill."""
    data = request.get_json(silent=True) or {}
    chunk_ids = data.get("chunk_ids") if isinstance(data.get("chunk_ids"), list) else None
    text = (data.get("text") or "").strip() or None
    query = (data.get("query") or "").strip() or None
    tags_in = data.get("tags")
    tags = ([t.strip() for t in tags_in.split(",") if t.strip()] if isinstance(tags_in, str)
            else [str(t).strip() for t in tags_in if str(t).strip()] if isinstance(tags_in, list)
            else None)
    if not (chunk_ids or text or query or tags):
        return jsonify({"error": "Provide context: chunk_ids, text, query, or tags."}), 400
    try:
        import skill_agent
        res = skill_agent.build_skill(
            chunk_ids=chunk_ids, text=text, query=query, tags=tags,
            where=(data.get("where") or "overall"),
            goal=(data.get("goal") or "").strip(),
            provider=(data.get("provider") or "auto").strip().lower(),
            model=(data.get("model") or "").strip() or None,
            run_rubric=bool(data.get("run_rubric", True)),
            run_triggering=bool(data.get("run_triggering", True)),
            use_tools=bool(data.get("use_tools", True)),
            backend=(data.get("backend") or "").strip().lower() or None,
            judge_provider=(data.get("judge_provider") or "").strip().lower() or None,
            judge_model=(data.get("judge_model") or "").strip() or None)
        res["stats"] = skills.stats()
        return jsonify(res)
    except Exception as e:  # noqa: BLE001  (SkillError -> actionable 400)
        return jsonify({"error": str(e)}), 400


def _skill_graph_or_error():
    try:
        import skill_graph
        return skill_graph, None
    except Exception as e:  # noqa: BLE001
        return None, (jsonify({"error": "LangGraph orchestration requires `langgraph` "
                              f"(pip install langgraph langgraph-checkpoint-sqlite): {e}"}), 503)


@app.route("/api/skill/graph/build", methods=["POST"])
def api_skill_graph_build():
    """Run the build as a checkpointed LangGraph StateGraph; pauses at human review."""
    mod, err = _skill_graph_or_error()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    chunk_ids = data.get("chunk_ids") if isinstance(data.get("chunk_ids"), list) else None
    tags_in = data.get("tags")
    tags = ([t.strip() for t in tags_in.split(",") if t.strip()] if isinstance(tags_in, str)
            else [str(t).strip() for t in tags_in if str(t).strip()] if isinstance(tags_in, list)
            else None)
    if not (chunk_ids or (data.get("text") or "").strip() or (data.get("query") or "").strip() or tags):
        return jsonify({"error": "Provide context: chunk_ids, text, query, or tags."}), 400
    try:
        res = mod.run_build(
            chunk_ids=chunk_ids, text=(data.get("text") or "").strip() or None,
            query=(data.get("query") or "").strip() or None, tags=tags,
            where=(data.get("where") or "overall"), goal=(data.get("goal") or "").strip(),
            provider=(data.get("provider") or "auto").strip().lower(),
            model=(data.get("model") or "").strip() or None,
            backend=(data.get("backend") or "").strip().lower() or None,
            use_tools=bool(data.get("use_tools", True)),
            run_rubric=bool(data.get("run_rubric", True)),
            run_triggering=bool(data.get("run_triggering", True)))
        res["stats"] = skills.stats()
        res["checkpoint"] = mod.checkpoint_backend()
        return jsonify(res)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


def _parse_skill_context(data):
    """Pull (chunk_ids, text, query, tags) from a build request body."""
    chunk_ids = data.get("chunk_ids") if isinstance(data.get("chunk_ids"), list) else None
    tags_in = data.get("tags")
    tags = ([t.strip() for t in tags_in.split(",") if t.strip()] if isinstance(tags_in, str)
            else [str(t).strip() for t in tags_in if str(t).strip()] if isinstance(tags_in, list)
            else None)
    return (chunk_ids, (data.get("text") or "").strip() or None,
            (data.get("query") or "").strip() or None, tags)


@app.route("/api/skill/graph/build_async", methods=["POST"])
def api_skill_graph_build_async():
    """Start a checkpointed LangGraph build in the BACKGROUND; returns a job_id to
    poll (so the UI isn't blocked while the build runs to the review interrupt)."""
    mod, err = _skill_graph_or_error()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    chunk_ids, text, query, tags = _parse_skill_context(data)
    if not (chunk_ids or text or query or tags):
        return jsonify({"error": "Provide context: chunk_ids, text, query, or tags."}), 400
    try:
        job = mod.start_build_async(
            chunk_ids=chunk_ids, text=text, query=query, tags=tags,
            where=(data.get("where") or "overall"), goal=(data.get("goal") or "").strip(),
            provider=(data.get("provider") or "auto").strip().lower(),
            model=(data.get("model") or "").strip() or None,
            backend=(data.get("backend") or "").strip().lower() or None,
            use_tools=bool(data.get("use_tools", True)),
            run_rubric=bool(data.get("run_rubric", True)),
            run_triggering=bool(data.get("run_triggering", True)))
        job["checkpoint"] = mod.checkpoint_backend()
        return jsonify(job)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/skill/graph/job/<job_id>")
def api_skill_graph_job(job_id):
    """Poll a background build: state is running | awaiting_review | done | error."""
    mod, err = _skill_graph_or_error()
    if err:
        return err
    job = mod.job_status(job_id)
    if not job:
        return jsonify({"error": "Unknown job."}), 404
    out = {"job_id": job_id, "state": job["state"], "thread_id": job["thread_id"],
           "skill_id": job.get("skill_id"), "error": job.get("error"),
           "checkpoint": mod.checkpoint_backend()}
    res = job.get("result") or {}
    if res:
        out.update({"awaiting_review": res.get("awaiting_review"), "gate": res.get("gate"),
                    "status": res.get("status"), "skill": res.get("skill"),
                    "eval": res.get("eval"), "thread_id": res.get("thread_id", job["thread_id"])})
    return jsonify(out)


@app.route("/api/skill/graph/resume", methods=["POST"])
def api_skill_graph_resume():
    """Resume a paused graph build with a human decision (accept | reject | revise)."""
    mod, err = _skill_graph_or_error()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    thread_id = (data.get("thread_id") or "").strip()
    decision = (data.get("decision") or "").strip().lower()
    if not thread_id or decision not in {"accept", "reject", "revise"}:
        return jsonify({"error": "thread_id and decision (accept|reject|revise) are required."}), 400
    score = data.get("score")
    try:
        score = float(score) if score is not None and score != "" else None
    except (TypeError, ValueError):
        score = None
    try:
        res = mod.resume_review(thread_id, decision=decision, score=score,
                                notes=(data.get("notes") or "").strip(),
                                reviewer=(data.get("reviewer") or "user").strip())
        res["stats"] = skills.stats()
        return jsonify(res)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/skill/graph/status")
def api_skill_graph_status():
    mod, err = _skill_graph_or_error()
    if err:
        return err
    thread_id = (request.args.get("thread_id") or "").strip()
    if not thread_id:
        return jsonify({"error": "thread_id is required."}), 400
    try:
        return jsonify(mod.get_status(thread_id))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/skill/<skill_id>/eval", methods=["POST"])
def api_skill_eval(skill_id):
    """Re-run evaluation (deterministic checks + rubric panel) on an existing skill."""
    s = skills.get_skill(skill_id)
    if not s:
        return jsonify({"error": "Skill not found."}), 404
    data = request.get_json(silent=True) or {}
    try:
        import skill_eval, skill_runs
        report = skill_eval.run_eval(
            s, provider=(data.get("provider") or "auto").strip().lower(),
            model=(data.get("model") or "").strip() or None,
            run_rubric=bool(data.get("run_rubric", True)),
            run_triggering=bool(data.get("run_triggering", True)),
            judge_provider=(data.get("judge_provider") or "").strip().lower() or None,
            judge_model=(data.get("judge_model") or "").strip() or None)
        updated = skills.record_eval(skill_id, report)
        trig = report.get("triggering") or {}
        skill_runs.record(kind="eval", skill_id=skill_id, skill_name=s.get("name", ""),
                          gate=report["gate"], status=updated["status"],
                          metrics={"deterministic_ratio": report["deterministic"]["ratio"],
                                   "rubric_mean": report.get("rubric_mean"),
                                   "trigger_f1": trig.get("f1")})
        return jsonify({"ok": True, "skill": updated, "eval": report, "stats": skills.stats()})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/skill/<skill_id>/review", methods=["POST"])
def api_skill_review(skill_id):
    """Record a human's alignment decision (accept / reject / revise) — the
    human-in-the-loop authority the automated gate defers to."""
    data = request.get_json(silent=True) or {}
    decision = (data.get("decision") or "").strip().lower()
    if decision not in {"accept", "reject", "revise"}:
        return jsonify({"error": "decision must be accept | reject | revise."}), 400
    score = data.get("score")
    try:
        score = float(score) if score is not None and score != "" else None
    except (TypeError, ValueError):
        score = None
    try:
        import skill_agent
        res = skill_agent.review_skill(
            skill_id, decision=decision, score=score,
            notes=(data.get("notes") or "").strip(),
            reviewer=(data.get("reviewer") or "user").strip(),
            rebuild_on_revise=bool(data.get("rebuild_on_revise", False)),
            provider=(data.get("provider") or "auto").strip().lower(),
            model=(data.get("model") or "").strip() or None)
        res["stats"] = skills.stats()
        return jsonify(res)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/skill/<skill_id>/rebuild", methods=["POST"])
def api_skill_rebuild(skill_id):
    """Re-run the pipeline for a skill, folding human revision notes into a new draft."""
    data = request.get_json(silent=True) or {}
    try:
        import skill_agent
        res = skill_agent.rebuild_skill(
            skill_id, provider=(data.get("provider") or "auto").strip().lower(),
            model=(data.get("model") or "").strip() or None,
            extra_guidance=(data.get("extra_guidance") or "").strip(),
            run_rubric=bool(data.get("run_rubric", True)))
        res["stats"] = skills.stats()
        return jsonify(res)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/skill/<skill_id>/refine", methods=["POST"])
def api_skill_refine(skill_id):
    """Self-improvement pass: rebuild the skill to fix its measured weaknesses
    (failed checks, triggering precision/recall, weakest rubric dimension)."""
    data = request.get_json(silent=True) or {}
    try:
        import skill_agent
        res = skill_agent.refine_skill(
            skill_id, provider=(data.get("provider") or "auto").strip().lower(),
            model=(data.get("model") or "").strip() or None,
            use_tools=bool(data.get("use_tools", True)),
            backend=(data.get("backend") or "").strip().lower() or None,
            run_rubric=bool(data.get("run_rubric", True)),
            run_triggering=bool(data.get("run_triggering", True)))
        res["stats"] = skills.stats()
        return jsonify(res)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/skill/<skill_id>/export", methods=["POST"])
def api_skill_export(skill_id):
    return jsonify(skills.export_skill(skill_id))


@app.route("/api/skill/<skill_id>", methods=["DELETE"])
def api_skill_delete(skill_id):
    return jsonify({"removed": skills.forget(skill_id), "stats": skills.stats()})


# --- MCP (Model Context Protocol) endpoints ---------------------------------
@app.route("/api/mcp/status")
def api_mcp_status():
    """Enabled MCP servers + loaded tools (read/write), and whether writes are allowed."""
    import mcp_tools
    return jsonify(mcp_tools.status())


@app.route("/api/mcp/call", methods=["POST"])
def api_mcp_call():
    """Invoke a READ MCP tool. Write tools are refused here (use /api/mcp/write)."""
    import mcp_tools, mcp_config
    data = request.get_json(silent=True) or {}
    server = (data.get("server") or "").strip()
    tool = (data.get("tool") or "").strip()
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    if not server or not tool:
        return jsonify({"error": "server and tool are required."}), 400
    if mcp_config.is_write_tool(server, tool):
        return jsonify({"error": f"{server}/{tool} is a write tool; use /api/mcp/write."}), 400
    try:
        return jsonify({"ok": True, "result": mcp_tools.call_tool(server, tool, args)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/mcp/write", methods=["POST"])
def api_mcp_write():
    """Gated write: deny-by-default; requires MCP_ALLOW_WRITES + ``approved: true``.
    Without approval it returns a preview + approval_required for a human to confirm."""
    import mcp_tools
    data = request.get_json(silent=True) or {}
    server = (data.get("server") or "").strip()
    tool = (data.get("tool") or "").strip()
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    if not server or not tool:
        return jsonify({"error": "server and tool are required."}), 400
    res = mcp_tools.execute_write(server, tool, args, approved=bool(data.get("approved")))
    return jsonify(res), (200 if (res.get("ok") or res.get("approval_required")) else 400)


@app.route("/api/mcp/ingest", methods=["POST"])
def api_mcp_ingest():
    """Call a READ MCP tool and stage its output into the knowledge graph (staging)."""
    import mcp_tools
    data = request.get_json(silent=True) or {}
    server = (data.get("server") or "").strip()
    tool = (data.get("tool") or "").strip()
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    if not server or not tool:
        return jsonify({"error": "server and tool are required."}), 400
    try:
        res = mcp_tools.ingest_result(server, tool, args,
                                      source_title=(data.get("source_title") or "").strip())
        res["stats"] = kg.stats()
        return jsonify(res), (200 if res.get("ok") else 400)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


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
    rerank = bool(data.get("rerank", True))
    mmr = bool(data.get("mmr", False))
    try:
        import rag
        hits = rag.retrieve(q, k=k, rerank_hits=rerank, mmr=mmr)
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
    rerank = bool(data.get("rerank", True))
    mmr = bool(data.get("mmr", False))
    try:
        import rag
        return jsonify(rag.answer(question, provider=provider, model=model,
                                  k=k, rerank_hits=rerank, mmr=mmr))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/eval", methods=["POST"])
def api_rag_eval():
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    k = int(data.get("k") or 6)
    max_q = int(data.get("max_questions") or 10)
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
                              k=k, rerank_hits=rerank, mmr=mmr)
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
                                               rerank=rerank, mmr=mmr,
                                               k=k, ragas=ragas))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/crossdoc/labels", methods=["POST"])
def api_rag_crossdoc_labels():
    """Seed the human-label file with LLM-drafted key_points for every cross-doc
    question (a human then fills in human_score + reviewed_answer to calibrate the judges)."""
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()
    overwrite = bool(data.get("overwrite", False))
    try:
        import crossdoc
        return jsonify(crossdoc.scaffold_human_labels(provider=provider, model=model, overwrite=overwrite))
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
    """Stage chunks into the KG. Returns ``(total, chunk_ids)`` so callers that
    project from the cached store can record which chunk nodes were created."""
    chunks = ingestion.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    total = len(chunks)
    if total == 0:
        return 0, []
    chunk_ids = []
    for i, c in enumerate(chunks, start=1):
        title = source_title if total == 1 else f"{source_title} [{i}/{total}]"
        chunk_tags = list(tags) + ([f"part:{i}/{total}"] if total > 1 else [])
        node = kg.add_chunk(c, source_url=source_url, source_title=title, tags=chunk_tags)
        chunk_ids.append(node["id"])
    return total, chunk_ids


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
    total, _ = _ingest_chunks(
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
            n, _ = _ingest_chunks(
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
            n, _ = _ingest_chunks(
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
    # threaded=True so background skill builds + job-status polls are served concurrently.
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
