import json
import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

import knowledge_graph as kg
import ingestion

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

PROVIDERS = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5-20251001",
        "models": [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-7",
        ],
    },
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "qwen": {
        "label": "Qwen (DashScope)",
        "env_key": "DASHSCOPE_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-72b-instruct"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
}


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
    base_url = os.environ.get(config["base_url_env"], config["base_url"])
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


def first_available_provider() -> str | None:
    for name, cfg in PROVIDERS.items():
        if os.environ.get(cfg["env_key"]):
            return name
    return None


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
            "configured": bool(os.environ.get(cfg["env_key"])),
            "env_key": cfg["env_key"],
        }
    return jsonify({"providers": payload, "auto": first_available_provider()})


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


ENTITY_PROMPT = """You are extracting a knowledge graph from a passage of text.

Identify the 8-15 MOST IMPORTANT entities — the named people, organizations,
places, products, technologies, methods, frameworks, or concepts that this
passage is actually ABOUT. Skip generic terms and sentence-starting words
like "In", "When", "Round". Prefer multi-word canonical names.

Then identify the SEMANTIC RELATIONSHIPS between those entities — who did
what to whom, what depends on what, what is a part of what. These triples
are what makes a knowledge graph useful.

Return ONLY a JSON object with this exact shape:

{{
  "entities": [
    {{
      "name": "short canonical form (1-5 words, Title Case)",
      "kind": "person | organization | place | concept | technology | method | event | product",
      "importance": 1-5,
      "confidence": "EXTRACTED | INFERRED | AMBIGUOUS"
    }}
  ],
  "relations": [
    {{
      "source": "<entity name as above>",
      "target": "<entity name as above>",
      "predicate": "short active-voice verb phrase (1-4 words)",
      "confidence": "EXTRACTED | INFERRED | AMBIGUOUS"
    }}
  ]
}}

Rules:
- Entity names in `relations` MUST exactly match names in `entities`.
- At least half the entities should appear in at least one relation.
- Predicates should be specific and meaningful (e.g. "expands prompt into",
  "evaluates output of", "replaces", "depends on"), not generic ("mentions",
  "related to", "is").

Text:
\"\"\"
{text}
\"\"\"

Return only the JSON object, no prose:"""


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        out = json.loads(m.group())
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def extract_kg_llm(text: str, provider: str, model: str) -> dict:
    """Returns {'entities': [...], 'relations': [...]}."""
    prompt = ENTITY_PROMPT.format(text=text[:8000])
    if provider == "anthropic":
        if Anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
            return {}
        client = Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        out = "".join(b.text for b in msg.content if hasattr(b, "text"))
    elif provider in PROVIDERS and OpenAI is not None:
        config = PROVIDERS[provider]
        api_key = os.environ.get(config["env_key"])
        if not api_key:
            return {}
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get(config["base_url_env"], config["base_url"]),
        )
        resp = client.chat.completions.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""
    else:
        return {}
    parsed = _parse_json_object(out)
    if not isinstance(parsed.get("entities"), list):
        parsed["entities"] = []
    if not isinstance(parsed.get("relations"), list):
        parsed["relations"] = []
    return parsed


@app.route("/api/kg/stats")
def api_kg_stats():
    return jsonify(kg.stats())


@app.route("/api/kg/graph")
def api_kg_graph():
    where = request.args.get("where", "current")
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
    provider = (data.get("provider") or "auto").strip().lower()
    model = (data.get("model") or "").strip()

    extract_fn = None
    used_provider = "heuristic"
    used_model = ""
    if use_ai:
        actual = provider if provider in PROVIDERS else first_available_provider()
        if actual:
            actual_model = model or PROVIDERS[actual]["default_model"]
            used_provider, used_model = actual, actual_model
            extract_fn = lambda t: extract_kg_llm(t, actual, actual_model)

    result = kg.integrate(extract_fn=extract_fn)
    result["provider_used"] = used_provider
    result["model_used"] = used_model
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
