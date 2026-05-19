import json
import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

import knowledge_graph as kg

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


ENTITY_PROMPT = """Extract 3-8 important named entities or key concepts from the text below.

Return ONLY a JSON array. Each item must be an object with:
- "name": short canonical form (1-4 words)
- "kind": one of "person", "organization", "place", "concept", "technology", "event"
- "confidence": one of "EXTRACTED" (named explicitly), "INFERRED" (strongly implied), "AMBIGUOUS"

Text:
{text}

Return only the JSON array, no prose:"""


def _parse_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        out = json.loads(m.group())
        return out if isinstance(out, list) else []
    except Exception:
        return []


def extract_entities_llm(text: str, provider: str, model: str) -> list:
    prompt = ENTITY_PROMPT.format(text=text[:6000])
    if provider == "anthropic":
        if Anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
            return []
        client = Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        out = "".join(b.text for b in msg.content if hasattr(b, "text"))
    elif provider in PROVIDERS and OpenAI is not None:
        config = PROVIDERS[provider]
        api_key = os.environ.get(config["env_key"])
        if not api_key:
            return []
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get(config["base_url_env"], config["base_url"]),
        )
        resp = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""
    else:
        return []
    return _parse_json_array(out)


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
            extract_fn = lambda t: extract_entities_llm(t, actual, actual_model)

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
