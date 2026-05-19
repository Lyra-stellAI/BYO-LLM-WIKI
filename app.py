import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

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


def claude_summary(title: str, url: str, text: str) -> str | None:
    if Anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    client = Anthropic()
    snippet = text[:MAX_CHARS_FOR_MODEL]
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Summarize the following web page in clear, concise prose. "
                f"Start with a one-sentence TL;DR, then 3-6 bullet points of key takeaways.\n\n"
                f"Title: {title}\nURL: {url}\n\nContent:\n{snippet}"
            ),
        }],
    )
    return "".join(block.text for block in msg.content if hasattr(block, "text"))


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

        summary = claude_summary(title, url, text)
        engine = "claude"
        if not summary:
            summary = extractive_summary(text)
            engine = "extractive"

        return jsonify({
            "title": title,
            "url": url,
            "summary": summary,
            "engine": engine,
            "chars": len(text),
        })
    except requests.HTTPError as e:
        return jsonify({"error": f"Could not fetch page: HTTP {e.response.status_code}"}), 502
    except requests.RequestException as e:
        return jsonify({"error": f"Could not fetch page: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
