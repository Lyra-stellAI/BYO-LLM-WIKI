"""Tests for the search bar's link context-extraction (app.py · POST /api/search).

The search bar must, given a *link*, fetch the page and extract its readable
context — not run a keyword web search on the URL string (which returns unrelated
hits, e.g. an arXiv link surfacing a stray Facebook post). The offline tests
monkeypatch the network so they're deterministic everywhere; the live test at the
bottom fetches the three reference links from the task and asserts real context
comes back. The live test skips automatically when there's no network.

Run directly (``python test_search.py``) or under pytest.
"""

import os
import sys
import tempfile

# Isolate state BEFORE importing app: throwaway data dir + no model keys so the
# import path stays offline and deterministic.
os.environ["KG_DATA_DIR"] = (os.environ.get("BYOWIKI_TEST_DATA_DIR")
                             or tempfile.mkdtemp(prefix="search_test_"))
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

import app  # noqa: E402

# The three reference links from the task, each with a few keywords we expect to
# appear in the extracted context (case-insensitive).
LINKS = [
    ("https://arxiv.org/abs/2509.02547",
     ("reinforcement learning", "survey", "agentic")),
    ("https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html",
     ("search tool", "verl")),
    ("https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture",
     ("multi-agent", "architecture")),
]


def _client():
    return app.app.test_client()


def _skip(msg):
    """Skip under pytest; print + return when run directly (no pytest dependency)."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"  SKIP  {msg}")


# --- offline routing tests (monkeypatched, deterministic everywhere) --------

def test_link_query_extracts_context():
    """A URL routes to fetch + extract context — web_search is never called."""
    orig_fetch, orig_search = app.fetch_page, app.web_search
    app.fetch_page = lambda url: {"title": "Example Title", "url": url,
                                  "text": "Alpha beta gamma delta. " * 80}

    def _no_search(*a, **k):
        raise AssertionError("web_search must not run for a URL")
    app.web_search = _no_search
    try:
        r = _client().post("/api/search", json={"query": "https://example.com/page"})
        data = r.get_json()
        assert r.status_code == 200, data
        assert data["kind"] == "link", data
        assert len(data["results"]) == 1, data
        res = data["results"][0]
        assert res["url"] == "https://example.com/page"
        assert res["title"] == "Example Title"
        assert res["chars"] > 500
        assert "Alpha beta gamma" in res["context"]
        assert res["snippet"]  # a short lead preview is present
    finally:
        app.fetch_page, app.web_search = orig_fetch, orig_search


def test_keyword_query_uses_web_search():
    """A non-URL still runs a keyword web search — fetch_page is never called."""
    orig_fetch, orig_search = app.fetch_page, app.web_search
    app.web_search = lambda q, max_results=8: [
        {"title": "Hit", "url": "https://example.com", "snippet": "s"}]

    def _no_fetch(url):
        raise AssertionError("fetch_page must not run for a keyword query")
    app.fetch_page = _no_fetch
    try:
        r = _client().post("/api/search", json={"query": "agentic rl survey"})
        data = r.get_json()
        assert r.status_code == 200, data
        assert data["kind"] == "web", data
        assert data["results"][0]["title"] == "Hit"
    finally:
        app.fetch_page, app.web_search = orig_fetch, orig_search


def test_empty_query_rejected():
    r = _client().post("/api/search", json={"query": "   "})
    assert r.status_code == 400


def test_fetch_error_reports_cleanly():
    """A network failure while fetching a link returns a clean 502, not a 500."""
    import requests
    orig_fetch = app.fetch_page

    def _boom(url):
        raise requests.ConnectionError("no route to host")
    app.fetch_page = _boom
    try:
        r = _client().post("/api/search", json={"query": "https://example.com/page"})
        data = r.get_json()
        assert r.status_code == 502, data
        assert "Could not fetch link" in data["error"]
    finally:
        app.fetch_page = orig_fetch


class _FakeResp:
    def __init__(self, content=b"", headers=None):
        self.content = content
        self.headers = headers or {}
        self.text = content.decode("utf-8", "replace")

    def raise_for_status(self):
        pass


def test_pdf_url_detected_and_parsed_not_as_html():
    """A PDF (e.g. an arXiv /pdf/ link) must be extracted with the PDF parser,
    never fed to the HTML parser (which returns the raw %PDF bytes as content)."""
    # detection: by content-type, by .pdf extension, by %PDF magic header
    assert app._looks_like_pdf("https://arxiv.org/pdf/2604.24026",
                               _FakeResp(b"%PDF-1.7\n", {"Content-Type": "application/pdf"}))
    assert app._looks_like_pdf("https://x.com/a.pdf", _FakeResp(b"junk"))
    assert app._looks_like_pdf("https://x.com/doc?x=1", _FakeResp(b"%PDF-1.4 ..."))
    assert not app._looks_like_pdf("https://x.com/page",
                                   _FakeResp(b"<html>hi</html>", {"Content-Type": "text/html"}))

    # fetch_page routes PDF bytes through ingestion.parse_file, not BeautifulSoup
    orig_get, orig_parse = app.requests.get, app.ingestion.parse_file
    app.requests.get = lambda *a, **k: _FakeResp(b"%PDF-1.7 fake bytes",
                                                 {"Content-Type": "application/pdf"})
    app.ingestion.parse_file = lambda name, content: "EXTRACTED PDF TEXT"
    try:
        page = app.fetch_page("https://arxiv.org/pdf/2604.24026")
        assert page["text"] == "EXTRACTED PDF TEXT", page
        assert not page["text"].startswith("%PDF")
    finally:
        app.requests.get, app.ingestion.parse_file = orig_get, orig_parse


def test_html_extraction_strips_citation_boilerplate():
    """Scholarly-page chrome (arXiv extra-services / labs tabs / BibTeX export
    widgets) must not leak into extracted text — otherwise it pollutes the KG
    with 'Bibliographic Explorer / BibTeX citation / Loading…' junk chunks."""
    import ingestion
    from bs4 import BeautifulSoup
    html = """<html><head><title>Paper</title></head><body>
      <nav class="site-nav">Home Login</nav>
      <article><h1>Paper</h1>
        <blockquote class="abstract">A method for context engineering.</blockquote>
        <p>The body has the real findings.</p></article>
      <div class="extra-services">
        <div class="labstabs">Bibliographic Explorer Toggle Bibliographic and Citation Tools</div>
        <div class="bib-cite">ADS Google Scholar Semantic Scholar export BibTeX citation</div>
        <div id="bibtex-modal" aria-hidden="true">BibTeX formatted citation loading... Data provided by: Bookmark</div></div>
      <div class="related-papers">Related Papers recommend</div>
      <footer>arXiv footer</footer></body></html>"""
    text = ingestion.main_text(BeautifulSoup(html, "lxml"))
    for junk in ("Bibliographic Explorer", "BibTeX", "Google Scholar", "Semantic Scholar",
                 "loading", "Bookmark", "Related Papers", "Home Login", "arXiv footer"):
        assert junk.lower() not in text.lower(), f"boilerplate leaked: {junk!r}"
    assert "context engineering" in text and "real findings" in text  # content kept


def test_extract_date_from_metadata():
    """_extract_date pulls a published/updated date (for sort-by-date on the Read
    tab); returns '' when the page has none."""
    from bs4 import BeautifulSoup
    s1 = BeautifulSoup(
        '<html><head><meta property="article:published_time" content="2026-03-14T09:00:00Z">'
        '</head><body>x</body></html>', "lxml")
    assert app._extract_date(s1) == "2026-03-14"
    s2 = BeautifulSoup('<html><body><time datetime="2025-12-01">Dec</time></body></html>', "lxml")
    assert app._extract_date(s2) == "2025-12-01"
    s3 = BeautifulSoup("<html><body><p>no date here</p></body></html>", "lxml")
    assert app._extract_date(s3) == ""
    # arXiv-style citation_date with slashes → normalized to ISO
    s4 = BeautifulSoup('<html><head><meta name="citation_date" content="2025/09/02">'
                       "</head><body>x</body></html>", "lxml")
    assert app._extract_date(s4) == "2025-09-02"
    # JSON-LD datePublished (modern CMS/news)
    s5 = BeautifulSoup('<html><head><script type="application/ld+json">'
                       '{"@type":"Article","datePublished":"2026-06-16T10:00:00Z"}'
                       "</script></head><body>x</body></html>", "lxml")
    assert app._extract_date(s5) == "2026-06-16"


def test_fetch_page_includes_date_field():
    """fetch_page returns a `date` (possibly empty) so extract results can sort by it."""
    orig = app.requests.get
    html = (b'<html><head><title>T</title>'
            b'<meta property="article:published_time" content="2026-01-09T00:00:00Z">'
            b'</head><body><article><p>Body content here.</p></article></body></html>')
    app.requests.get = lambda *a, **k: _FakeResp(html, {"Content-Type": "text/html"})
    try:
        page = app.fetch_page("https://example.com/post")
        assert page.get("date") == "2026-01-09", page
    finally:
        app.requests.get = orig


# --- live test against the three reference links ----------------------------

def test_live_links_extract_context():
    """End-to-end: each reference link yields real extracted context."""
    client = _client()
    for url, keywords in LINKS:
        r = client.post("/api/search", json={"query": url})
        data = r.get_json()
        # No network in this environment? Skip rather than fail.
        if r.status_code == 502 and "fetch link" in (data.get("error") or "").lower():
            _skip(f"network unavailable: {data.get('error')}")
            return
        assert r.status_code == 200, data
        assert data["kind"] == "link", data
        assert len(data["results"]) == 1, data
        res = data["results"][0]
        assert res["url"] == url
        assert res["chars"] > 500, f"too little context from {url}: {res['chars']} chars"
        haystack = (res["title"] + " " + res["context"]).lower()
        assert any(k in haystack for k in keywords), \
            f"none of {keywords} found in context from {url}"
        print(f"  link ok  {res['chars']:>6} chars · {res['title'][:60]}")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} search tests passed.")


if __name__ == "__main__":
    _run_all()
