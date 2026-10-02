"""Tests for the optional You.com web-search backend (app.py · web_search).

WEB_SEARCH_PROVIDER=youcom routes web_search() through You.com's hosted MCP
server; any other value (or unset) keeps the DuckDuckGo (ddgs) default. These
tests are offline: a scripted fake stands in for the MCP endpoint
(initialize → initialized → tools/call) so the full JSON-RPC wire format —
including SSE streams, progress notifications, id-less result frames, and
non-ASCII payloads — is exercised deterministically without a network. A live
smoke test at the bottom validates the keyless free profile end-to-end and
skips automatically when there's no network.

Run directly (``python test_web_search_youcom.py``) or under pytest.
"""

import json
import os
import sys
import tempfile

# Isolate state BEFORE importing app: throwaway data dir + no model keys so the
# import path stays offline and deterministic.
os.environ["KG_DATA_DIR"] = (os.environ.get("BYOWIKI_TEST_DATA_DIR")
                             or tempfile.mkdtemp(prefix="youcom_test_"))
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)
# The You.com backend is env-driven at call time; start from a clean slate.
for _v in ("WEB_SEARCH_PROVIDER", "YDC_API_KEY", "YOUCOM_MCP_URL"):
    os.environ.pop(_v, None)

import app  # noqa: E402


def _skip(msg):
    """Skip under pytest; print + return when run directly (no pytest dependency)."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"  SKIP  {msg}")


class _EnvOverride:
    """Set env vars for the duration of a test; restore the originals after."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.saved = {}

    def __enter__(self):
        for key, val in self.kwargs.items():
            self.saved[key] = os.environ.get(key)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def __exit__(self, *exc):
        for key, val in self.saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


class _FakeResp:
    def __init__(self, text="", headers=None, status=200):
        self.text = text
        self.headers = headers or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeYoucomMCP:
    """Scripted You.com MCP endpoint (Streamable HTTP).

    Records every (url, payload, headers) call. Answers ``initialize`` with a
    session id, ``notifications/initialized`` with 202 + empty body, and
    ``tools/call`` with ``result_body`` — a bare JSON object or an SSE stream.
    """

    def __init__(self, result_body):
        self.result_body = result_body
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "payload": json,
                           "headers": dict(headers or {})})
        method = (json or {}).get("method")
        if method == "initialize":
            return _FakeResp(
                '{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"you.com"}}}',
                headers={"Mcp-Session-Id": "sess-123"})
        if method == "notifications/initialized":
            return _FakeResp("", status=202)
        if method == "tools/call":
            return _FakeResp(self.result_body)
        raise AssertionError(f"unexpected MCP method: {method!r}")


def _result_body(items, *, sse=False, progress=False, with_id=False):
    """Build a tools/call response body: the JSON-RPC envelope around You.com's
    ``{"results":{"web":[…]}}`` payload — optionally as an SSE stream, with a
    progress notification first, and/or with the id stripped from the result
    frame (the server omits it on result messages)."""
    inner = json.dumps({"results": {"web": items}})
    envelope = {"jsonrpc": "2.0",
                "result": {"content": [{"type": "text", "text": inner}]}}
    if with_id:
        envelope["id"] = 2
    if not sse:
        return json.dumps(envelope)
    frames = []
    if progress:
        frames.append("event: message\ndata: " + json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/progress",
             "params": {"progress": 1}}))
    frames.append("event: message\ndata: " + json.dumps(envelope))
    return "\n\n".join(frames) + "\n\n"


class _FakeDDGS:
    """DuckDuckGo stand-in recording queries; returns one canned hit."""

    seen = []

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=8):
        _FakeDDGS.seen.append((query, max_results))
        return [{"title": "DDG hit", "href": "https://example.com/ddg",
                 "body": "ddg snippet"}]


# --- offline tests (scripted MCP endpoint, deterministic everywhere) ----------

def test_default_backend_is_duckduckgo():
    """Unset (or any non-youcom) WEB_SEARCH_PROVIDER keeps the DuckDuckGo
    backend — the You.com path must never run."""
    orig_ddgs, orig_youcom = app.DDGS, app._youcom_search

    def _boom(*a, **k):
        raise AssertionError("You.com must not run when WEB_SEARCH_PROVIDER is unset")
    app.DDGS = _FakeDDGS
    app._youcom_search = _boom
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER=None):
            results = app.web_search("agentic rl")
        assert results == [{"title": "DDG hit", "url": "https://example.com/ddg",
                            "snippet": "ddg snippet"}], results
        with _EnvOverride(WEB_SEARCH_PROVIDER="duckduckgo"):
            app.web_search("second call")
        assert _FakeDDGS.seen == [("agentic rl", 8), ("second call", 8)]
    finally:
        app.DDGS, app._youcom_search = orig_ddgs, orig_youcom


def test_youcom_backend_flows_through_mcp_and_maps_results():
    """youcom speaks the MCP wire format (initialize → initialized →
    tools/call, session id passed along) and maps results to the exact
    {title, url, snippet} shape the DuckDuckGo backend returns — with the
    first highlight as the snippet when no description is present."""
    items = [
        {"title": "You hit", "url": "https://example.com/you",
         "description": "a description"},
        {"title": "No desc", "url": "https://example.com/hl",
         "highlights": ["first highlight", "second"]},
    ]
    fake = _FakeYoucomMCP(_result_body(items))
    orig_post, orig_ddgs = app.requests.post, app.DDGS
    app.requests.post = fake
    app.DDGS = None  # prove the You.com path never needs ddgs
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom"):
            results = app.web_search("context engineering", max_results=5)
        assert results == [
            {"title": "You hit", "url": "https://example.com/you",
             "snippet": "a description"},
            {"title": "No desc", "url": "https://example.com/hl",
             "snippet": "first highlight"},
        ], results
        assert [c["payload"]["method"] for c in fake.calls] == [
            "initialize", "notifications/initialized", "tools/call"]
        assert fake.calls[0]["url"] == app.YOUCOM_MCP_URL_FREE
        assert "Authorization" not in fake.calls[0]["headers"]
        assert fake.calls[1]["headers"].get("Mcp-Session-Id") == "sess-123"
        assert fake.calls[2]["headers"].get("Mcp-Session-Id") == "sess-123"
        params = fake.calls[2]["payload"]["params"]
        assert params["name"] == "you-search"
        assert params["arguments"] == {"query": "context engineering", "count": 5}
    finally:
        app.requests.post, app.DDGS = orig_post, orig_ddgs


def test_youcom_authenticated_uses_bearer_and_authed_url():
    """YDC_API_KEY switches to the authenticated endpoint with a Bearer header;
    without it the keyless free profile is used (no Authorization at all)."""
    fake = _FakeYoucomMCP(_result_body([]))
    orig_post = app.requests.post
    app.requests.post = fake
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom", YDC_API_KEY="k-123"):
            app.web_search("q")
        assert fake.calls[0]["url"] == app.YOUCOM_MCP_URL_AUTHED
        assert fake.calls[0]["headers"].get("Authorization") == "Bearer k-123"

        fake.calls.clear()
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom", YDC_API_KEY=None):
            app.web_search("q")
        assert fake.calls[0]["url"] == app.YOUCOM_MCP_URL_FREE
        assert "Authorization" not in fake.calls[0]["headers"]
    finally:
        app.requests.post = orig_post


def test_youcom_url_override():
    """YOUCOM_MCP_URL overrides the endpoint (e.g. a local proxy)."""
    fake = _FakeYoucomMCP(_result_body([]))
    orig_post = app.requests.post
    app.requests.post = fake
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom",
                           YOUCOM_MCP_URL="http://127.0.0.1:8999/mcp"):
            app.web_search("q")
        assert fake.calls[0]["url"] == "http://127.0.0.1:8999/mcp"
    finally:
        app.requests.post = orig_post


def test_youcom_sse_stream_progress_and_idless_result():
    """tools/call may be answered with an SSE stream: a progress notification
    first, then the result frame — with the JSON-RPC id omitted. The parser
    must skip notifications and still find the result."""
    items = [{"title": "SSE", "url": "https://example.com/sse",
              "description": "streamed"}]
    body = _result_body(items, sse=True, progress=True)  # no id in the frame
    fake = _FakeYoucomMCP(body)
    orig_post = app.requests.post
    app.requests.post = fake
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom"):
            results = app.web_search("query")
        assert results == [{"title": "SSE", "url": "https://example.com/sse",
                            "snippet": "streamed"}], results
    finally:
        app.requests.post = orig_post


def test_youcom_nonascii_payload_survives_sse_split():
    """``str.splitlines`` splits on U+0085/U+2028 as well as LF/CRLF — inside a
    UTF-8 result payload that would shred a description mid-sentence. The
    parser splits frames on LF/CRLF only, so Unicode separators stay intact."""
    desc = "Sentence one.\u0085 Sentence two.\u2028 Sentence three."
    items = [{"title": "多智能体 with unicode separators",
              "url": "https://example.com/cjk", "description": desc}]
    fake = _FakeYoucomMCP(_result_body(items, sse=True))
    orig_post = app.requests.post
    app.requests.post = fake
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom"):
            results = app.web_search("query")
        assert results[0]["snippet"] == desc, results
        assert results[0]["title"] == "多智能体 with unicode separators"
    finally:
        app.requests.post = orig_post


def test_youcom_failure_falls_back_to_duckduckgo():
    """A network failure on the You.com path must fall open to DuckDuckGo so
    search keeps answering."""
    def _boom(*a, **k):
        import requests
        raise requests.ConnectionError("no route to host")
    orig_post, orig_ddgs = app.requests.post, app.DDGS
    app.requests.post = _boom
    app.DDGS = _FakeDDGS
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom"):
            results = app.web_search("fallback please")
        assert results == [{"title": "DDG hit", "url": "https://example.com/ddg",
                            "snippet": "ddg snippet"}], results
    finally:
        app.requests.post, app.DDGS = orig_post, orig_ddgs


def test_youcom_iserror_result_falls_back_to_duckduckgo():
    """An isError result (e.g. daily quota exhausted) is a failure, not an
    answer — it must also fall back to DuckDuckGo."""
    envelope = {"jsonrpc": "2.0",
                "result": {"isError": True,
                           "content": [{"type": "text",
                                        "text": "daily quota exceeded"}]}}
    fake = _FakeYoucomMCP(json.dumps(envelope))
    orig_post, orig_ddgs = app.requests.post, app.DDGS
    app.requests.post = fake
    app.DDGS = _FakeDDGS
    try:
        with _EnvOverride(WEB_SEARCH_PROVIDER="youcom"):
            results = app.web_search("quota test")
        assert results == [{"title": "DDG hit", "url": "https://example.com/ddg",
                            "snippet": "ddg snippet"}], results
    finally:
        app.requests.post, app.DDGS = orig_post, orig_ddgs


# --- live smoke test against the real keyless free profile -------------------

def test_live_youcom_keyless_search():
    """Live end-to-end: the keyless free profile answers a real query with no
    YDC_API_KEY — the same zero-setup property that makes ddgs the default.
    Called through _youcom_search directly so a failure is visible rather
    than silently falling back. Skips without network."""
    try:
        results = app._youcom_search("context engineering for llm agents", 3)
    except Exception as e:  # noqa: BLE001 — offline environments skip
        _skip(f"network unavailable: {e}")
        return
    if not results:
        _skip("empty result set (free-profile daily quota may be exhausted)")
        return
    for r in results:
        assert set(r) == {"title", "url", "snippet"}, r
        assert r["url"].startswith("http"), r
    print(f"  live ok  {len(results)} results · {results[0]['title'][:60]}")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} youcom web-search tests passed.")


if __name__ == "__main__":
    _run_all()
