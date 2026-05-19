const $ = (sel) => document.querySelector(sel);
const queryEl = $("#query");
const searchBtn = $("#searchBtn");
const summarizeBtn = $("#summarizeBtn");
const statusEl = $("#status");
const resultsEl = $("#results");

function escapeHTML(str = "") {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function setStatus(html, kind = "info") {
  statusEl.className = "status" + (kind === "error" ? " error" : "");
  statusEl.innerHTML = html;
  statusEl.classList.remove("hidden");
}

function clearStatus() {
  statusEl.classList.add("hidden");
  statusEl.innerHTML = "";
}

function setBusy(busy) {
  searchBtn.disabled = busy;
  summarizeBtn.disabled = busy;
  queryEl.disabled = busy;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function renderSearchResults(data) {
  if (!data.results || data.results.length === 0) {
    resultsEl.innerHTML = `<div class="status">No results found for "${escapeHTML(data.query)}".</div>`;
    return;
  }
  const cards = data.results.map((r) => `
    <article class="result-card">
      <h3><a href="${escapeHTML(r.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(r.title || r.url)}</a></h3>
      <div class="url">${escapeHTML(r.url)}</div>
      <p class="snippet">${escapeHTML(r.snippet || "")}</p>
      <button class="btn btn-secondary" data-summarize="${escapeHTML(r.url)}" style="margin-top:10px; padding:8px 14px; font-size:0.85rem;">Summarize this</button>
    </article>
  `).join("");
  resultsEl.innerHTML = cards;

  resultsEl.querySelectorAll("[data-summarize]").forEach((btn) => {
    btn.addEventListener("click", () => {
      queryEl.value = btn.dataset.summarize;
      doSummarize();
    });
  });
}

function renderSummary(data) {
  const sourceLine = data.url
    ? `Source: <a href="${escapeHTML(data.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(data.url)}</a>`
    : "";
  resultsEl.innerHTML = `
    <article class="summary-card">
      <div class="meta">
        <span class="badge">${data.engine === "claude" ? "AI summary" : "Extractive summary"}</span>
        <span>${data.chars.toLocaleString()} chars analyzed</span>
        ${sourceLine ? `<span>${sourceLine}</span>` : ""}
      </div>
      <h2>${escapeHTML(data.title)}</h2>
      <div class="summary-body">${escapeHTML(data.summary)}</div>
    </article>
  `;
}

async function doSearch() {
  const query = queryEl.value.trim();
  if (!query) { setStatus("Type something to search.", "error"); return; }
  clearStatus(); resultsEl.innerHTML = ""; setBusy(true);
  setStatus(`<span class="spinner"></span>Searching for "${escapeHTML(query)}"…`);
  try {
    const data = await postJSON("/api/search", { query });
    clearStatus();
    renderSearchResults(data);
  } catch (e) {
    setStatus(escapeHTML(e.message), "error");
  } finally {
    setBusy(false);
  }
}

async function doSummarize() {
  const input = queryEl.value.trim();
  if (!input) { setStatus("Paste a URL or text to summarize.", "error"); return; }
  clearStatus(); resultsEl.innerHTML = ""; setBusy(true);
  setStatus(`<span class="spinner"></span>Reading and summarizing…`);
  try {
    const data = await postJSON("/api/summarize", { input });
    clearStatus();
    renderSummary(data);
  } catch (e) {
    setStatus(escapeHTML(e.message), "error");
  } finally {
    setBusy(false);
  }
}

searchBtn.addEventListener("click", doSearch);
summarizeBtn.addEventListener("click", doSummarize);

queryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    if (e.shiftKey) doSummarize();
    else doSearch();
  }
});
