const $ = (sel) => document.querySelector(sel);
const queryEl = $("#query");
const searchBtn = $("#searchBtn");
const extractBtn = $("#extractBtn");
const summarizeBtn = $("#summarizeBtn");
const statusEl = $("#status");
const resultsEl = $("#results");
const providerEl = $("#provider");
const modelEl = $("#model");
const modelListEl = $("#model-suggestions");
const providerStatusEl = $("#providerStatus");

let providerInfo = {};

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
  if (extractBtn) extractBtn.disabled = busy;
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

function isUrl(text = "") {
  try {
    const u = new URL(text.trim());
    return u.protocol === "http:" || u.protocol === "https:";
  } catch (e) {
    return false;
  }
}

function renderLinkContext(r) {
  const context = r.context || r.snippet || "";
  resultsEl.innerHTML = `
    <article class="summary-card">
      <div class="meta">
        <span class="badge">Extracted context</span>
        <span>${(r.chars || context.length).toLocaleString()} chars extracted</span>
        <span>Source: <a href="${escapeHTML(r.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(r.url)}</a></span>
      </div>
      <h2>${escapeHTML(r.title || r.url)}</h2>
      <div class="summary-body context-extract">${escapeHTML(context)}</div>
      <div class="context-actions">
        <button class="btn btn-secondary" data-summarize="${escapeHTML(r.url)}">Summarize this</button>
        <button class="btn btn-primary" data-ingest-store="1">Ingest to store</button>
        <button class="btn btn-accent" data-addkg-context="1">+ KG</button>
      </div>
    </article>
  `;
  resultsEl.querySelector("[data-summarize]")?.addEventListener("click", () => {
    queryEl.value = r.url;
    doSummarize();
  });
  resultsEl.querySelector("[data-ingest-store]")?.addEventListener("click", (e) => {
    ingestToStore({ url: r.url || "", source_title: r.title || "", text: context,
                    kind: r.url ? "url" : "text" }, e.currentTarget);
  });
  resultsEl.querySelector("[data-addkg-context]")?.addEventListener("click", () => {
    window.kg.openModal({ text: context, source_title: r.title || "", source_url: r.url || "" });
  });
}

// --- Cached store: extract many URLs, then ingest once for reuse everywhere --
let extractItems = [];

// Web-search "Show more" state: initial count, per-click step, hard cap, and the
// URLs checked before a re-render (preserved across "Show more").
const WEB_BASE = 10, WEB_STEP = 10, WEB_MAX = 30;
let webN = WEB_BASE, lastWebQuery = "", preservedChecks = new Set();

function parseUrls(text = "") {
  return text.split(/[\s,]+/).map((s) => s.trim()).filter((s) => isUrl(s));
}

function errBlock(errors) {
  if (!errors || !errors.length) return "";
  return `<div class="status error">${errors.map((e) =>
    `${escapeHTML(e.url || "")}: ${escapeHTML(e.error || "")}`).join("<br>")}</div>`;
}

function updateIngestAllCount() {
  const allBtn = document.getElementById("ingestAllBtn");
  if (!allBtn) return;
  const n = extractItems.filter((r) => !r.already_cached).length;
  if (n <= 0) allBtn.remove();
  else allBtn.textContent = `Ingest all new (${n})`;
}

async function ingestToStore(item, btn) {
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Caching…";
  try {
    const data = await postJSON("/api/cache/ingest", { items: [item] });
    const it = (data.items || [])[0];
    btn.textContent = it && it.created === false ? "Already cached ✓" : "Cached ✓";
    // Keep extractItems in sync so "Ingest all new" won't re-send this one.
    const idx = btn.dataset ? btn.dataset.ingest : undefined;
    if (idx !== undefined && extractItems[Number(idx)]) {
      extractItems[Number(idx)].already_cached = true;
      updateIngestAllCount();
    }
    setStatus(`Cached "${escapeHTML(item.source_title || item.url || "item")}" — now reusable in the KG and Q&A tabs (no re-fetch).`, "info");
  } catch (e) {
    btn.disabled = false;
    btn.textContent = orig;
    setStatus(escapeHTML(e.message), "error");
  }
}

async function ingestAllNew(btn) {
  const pending = extractItems.map((r, i) => ({ r, i })).filter(({ r }) => !r.already_cached);
  if (!pending.length) return;
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = "Caching…";
  try {
    const items = pending.map(({ r }) => ({ url: r.url, source_title: r.title, text: r.text }));
    const data = await postJSON("/api/cache/ingest", { items });
    const okUrls = new Set((data.items || []).map((it) => it.source_url));
    pending.forEach(({ r, i }) => {
      if (okUrls.has(r.url)) {
        r.already_cached = true;
        const b = resultsEl.querySelector(`[data-ingest="${i}"]`);
        if (b) { b.disabled = true; b.textContent = "Cached ✓"; }
      }
    });
    const errs = (data.errors || []).length;
    setStatus(`Cached ${data.created} new item(s)${data.reused ? `, ${data.reused} already cached` : ""}${errs ? `, ${errs} failed` : ""}. Reusable in the KG and Q&A tabs.`, errs ? "error" : "info");
    updateIngestAllCount();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = orig;
    setStatus(escapeHTML(e.message), "error");
  }
}

function renderExtractResults(payload) {
  const results = payload.results || [];
  const errors = payload.errors || [];
  extractItems = results;
  if (!results.length) {
    resultsEl.innerHTML = `<div class="status">No content could be extracted.</div>` + errBlock(errors);
    return;
  }
  const newCount = results.filter((r) => !r.already_cached).length;
  const header = `<div class="extract-header">
    <span>Extracted ${results.length} page(s)${newCount < results.length ? ` · ${results.length - newCount} already cached` : ""}.</span>
    ${newCount ? `<button class="btn btn-primary btn-sm" id="ingestAllBtn">Ingest all new (${newCount})</button>` : ""}
  </div>`;
  const cards = results.map((r, i) => `
    <article class="result-card" data-idx="${i}">
      <h3><a href="${escapeHTML(r.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(r.title || r.url)}</a></h3>
      <div class="url">${escapeHTML(r.url)}</div>
      <div class="meta">
        <span class="badge">${r.already_cached ? "cached" : "new"}</span>
        <span>${(r.chars || 0).toLocaleString()} chars</span>
      </div>
      <p class="snippet">${escapeHTML(r.snippet || "")}</p>
      <div class="context-actions">
        <button class="btn btn-secondary" data-summarize="${escapeHTML(r.url)}">Summarize this</button>
        <button class="btn btn-primary" data-ingest="${i}" ${r.already_cached ? "disabled" : ""}>${r.already_cached ? "Already cached ✓" : "Ingest to store"}</button>
      </div>
    </article>
  `).join("");
  resultsEl.innerHTML = header + errBlock(errors) + cards;
  resultsEl.querySelectorAll("[data-summarize]").forEach((btn) =>
    btn.addEventListener("click", () => { queryEl.value = btn.dataset.summarize; doSummarize(); }));
  resultsEl.querySelectorAll("[data-ingest]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const r = extractItems[Number(btn.dataset.ingest)];
      ingestToStore({ url: r.url, source_title: r.title, text: r.text }, btn);
    }));
  document.getElementById("ingestAllBtn")?.addEventListener("click", (e) => ingestAllNew(e.currentTarget));
}

async function doExtractAll() {
  const urlList = parseUrls(queryEl.value);
  if (!urlList.length) {
    setStatus("Paste one or more URLs (comma / space separated) to extract.", "error");
    return;
  }
  clearStatus(); resultsEl.innerHTML = ""; setBusy(true);
  setStatus(`<span class="spinner"></span>Fetching ${urlList.length} page(s)…`);
  try {
    const data = await postJSON("/api/cache/extract", { urls: urlList });
    clearStatus();
    renderExtractResults(data);
  } catch (e) {
    setStatus(escapeHTML(e.message), "error");
  } finally {
    setBusy(false);
  }
}

function updateSearchPickCount() {
  const n = resultsEl.querySelectorAll(".search-pick:checked").length;
  const b = document.getElementById("extractSelectedBtn");
  if (b) { b.disabled = n === 0; b.textContent = `Extract & cache selected (${n})`; }
}

// Extract the checked search results and cache them in one action — no copy/paste.
async function extractSelectedSearch(btn) {
  const urls = [...resultsEl.querySelectorAll(".search-pick:checked")].map((c) => c.value);
  if (!urls.length) return;
  setBusy(true);
  setStatus(`<span class="spinner"></span>Extracting ${urls.length} result(s)…`);
  try {
    const data = await postJSON("/api/cache/extract", { urls });
    renderExtractResults(data);            // shows the extracted cards + sets extractItems
    const allBtn = document.getElementById("ingestAllBtn");
    if (allBtn) await ingestAllNew(allBtn); // then cache the new ones automatically
    else setStatus("All selected results were already cached.", "info");
  } catch (e) {
    setStatus(escapeHTML(e.message), "error");
  } finally {
    setBusy(false);
  }
}

function renderSearchResults(data) {
  if (data.kind === "link" && data.results && data.results.length) {
    renderLinkContext(data.results[0]);
    return;
  }
  if (!data.results || data.results.length === 0) {
    resultsEl.innerHTML = `<div class="status">No results found for "${escapeHTML(data.query)}".</div>`;
    return;
  }
  const toolbar = `<div class="search-toolbar">
    <label class="search-pick-all"><input type="checkbox" id="searchPickAll" /> Select all</label>
    <button class="btn btn-primary btn-sm" id="extractSelectedBtn" disabled>Extract & cache selected (0)</button>
  </div>`;
  const cards = data.results.map((r) => `
    <article class="result-card pickable">
      <input type="checkbox" class="search-pick" value="${escapeHTML(r.url)}" ${preservedChecks.has(r.url) ? "checked" : ""} aria-label="Select this result" />
      <div class="result-card-body">
        <h3><a href="${escapeHTML(r.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(r.title || r.url)}</a></h3>
        <div class="url">${escapeHTML(r.url)}</div>
        <p class="snippet">${escapeHTML(r.snippet || "")}</p>
        <button class="btn btn-secondary btn-sm" data-summarize="${escapeHTML(r.url)}">Summarize this</button>
      </div>
    </article>
  `).join("");
  // Offer "Show more" when we got a full page back (more may exist) and we're
  // under the cap.
  const more = (data.results.length >= webN && webN < WEB_MAX)
    ? `<div class="search-more"><button class="btn btn-secondary btn-sm" id="searchMoreBtn" type="button">Show more results</button></div>`
    : "";
  resultsEl.innerHTML = toolbar + cards + more;
  preservedChecks = new Set();   // consumed

  resultsEl.querySelectorAll("[data-summarize]").forEach((btn) => {
    btn.addEventListener("click", () => { queryEl.value = btn.dataset.summarize; doSummarize(); });
  });
  resultsEl.querySelectorAll(".search-pick").forEach((c) =>
    c.addEventListener("change", updateSearchPickCount));
  document.getElementById("searchPickAll")?.addEventListener("change", (e) => {
    resultsEl.querySelectorAll(".search-pick").forEach((c) => { c.checked = e.target.checked; });
    updateSearchPickCount();
  });
  document.getElementById("extractSelectedBtn")?.addEventListener("click", (e) =>
    extractSelectedSearch(e.currentTarget));
  document.getElementById("searchMoreBtn")?.addEventListener("click", () => doSearch(true));
  updateSearchPickCount();   // reflect any restored selections in the button
}

function renderSummary(data) {
  const sourceLine = data.url
    ? `Source: <a href="${escapeHTML(data.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(data.url)}</a>`
    : "";
  const badge = data.engine === "ai"
    ? `AI · ${escapeHTML(data.provider)}${data.model ? " · " + escapeHTML(data.model) : ""}`
    : "Extractive summary";
  resultsEl.innerHTML = `
    <article class="summary-card">
      <div class="meta">
        <span class="badge">${badge}</span>
        <span>${data.chars.toLocaleString()} chars analyzed</span>
        ${sourceLine ? `<span>${sourceLine}</span>` : ""}
      </div>
      <h2>${escapeHTML(data.title)}</h2>
      <div class="summary-body">${escapeHTML(data.summary)}</div>
      <div class="context-actions">
        <button class="btn btn-primary" data-save-summary="1">Save summary to store</button>
      </div>
    </article>
  `;
  // The summary is reusable content: store it as its own cache record (kind
  // 'summary') so it can be vectorized for Q&A or pushed into the KG later,
  // without colliding with the source's own projection (no source_url).
  resultsEl.querySelector("[data-save-summary]")?.addEventListener("click", (e) => {
    const origin = data.url ? data.url : "pasted text";
    ingestToStore({
      kind: "summary",
      source_title: `${data.title} — summary`,
      text: data.summary,
      note: `${badge} summary of ${origin}`,
      tags: ["summary", data.engine === "ai" ? `provider:${data.provider}` : "extractive"],
      origin: "summary",
    }, e.currentTarget);
  });
}

async function loadProviders() {
  try {
    const res = await fetch("/api/providers");
    const data = await res.json();
    providerInfo = data.providers || {};
    updateModelSuggestions();
    updateProviderStatus();
  } catch (e) {
    providerStatusEl.textContent = "Could not load provider info.";
  }
}

function updateModelSuggestions() {
  const p = providerEl.value;
  modelListEl.innerHTML = "";
  if (p === "auto" || p === "extractive") {
    modelEl.value = "";
    modelEl.placeholder = p === "auto" ? "auto" : "(not used)";
    modelEl.disabled = (p === "extractive");
    return;
  }
  modelEl.disabled = false;
  const info = providerInfo[p];
  if (!info) return;
  for (const m of info.models) {
    const opt = document.createElement("option");
    opt.value = m;
    modelListEl.appendChild(opt);
  }
  modelEl.value = info.default_model;
  modelEl.placeholder = info.default_model;
}

function updateProviderStatus() {
  const p = providerEl.value;
  if (p === "extractive") {
    providerStatusEl.textContent = "No API key needed.";
    providerStatusEl.className = "provider-status ok";
    return;
  }
  if (p === "auto") {
    const ready = Object.entries(providerInfo)
      .filter(([, v]) => v.configured).map(([k]) => k);
    if (ready.length) {
      providerStatusEl.textContent = `Ready: ${ready.join(", ")}`;
      providerStatusEl.className = "provider-status ok";
    } else {
      providerStatusEl.textContent = "No keys set — will use extractive.";
      providerStatusEl.className = "provider-status warn";
    }
    return;
  }
  const info = providerInfo[p];
  if (info && info.configured) {
    providerStatusEl.textContent = `${info.env_key} detected.`;
    providerStatusEl.className = "provider-status ok";
  } else if (info) {
    providerStatusEl.textContent = `Set ${info.env_key} to use ${p}.`;
    providerStatusEl.className = "provider-status warn";
  }
}

async function doSearch(more = false) {
  const query = queryEl.value.trim();
  if (!query) { setStatus("Type something to search.", "error"); return; }
  // "Show more" grows the count for the same query and keeps current selections;
  // a fresh query resets both.
  if (more && query === lastWebQuery) {
    webN = Math.min(webN + WEB_STEP, WEB_MAX);
    preservedChecks = new Set([...resultsEl.querySelectorAll(".search-pick:checked")].map((c) => c.value));
  } else {
    webN = WEB_BASE; lastWebQuery = query; preservedChecks = new Set();
  }
  clearStatus(); resultsEl.innerHTML = ""; setBusy(true);
  setStatus(isUrl(query)
    ? `<span class="spinner"></span>Fetching and extracting context from the link…`
    : `<span class="spinner"></span>Searching for "${escapeHTML(query)}"…`);
  try {
    const data = await postJSON("/api/search", { query, max_results: webN });
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
    const data = await postJSON("/api/summarize", {
      input,
      provider: providerEl.value,
      model: modelEl.value.trim(),
    });
    clearStatus();
    renderSummary(data);
  } catch (e) {
    setStatus(escapeHTML(e.message), "error");
  } finally {
    setBusy(false);
  }
}

searchBtn.addEventListener("click", () => doSearch(false));
if (extractBtn) extractBtn.addEventListener("click", doExtractAll);
summarizeBtn.addEventListener("click", doSummarize);
providerEl.addEventListener("change", () => {
  updateModelSuggestions();
  updateProviderStatus();
});

queryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    if (e.shiftKey) doSummarize();
    else doSearch();
  }
});

loadProviders();

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.toggle("active", p.id === `tab-${target}`);
    });
    if (target === "kg" && window.kg) {
      window.kg.refresh && window.kg.refresh();
      window.kg.loadCache && window.kg.loadCache();
    }
    if (target === "cache" && window.cacheView) window.cacheView.refresh();
  });
});

const addKgBtn = $("#addKgBtn");
addKgBtn.addEventListener("click", () => {
  const text = queryEl.value.trim();
  if (!text) {
    setStatus("Type or paste a chunk of text first, then click + KG.", "error");
    return;
  }
  const summaryCard = resultsEl.querySelector(".summary-card");
  const title = summaryCard?.querySelector("h2")?.textContent || "";
  const link = summaryCard?.querySelector(".meta a")?.href || "";
  window.kg.openModal({ text, source_title: title, source_url: link });
});

const selectionMenu = $("#selectionMenu");
const saveSelectionBtn = $("#saveSelection");
let pendingSelection = null;

function hideSelectionMenu() {
  selectionMenu.classList.add("hidden");
  pendingSelection = null;
}

function showSelectionMenu(selection) {
  const range = selection.getRangeAt(0);
  const rect = range.getBoundingClientRect();
  const top = window.scrollY + rect.top - 40;
  const left = window.scrollX + rect.left + rect.width / 2 - 72;
  selectionMenu.style.top = `${Math.max(top, window.scrollY + 8)}px`;
  selectionMenu.style.left = `${Math.max(left, 8)}px`;
  selectionMenu.classList.remove("hidden");
}

document.addEventListener("mouseup", () => {
  setTimeout(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return hideSelectionMenu();
    const text = sel.toString().trim();
    if (text.length < 20) return hideSelectionMenu();
    const anchor = sel.anchorNode;
    if (!anchor || !resultsEl.contains(anchor)) return hideSelectionMenu();
    pendingSelection = text;
    showSelectionMenu(sel);
  }, 0);
});

document.addEventListener("mousedown", (e) => {
  if (!selectionMenu.contains(e.target)) hideSelectionMenu();
});

saveSelectionBtn.addEventListener("click", () => {
  if (!pendingSelection) return;
  const summaryCard = resultsEl.querySelector(".summary-card");
  const title = summaryCard?.querySelector("h2")?.textContent || "";
  const link = summaryCard?.querySelector(".meta a")?.href || "";
  window.kg.openModal({ text: pendingSelection, source_title: title, source_url: link });
  hideSelectionMenu();
});

// --- Theme toggle (light / dark, persisted per browser) ---------------------
const themeToggle = $("#themeToggle");
if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("wiki-theme", next); } catch (e) { /* ignore */ }
  });
}
