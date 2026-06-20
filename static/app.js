const $ = (sel) => document.querySelector(sel);
const queryEl = $("#query");
const searchBtn = $("#searchBtn");
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
        <button class="btn btn-accent" data-addkg-context="1">+ KG</button>
      </div>
    </article>
  `;
  resultsEl.querySelector("[data-summarize]")?.addEventListener("click", () => {
    queryEl.value = r.url;
    doSummarize();
  });
  resultsEl.querySelector("[data-addkg-context]")?.addEventListener("click", () => {
    window.kg.openModal({ text: context, source_title: r.title || "", source_url: r.url || "" });
  });
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
    </article>
  `;
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

async function doSearch() {
  const query = queryEl.value.trim();
  if (!query) { setStatus("Type something to search.", "error"); return; }
  clearStatus(); resultsEl.innerHTML = ""; setBusy(true);
  setStatus(isUrl(query)
    ? `<span class="spinner"></span>Fetching and extracting context from the link…`
    : `<span class="spinner"></span>Searching for "${escapeHTML(query)}"…`);
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

searchBtn.addEventListener("click", doSearch);
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
    if (target === "kg" && window.kg && window.kg.refresh) window.kg.refresh();
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
