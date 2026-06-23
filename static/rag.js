(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s = "") => String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  // tiny markdown (shared shape with kg.js)
  function mdLite(src = "") {
    const lines = esc(src).split("\n");
    const out = []; let inList = false;
    const inline = (t) => t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(\d+)\]/g, '<span class="cite-id">[$1]</span>');
    for (const line of lines) {
      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { if (inList) { out.push("</ul>"); inList = false; } out.push(`<h4>${inline(h[2])}</h4>`); continue; }
      const li = line.match(/^\s*[-*]\s+(.*)$/);
      if (li) { if (!inList) { out.push("<ul>"); inList = true; } out.push(`<li>${inline(li[1])}</li>`); continue; }
      if (inList) { out.push("</ul>"); inList = false; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (inList) out.push("</ul>");
    return out.join("");
  }

  const urls = $("#ragUrls");
  const ingestBtn = $("#ragIngestBtn");
  const ingestProg = $("#ragIngestProgress");
  const question = $("#ragQuestion");
  const askBtn = $("#ragAskBtn");
  const searchBtn = $("#ragSearchBtn");
  const answer = $("#ragAnswer");
  const mmrToggle = $("#ragMmr");
  const statusEl = $("#ragStatus");

  function providerModel() {
    return {
      provider: document.querySelector("#provider")?.value || "auto",
      model: document.querySelector("#model")?.value?.trim() || "",
    };
  }
  function show(html) { answer.classList.remove("hidden"); answer.innerHTML = html; }

  async function loadStats() {
    try {
      const s = await fetch("/api/rag/stats").then((r) => r.json());
      $("#ragDocs").textContent = s.documents ?? 0;
      $("#ragSections").textContent = s.sections ?? 0;
      $("#ragChunks").textContent = s.chunks ?? 0;
      $("#ragIndex").textContent = (s.index || "—") + (s.embed_model ? ` · ${s.embed_model}` : "");
    } catch (e) { /* ignore */ }
  }

  async function loadStatus() {
    try {
      const s = await fetch("/api/agent/status").then((r) => r.json());
      const trace = s.tracing && s.tracing.enabled ? ` · tracing → ${s.tracing.project}` : "";
      if (!s.provider_ready) { statusEl.textContent = "Set an API key to ingest/ask"; statusEl.className = "provider-status warn"; }
      else { statusEl.textContent = `Ready: ${s.provider_ready}${trace}`; statusEl.className = "provider-status ok"; }
    } catch (e) { /* ignore */ }
  }

  // Both flows write to #ragIngestProgress, so only one may run at a time.
  function lockIngest(on) {
    ingestBtn.disabled = on;
    if (cacheVectorizeBtn) cacheVectorizeBtn.disabled = on;
  }

  ingestBtn.addEventListener("click", async () => {
    const list = urls.value.split("\n").map((u) => u.trim()).filter((u) => u.startsWith("http"));
    if (!list.length) { ingestProg.textContent = "Paste one or more URLs."; ingestProg.className = "ingest-progress error"; return; }
    lockIngest(true); const prev = ingestBtn.textContent; ingestBtn.textContent = "Ingesting…";
    ingestProg.className = "ingest-progress"; ingestProg.textContent = `Ingesting ${list.length} page(s)… this can take a few minutes.`;
    try {
      const res = await fetch("/api/rag/ingest", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: list, ...providerModel() }),
      }).then((r) => r.json());
      if (res.error) throw new Error(res.error);
      const ok = (res.results || []).length, errs = (res.errors || []).length;
      ingestProg.className = "ingest-progress ok";
      ingestProg.textContent = `Ingested ${ok} page(s)${errs ? `, ${errs} failed` : ""}. ` +
        `${res.vector_stats.chunks} chunks across ${res.vector_stats.sections} sections.`;
      await loadStats();
      if (window.kg && window.kg.refresh) window.kg.refresh();
    } catch (e) {
      ingestProg.className = "ingest-progress error"; ingestProg.textContent = e.message;
    } finally { lockIngest(false); ingestBtn.textContent = prev; }
  });

  // --- Vectorize already-cached content (no re-fetch) -----------------------
  const cacheList = $("#ragCacheList");
  const cacheAll = $("#ragCacheAll");
  const cacheVectorizeBtn = $("#ragCacheVectorizeBtn");

  function renderCachePicker(items) {
    if (!items.length) {
      cacheList.innerHTML = '<div class="cache-empty">No cached items available. Extract some URLs on the Read tab first.</div>';
      return;
    }
    cacheList.innerHTML = items.map((it) => `
      <label class="cache-row">
        <input type="checkbox" value="${esc(it.id)}" />
        <span class="cache-row-main">
          <span class="cache-row-title">${esc(it.source_title || it.source_url || it.id)}${it.vector_stale ? ' <span class="badge">drifted</span>' : ""}</span>
          <span class="cache-row-meta">${(it.chars || 0).toLocaleString()} chars${it.source_url ? " · " + esc(it.source_url) : ""}</span>
        </span>
      </label>
    `).join("");
  }

  async function loadRagCache() {
    if (!cacheList) return;
    cacheList.innerHTML = '<div class="cache-empty">Loading…</div>';
    if (cacheAll) cacheAll.checked = false;
    try {
      const data = await fetch("/api/cache/items?needs_vectors=true").then((r) => r.json());
      renderCachePicker(data.items || []);
    } catch (e) {
      cacheList.innerHTML = `<div class="cache-empty">Could not load cache: ${esc(e.message)}</div>`;
    }
  }

  if (cacheVectorizeBtn) cacheVectorizeBtn.addEventListener("click", async () => {
    const ids = [...cacheList.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value);
    if (!ids.length) { ingestProg.textContent = "Select at least one cached item."; ingestProg.className = "ingest-progress error"; return; }
    lockIngest(true); const prev = cacheVectorizeBtn.textContent; cacheVectorizeBtn.textContent = "Vectorizing…";
    ingestProg.className = "ingest-progress"; ingestProg.textContent = `Vectorizing ${ids.length} cached item(s)… this can take a few minutes.`;
    try {
      const res = await fetch("/api/cache/to-vectors", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_ids: ids, ...providerModel() }),
      }).then((r) => r.json());
      if (res.error) throw new Error(res.error);
      const ok = (res.results || []).length, errs = (res.errors || []).length;
      ingestProg.className = "ingest-progress ok";
      ingestProg.textContent = `Vectorized ${ok} item(s)${errs ? `, ${errs} failed` : ""}. ` +
        `${res.vector_stats.chunks} chunks across ${res.vector_stats.sections} sections.`;
      await loadStats();
      await loadRagCache();
      if (window.kg && window.kg.refresh) window.kg.refresh();
    } catch (e) {
      ingestProg.className = "ingest-progress error"; ingestProg.textContent = e.message;
    } finally { lockIngest(false); cacheVectorizeBtn.textContent = prev; }
  });

  if (cacheAll) cacheAll.addEventListener("change", () => {
    cacheList.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = cacheAll.checked; });
  });
  $("#ragCacheRefresh")?.addEventListener("click", loadRagCache);

  // --- Focus: scope the answer to specific cached docs (drives ICL regimes) --
  const focusList = $("#ragFocusList");
  const focusAll = $("#ragFocusAll");

  function renderFocusPicker(items) {
    if (!items.length) {
      focusList.innerHTML = '<div class="cache-empty">No cached items yet — add some on the Read tab.</div>';
      return;
    }
    focusList.innerHTML = items.map((it) => `
      <label class="cache-row">
        <input type="checkbox" value="${esc(it.id)}" />
        <span class="cache-row-main">
          <span class="cache-row-title">${esc(it.source_title || it.source_url || it.id)}${it.vectorized ? ' <span class="badge cache-b-vec">Q&amp;A</span>' : ""}</span>
          <span class="cache-row-meta">${(it.chars || 0).toLocaleString()} chars${it.source_url ? " · " + esc(it.source_url) : ""}</span>
        </span>
      </label>`).join("");
  }

  async function loadFocusList() {
    if (!focusList) return;
    focusList.innerHTML = '<div class="cache-empty">Loading…</div>';
    if (focusAll) focusAll.checked = false;
    try {
      const data = await fetch("/api/cache/items").then((r) => r.json());
      renderFocusPicker(data.items || []);
    } catch (e) {
      focusList.innerHTML = `<div class="cache-empty">Could not load: ${esc(e.message)}</div>`;
    }
  }

  function selectedFocusIds() {
    return focusList ? [...focusList.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value) : [];
  }

  // Plain "where the answer came from" — the internal strategy (full docs vs
  // sections vs library retrieval) is an implementation detail, not shown.
  function sourceNote(regime, scope) {
    scope = scope || {};
    if (regime && regime.indexOf("icl") === 0) {
      const n = scope.documents || 0;
      return n ? `from ${n} selected document${n > 1 ? "s" : ""}` : "from your selection";
    }
    if (regime === "rag") {
      const n = scope.documents || scope.passages || 0;
      return n ? `from ${n} source${n > 1 ? "s" : ""} in your library` : "from your library";
    }
    return "";
  }

  if (focusAll) focusAll.addEventListener("change", () => {
    focusList.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = focusAll.checked; });
  });
  $("#ragFocusRefresh")?.addEventListener("click", loadFocusList);

  async function doAsk() {
    const q = question.value.trim(); if (!q) return;
    askBtn.disabled = true; const prev = askBtn.textContent; askBtn.textContent = "Thinking…";
    show(`<div class="ask-thinking"><span class="spinner"></span>Retrieving and answering…</div>`);
    try {
      const res = await fetch("/api/rag/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, mmr: mmrToggle.checked,
          rerank: document.querySelector("#ragRerank")?.checked || false,
          item_ids: selectedFocusIds(), ...providerModel() }),
      }).then((r) => r.json());
      if (res.error) throw new Error(res.error);
      let html = `<div class="ask-body">${mdLite(res.answer || "")}</div>`;
      if (res.citations && res.citations.length) {
        html += `<div class="ask-citations"><h4>Sources</h4>` + res.citations.map((c) =>
          `<div class="cite"><span class="cite-id">[${c.n}]</span> ` +
          `<a href="${esc(c.url)}" target="_blank" rel="noopener noreferrer">${esc(c.title || c.url)}</a> ` +
          `<span style="color:var(--muted)">· ${esc(c.date || "n/a")} · score ${esc(String(c.score))}</span>` +
          (c.preview ? `<div class="cite-prev">${esc(c.preview)}</div>` : "") + `</div>`).join("") + `</div>`;
      }
      const used = [res.provider, res.model].filter(Boolean).join(" · ");
      const memN = (res.memories_used || []).length;
      const memNote = memN ? ` · recalled ${memN} ${memN === 1 ? "memory" : "memories"}` : "";
      const note = sourceNote(res.regime, res.scope);
      if (used) html += `<div class="ask-foot">Answered by ${esc(used)}${note ? " · " + esc(note) : ""}${memNote}</div>`;
      show(html);
    } catch (e) { show(`<div style="color:var(--error)">${esc(e.message)}</div>`); }
    finally { askBtn.disabled = false; askBtn.textContent = prev; }
  }

  const SEARCH_BASE = 10, SEARCH_STEP = 10, SEARCH_MAX = 50;  // initial / per-click / cap (matches server)
  let searchK = SEARCH_BASE, lastSearchQ = "";

  async function doSearch(more = false) {
    const q = question.value.trim(); if (!q) return;
    // A fresh query resets the count; "Show more" grows it for the same query.
    if (more && q === lastSearchQ) searchK += SEARCH_STEP;
    else { searchK = SEARCH_BASE; lastSearchQ = q; }
    searchBtn.disabled = true; const prev = searchBtn.textContent; searchBtn.textContent = "Searching…";
    show(`<div class="ask-thinking"><span class="spinner"></span>Hierarchical retrieval…</div>`);
    try {
      const res = await fetch("/api/rag/search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k: searchK, mmr: mmrToggle.checked,
          rerank: document.querySelector("#ragRerank")?.checked || false }),
      }).then((r) => r.json());
      if (res.error) throw new Error(res.error);
      if (!res.hits || !res.hits.length) { show(`<div class="ask-note">No matches yet — ingest some pages first.</div>`); return; }
      let html = `<div class="ask-citations"><h4>Top passages (${res.hits.length})</h4>` + res.hits.map((h, i) =>
        `<div class="cite"><span class="cite-id">[${i + 1}]</span> ` +
        `<a href="${esc(h.url)}" target="_blank" rel="noopener noreferrer">${esc(h.title || h.url)}</a> ` +
        `<span style="color:var(--muted)">· ${esc(h.date || "n/a")} · score ${esc(String(h.score))} (chunk ${esc(String(h.chunk_score))} / sec ${esc(String(h.section_score))})</span>` +
        `<div class="cite-prev"><em>${esc(h.contextual_summary || "")}</em><br>${esc(h.preview || h.text || "")}</div>` +
        ((h.graph && h.graph.entities && h.graph.entities.length) ? `<div class="cite-prev">entities: ${esc(h.graph.entities.join(", "))}</div>` : "") +
        `</div>`).join("") + `</div>`;
      // Offer "Show more" only when we got a full page back (more may exist)
      // and we haven't hit the cap.
      if (res.hits.length >= searchK && searchK < SEARCH_MAX) {
        html += `<div class="ask-more"><button id="ragSearchMore" class="btn btn-secondary btn-sm" type="button">Show more results</button></div>`;
      }
      show(html);
      document.querySelector("#ragSearchMore")?.addEventListener("click", () => doSearch(true));
    } catch (e) { show(`<div style="color:var(--error)">${esc(e.message)}</div>`); }
    finally { searchBtn.disabled = false; searchBtn.textContent = prev; }
  }

  askBtn.addEventListener("click", doAsk);
  searchBtn.addEventListener("click", () => doSearch(false));
  question.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); doAsk(); } });

  // Refresh stats + the cache picker when the Library tab is opened.
  document.querySelectorAll('.tab[data-tab="rag"]').forEach((t) =>
    t.addEventListener("click", () => { loadStats(); loadStatus(); loadRagCache(); loadFocusList(); }));

  loadStats(); loadStatus(); loadRagCache(); loadFocusList();
})();
