(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s = "") => String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  const recallInput = $("#memRecallInput");
  const recallBtn = $("#memRecallBtn");
  const recallOut = $("#memRecallOut");
  const addBtn = $("#memAddBtn");
  const addProg = $("#memAddProg");
  const listEl = $("#memList");
  const filterEl = $("#memFilter");
  const statusEl = $("#memStatus");
  const tabCount = $("#memTabCount");

  async function postJSON(url, body, method = "POST") {
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  function applyStats(s) {
    if (!s) return;
    const bk = s.by_kind || {};
    $("#memTotal").textContent = s.total ?? 0;
    $("#memFacts").textContent = bk.fact ?? 0;
    $("#memAnswers").textContent = bk.answer ?? 0;
    $("#memPrefs").textContent = bk.preference ?? 0;
    $("#memGaps").textContent = bk.gap ?? 0;
    $("#memCorrections").textContent = bk.correction ?? 0;
    $("#memUses").textContent = s.uses ?? 0;
    if (s.total) { tabCount.textContent = s.total; tabCount.classList.remove("hidden"); }
    else { tabCount.classList.add("hidden"); }
    const recall = s.embeddings ? "semantic recall" : "keyword recall (set OPENAI_API_KEY for semantic)";
    statusEl.textContent = `${s.embedded ?? 0}/${s.total ?? 0} embedded · ${recall}`;
    statusEl.className = "provider-status " + (s.embeddings ? "ok" : "warn");
  }

  function memItem(m) {
    const tags = (m.tags || []).map((t) => `<span class="kg-tag">${esc(t)}</span>`).join("");
    return `<div class="kg-item" data-id="${esc(m.id)}">
      <div class="kg-preview">${esc(m.text || m.preview || "")}</div>
      <div class="kg-meta">
        <span class="mem-badge mem-${esc(m.kind)}">${esc(m.kind)}</span>
        <span class="kg-tag">salience ${esc(String(m.salience ?? 3))}</span>
        <span class="kg-tag">used ×${esc(String(m.use_count ?? 0))}</span>
        ${m.confidence ? `<span class="kg-tag">${esc(m.confidence)}</span>` : ""}
        ${tags}
        <span class="kg-tag mem-act" data-rate="up" title="Reinforce">👍</span>
        <span class="kg-tag mem-act" data-rate="down" title="Demote">👎</span>
        <span class="kg-tag danger" data-del title="Forget">forget</span>
      </div>
    </div>`;
  }

  async function loadList() {
    try {
      const kind = filterEl.value;
      const data = await fetch(`/api/memory/list${kind ? `?kind=${encodeURIComponent(kind)}` : ""}`)
        .then((r) => r.json());
      applyStats(data.stats);
      const rows = data.memories || [];
      listEl.innerHTML = rows.length
        ? rows.map(memItem).join("")
        : `<div class="empty">No memories yet. Ask questions, run Maintain, or add one above.</div>`;
    } catch (e) {
      listEl.innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    }
  }

  async function loadStats() {
    try { applyStats(await fetch("/api/memory/stats").then((r) => r.json())); }
    catch (e) { /* ignore */ }
  }

  async function doRecall() {
    const q = recallInput.value.trim();
    if (!q) return;
    recallOut.classList.remove("hidden");
    recallOut.innerHTML = `<div class="ask-thinking"><span class="spinner"></span>Recalling…</div>`;
    try {
      const data = await postJSON("/api/memory/recall", { query: q, k: 8 });
      const rows = data.memories || [];
      if (!rows.length) { recallOut.innerHTML = `<div class="ask-note">No relevant memories yet.</div>`; return; }
      recallOut.innerHTML = `<h4>Recalled (${rows.length})</h4>` + rows.map((m) =>
        `<div class="cite"><span class="mem-badge mem-${esc(m.kind)}">${esc(m.kind)}</span> ` +
        `<span style="color:var(--muted)">score ${esc(String(m.score))} · sim ${esc(String(m.similarity))}</span>` +
        `<div class="cite-prev">${esc(m.text || "")}</div></div>`).join("");
    } catch (e) {
      recallOut.innerHTML = `<div style="color:var(--error)">${esc(e.message)}</div>`;
    }
  }

  async function doAdd() {
    const text = $("#memText").value.trim();
    if (!text) { addProg.textContent = "Write something to remember."; addProg.className = "ingest-progress error"; return; }
    addBtn.disabled = true; const prev = addBtn.textContent; addBtn.textContent = "Adding…";
    try {
      await postJSON("/api/memory/add", {
        text,
        kind: $("#memKind").value,
        salience: parseInt($("#memSalience").value, 10) || 3,
        tags: $("#memTags").value,
      });
      $("#memText").value = ""; $("#memTags").value = "";
      addProg.className = "ingest-progress ok"; addProg.textContent = "Saved.";
      await loadList();
    } catch (e) {
      addProg.className = "ingest-progress error"; addProg.textContent = e.message;
    } finally { addBtn.disabled = false; addBtn.textContent = prev; }
  }

  listEl.addEventListener("click", async (e) => {
    const item = e.target.closest(".kg-item");
    if (!item) return;
    const id = item.dataset.id;
    try {
      if (e.target.hasAttribute("data-del")) {
        await postJSON(`/api/memory/${encodeURIComponent(id)}`, null, "DELETE");
        await loadList();
      } else if (e.target.hasAttribute("data-rate")) {
        await postJSON("/api/memory/feedback", { memory_id: id, rating: e.target.dataset.rate });
        await loadList();
      }
    } catch (err) { /* ignore transient errors */ }
  });

  recallBtn.addEventListener("click", doRecall);
  recallInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); doRecall(); } });
  addBtn.addEventListener("click", doAdd);
  $("#memRefresh").addEventListener("click", loadList);
  filterEl.addEventListener("change", loadList);

  document.querySelectorAll('.tab[data-tab="memory"]').forEach((t) =>
    t.addEventListener("click", loadList));

  loadStats();
  loadList();
})();
