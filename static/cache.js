// Cache tab: browse + manage the canonical cached-content store.
// Lists every cached item with status badges (in KG / vectorized / drifted),
// previews raw text, and deletes items (cascading the KG + vector cleanup the
// /api/cache/item DELETE route performs). Read-only consumers elsewhere; this is
// the one place to see the whole store.
(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s = "") => String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  const listEl = $("#cacheBrowserList");
  const detailEl = $("#cacheDetail");
  const statsEl = $("#cacheStats");
  const searchEl = $("#cacheSearch");
  const tabCount = $("#cacheTabCount");
  if (!listEl) return;

  let items = [];           // last-loaded items (unfiltered)
  let openId = null;        // currently previewed item

  function badges(it) {
    const out = [];
    if (it.in_kg) out.push('<span class="badge cache-b-kg">KG</span>');
    if (it.vectorized) out.push('<span class="badge cache-b-vec">Q&amp;A</span>');
    if (it.kg_stale || it.vector_stale) out.push('<span class="badge cache-b-drift">drifted</span>');
    if (!it.in_kg && !it.vectorized) out.push('<span class="badge cache-b-new">unused</span>');
    return out.join(" ");
  }

  function render() {
    const q = (searchEl.value || "").trim().toLowerCase();
    const rows = items.filter((it) =>
      !q || (it.source_title || "").toLowerCase().includes(q)
         || (it.source_url || "").toLowerCase().includes(q));
    if (!rows.length) {
      listEl.innerHTML = `<div class="cache-empty">${items.length
        ? "No items match the filter."
        : "Nothing cached yet — extract some URLs on the Read tab."}</div>`;
      return;
    }
    listEl.innerHTML = rows.map((it) => `
      <div class="cache-item" data-id="${esc(it.id)}">
        <div class="cache-item-main">
          <div class="cache-item-title">${esc(it.source_title || it.source_url || it.id)}</div>
          <div class="cache-item-meta">
            ${(it.chars || 0).toLocaleString()} chars · ${esc(it.kind || "?")}${it.origin ? " · " + esc(it.origin) : ""}${it.source_url ? " · " + esc(it.source_url) : ""}
          </div>
        </div>
        <div class="cache-item-badges">${badges(it)}</div>
        <div class="cache-item-actions">
          <button class="btn btn-secondary btn-sm" data-view="${esc(it.id)}">View</button>
          <button class="btn btn-sm cache-del" data-del="${esc(it.id)}">Delete</button>
        </div>
      </div>`).join("");

    listEl.querySelectorAll("[data-view]").forEach((b) =>
      b.addEventListener("click", () => view(b.dataset.view)));
    listEl.querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", () => del(b.dataset.del)));
  }

  async function refresh() {
    listEl.innerHTML = '<div class="cache-empty">Loading…</div>';
    detailEl.classList.add("hidden");
    try {
      const data = await fetch("/api/cache/items").then((r) => r.json());
      items = data.items || [];
      const c = data.counts || {};
      statsEl.textContent = `${c.total ?? items.length} items · ${c.in_kg ?? 0} in KG · ${c.vectorized ?? 0} vectorized · ${data.backend || "local"}`;
      statsEl.className = "provider-status ok";
      if (tabCount) {
        tabCount.textContent = c.total ?? items.length;
        tabCount.classList.toggle("hidden", !(c.total ?? items.length));
      }
      render();
    } catch (e) {
      listEl.innerHTML = `<div class="cache-empty">Could not load cache: ${esc(e.message)}</div>`;
    }
  }

  async function view(id) {
    if (openId === id) {  // toggle closed
      detailEl.classList.add("hidden");
      openId = null;
      return;
    }
    openId = id;
    detailEl.classList.remove("hidden");
    detailEl.innerHTML = '<div class="cache-empty">Loading…</div>';
    try {
      const rec = await fetch("/api/cache/item/" + encodeURIComponent(id)).then((r) => r.json());
      if (rec.error) throw new Error(rec.error);
      const src = rec.source_url
        ? `<a href="${esc(rec.source_url)}" target="_blank" rel="noopener noreferrer">${esc(rec.source_url)}</a>`
        : `<em>${esc(rec.kind)} (no URL)</em>`;
      detailEl.innerHTML = `
        <div class="cache-detail-head">
          <h3>${esc(rec.source_title || rec.id)}</h3>
          <button class="btn btn-secondary btn-sm" id="cacheDetailClose">Close</button>
        </div>
        <div class="cache-item-meta">${badges(rec)} · ${(rec.chars || 0).toLocaleString()} chars · ${src}</div>
        ${rec.note ? `<div class="cache-item-meta">${esc(rec.note)}</div>` : ""}
        <pre class="cache-raw">${esc(rec.raw_text || "")}</pre>`;
      $("#cacheDetailClose").addEventListener("click", () => {
        detailEl.classList.add("hidden"); openId = null;
      });
    } catch (e) {
      detailEl.innerHTML = `<div class="cache-empty">Could not load item: ${esc(e.message)}</div>`;
    }
  }

  async function del(id) {
    const it = items.find((x) => x.id === id);
    const label = it ? (it.source_title || it.source_url || id) : id;
    const warn = it && (it.in_kg || it.vectorized)
      ? "\n\nThis also removes its KG chunks and Q&A vectors." : "";
    if (!window.confirm(`Delete "${label}" from the cache?${warn}`)) return;
    try {
      const res = await fetch("/api/cache/item/" + encodeURIComponent(id), { method: "DELETE" })
        .then((r) => r.json());
      const c = res.cleanup || {};
      statsEl.textContent = `Deleted${(c.kg_chunks_removed || c.vectors_removed)
        ? ` (removed ${c.kg_chunks_removed || 0} KG chunks, ${c.vectors_removed || 0} vectors)` : ""}.`;
      statsEl.className = "provider-status";
      if (openId === id) { detailEl.classList.add("hidden"); openId = null; }
      await refresh();
      // KG/Q&A stats may have changed if projections were removed.
      if (window.kg && window.kg.refresh) window.kg.refresh();
    } catch (e) {
      statsEl.textContent = "Delete failed: " + e.message;
      statsEl.className = "provider-status warn";
    }
  }

  searchEl.addEventListener("input", render);
  $("#cacheRefresh").addEventListener("click", refresh);

  window.cacheView = { refresh };
})();
