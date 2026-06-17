(function () {
  const $ = (sel) => document.querySelector(sel);

  const modal = $("#kgModal");
  const mText = $("#mText");
  const mTitle = $("#mTitle");
  const mUrl = $("#mUrl");
  const mTags = $("#mTags");
  const mNote = $("#mNote");
  const mSave = $("#mSave");

  const currentList = $("#currentList");
  const overallList = $("#overallList");
  const kgGraphEl = $("#kgGraph");
  const kgLegendEl = $("#kgLegend");
  const kgDetail = $("#kgDetail");
  const tabCount = $("#tabCount");
  const integrateBtn = $("#integrateBtn");
  const kgQueryEl = $("#kgQuery");
  const kgSearchBtn = $("#kgSearchBtn");
  const kgRefreshBtn = $("#kgRefresh");

  // Agent panel
  const askInput = $("#askInput");
  const askBtn = $("#askBtn");
  const maintainBtn = $("#maintainBtn");
  const askAnswer = $("#askAnswer");
  const agentStatusEl = $("#agentStatus");

  // Layer metadata: color + label, ordered from evidence up to synthesis.
  const LAYERS = {
    source:    { label: "Source",    color: "#34d399" },
    chunk:     { label: "Chunk",     color: "#5b9dff" },
    entity:    { label: "Entity",    color: "#7c5bff" },
    topic:     { label: "Topic",     color: "#fbbf24" },
    synthesis: { label: "Synthesis", color: "#f472b6" },
  };
  const enabledLayers = new Set(Object.keys(LAYERS));
  let lastGraph = { nodes: [], edges: [] };
  let agentReady = false;

  function esc(s = "") {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // Minimal, safe markdown -> HTML for agent answers/reports.
  function mdLite(src = "") {
    const lines = esc(src).split("\n");
    const out = [];
    let inList = false;
    const inline = (t) =>
      t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
       .replace(/`([^`]+)`/g, "<code>$1</code>");
    for (let line of lines) {
      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        if (inList) { out.push("</ul>"); inList = false; }
        out.push(`<h4>${inline(h[2])}</h4>`);
        continue;
      }
      const li = line.match(/^\s*[-*]\s+(.*)$/);
      if (li) {
        if (!inList) { out.push("<ul>"); inList = true; }
        out.push(`<li>${inline(li[1])}</li>`);
        continue;
      }
      if (inList) { out.push("</ul>"); inList = false; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (inList) out.push("</ul>");
    return out.join("");
  }

  function openModal({ text = "", source_title = "", source_url = "" } = {}) {
    mText.value = text;
    mTitle.value = source_title;
    mUrl.value = source_url;
    mTags.value = "";
    mNote.value = "";
    modal.classList.remove("hidden");
    mTags.focus();
  }

  function closeModal() { modal.classList.add("hidden"); }

  modal.querySelectorAll("[data-close]").forEach((el) =>
    el.addEventListener("click", closeModal)
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) closeModal();
  });

  mSave.addEventListener("click", async () => {
    const text = mText.value.trim();
    if (!text) return;
    mSave.disabled = true;
    mSave.textContent = "Saving…";
    try {
      const res = await fetch("/api/kg/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          source_title: mTitle.value.trim(),
          source_url: mUrl.value.trim(),
          tags: mTags.value.split(",").map((t) => t.trim()).filter(Boolean),
          note: mNote.value.trim(),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Save failed (${res.status})`);
      closeModal();
      await refresh();
      flash(`Saved to staging (${data.stats.current.chunks} chunks ready to integrate).`);
    } catch (e) {
      alert(e.message);
    } finally {
      mSave.disabled = false;
      mSave.textContent = "Save to staging";
    }
  });

  function flash(msg, kind = "info") {
    kgDetail.classList.remove("empty");
    const color = kind === "error" ? "var(--error)" : "var(--success)";
    kgDetail.innerHTML = `<div style="color:${color}">${esc(msg)}</div>`;
  }

  async function refresh() {
    try {
      const [statsRes, current, overall] = await Promise.all([
        fetch("/api/kg/stats").then((r) => r.json()),
        fetch("/api/kg/graph?where=current").then((r) => r.json()),
        fetch("/api/kg/graph?where=overall").then((r) => r.json()),
      ]);
      renderStats(statsRes);
      renderList(currentList, current.nodes.filter((n) => n.type === "chunk"), "current", "Nothing in staging yet. Highlight text in a summary or paste into the input and click + KG.");
      const overallChunks = overall.nodes.filter((n) => n.type === "chunk").slice().reverse().slice(0, 20);
      renderList(overallList, overallChunks, "overall", "Nothing integrated yet. Add chunks to staging and click Integrate.");
      lastGraph = overall;
      renderGraph(overall);
      const totalStaged = statsRes.current.chunks;
      if (totalStaged > 0) {
        tabCount.textContent = totalStaged;
        tabCount.classList.remove("hidden");
      } else {
        tabCount.classList.add("hidden");
      }
    } catch (e) {
      console.error(e);
    }
  }

  function renderStats(stats) {
    const o = stats.overall || {};
    const setText = (id, v) => { const el = $(id); if (el) el.textContent = v ?? 0; };
    setText("#overallSources", o.sources);
    setText("#currentChunks", stats.current ? stats.current.chunks : 0);
    setText("#overallChunks", o.chunks);
    setText("#overallEntities", o.entities);
    setText("#overallTopics", o.topics);
    setText("#overallSyntheses", o.syntheses);
    setText("#overallEdges", o.edges);
  }

  function renderList(container, nodes, where, emptyMsg) {
    if (!nodes.length) {
      container.innerHTML = `<div class="empty">${esc(emptyMsg)}</div>`;
      return;
    }
    container.innerHTML = nodes.map((n) => `
      <div class="kg-item" data-id="${esc(n.id)}" data-where="${where}">
        <div class="kg-preview">${esc(n.preview || n.text || "")}</div>
        <div class="kg-meta">
          ${n.source_title ? `<span>${esc(n.source_title)}</span>` : ""}
          ${(n.tags || []).map((t) => `<span class="kg-tag">${esc(t)}</span>`).join("")}
          ${n.created_at ? `<span>${esc(new Date(n.created_at).toLocaleString())}</span>` : ""}
          <span class="kg-tag danger" data-action="remove">remove</span>
        </div>
      </div>
    `).join("");

    container.querySelectorAll(".kg-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.dataset.action === "remove") {
          e.stopPropagation();
          removeNode(el.dataset.id, el.dataset.where);
          return;
        }
        showDetailById(el.dataset.id, el.dataset.where);
      });
    });
  }

  async function removeNode(id, where) {
    if (!confirm("Remove this node?")) return;
    await fetch(`/api/kg/node/${encodeURIComponent(id)}?where=${where}`, { method: "DELETE" });
    await refresh();
  }

  async function showDetailById(id, where) {
    const g = await fetch(`/api/kg/graph?where=${where}`).then((r) => r.json());
    const node = g.nodes.find((n) => n.id === id);
    if (!node) return;
    renderDetail(node, g);
  }

  function renderDetail(node, graph) {
    kgDetail.classList.remove("empty");
    const layer = LAYERS[node.type];
    const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));

    // Group edges by semantic kind for a readable profile.
    const groups = {};
    for (const e of graph.edges) {
      if (e.from !== node.id && e.to !== node.id) continue;
      const outward = e.from === node.id;
      const other = byId[outward ? e.to : e.from];
      if (!other) continue;
      const key = e.label || e.kind || "linked";
      const arrow = outward ? "→" : "←";
      (groups[key] = groups[key] || []).push(
        `<span class="kg-tag">${arrow} ${esc(other.name || other.title || (other.preview || "").slice(0, 40) || other.id)}` +
        `${e.confidence && e.kind === "relation" ? ` <em style="opacity:.6">· ${esc(e.confidence)}</em>` : ""}</span>`
      );
    }

    const parts = [];
    const title = node.name || node.title || node.source_title || node.id;
    parts.push(
      `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">` +
      (layer ? `<span class="layer-dot" style="background:${layer.color}"></span>` : "") +
      `<strong style="font-size:1rem">${esc(title)}</strong>` +
      (layer ? `<span class="kg-tag">${layer.label}</span>` : "") +
      (node.kind ? `<span class="kg-tag">${esc(node.kind)}</span>` : "") +
      `</div>`
    );
    if (node.source_url) {
      parts.push(`<div style="margin-top:4px"><a href="${esc(node.source_url)}" target="_blank" rel="noopener noreferrer">${esc(node.source_url)}</a></div>`);
    }
    if (node.url) {
      parts.push(`<div style="margin-top:4px"><a href="${esc(node.url)}" target="_blank" rel="noopener noreferrer">${esc(node.url)}</a></div>`);
    }
    const meta = [];
    if (typeof node.mentions === "number") meta.push(`Mentions: ${node.mentions}`);
    if (typeof node.importance === "number") meta.push(`Importance: ${node.importance}/5`);
    if (meta.length) parts.push(`<div style="margin-top:4px;color:var(--muted)">${esc(meta.join(" · "))}</div>`);
    if (node.aliases && node.aliases.length) {
      parts.push(`<div style="margin-top:6px;color:var(--muted);font-size:.85rem">Also known as: ${node.aliases.map(esc).join(", ")}</div>`);
    }
    if (node.summary) parts.push(`<div style="margin-top:8px">${esc(node.summary)}</div>`);
    if (node.abstract) parts.push(`<div style="margin-top:8px">${esc(node.abstract)}</div>`);
    if (node.tags?.length) {
      parts.push(`<div style="margin:8px 0">${node.tags.map((t) => `<span class="kg-tag">${esc(t)}</span>`).join(" ")}</div>`);
    }
    if (node.note) parts.push(`<div style="margin-top:6px;color:var(--muted);font-style:italic">${esc(node.note)}</div>`);
    if (node.text) parts.push(`<div style="white-space:pre-wrap;margin-top:10px">${esc(node.text)}</div>`);

    const groupKeys = Object.keys(groups);
    if (groupKeys.length) {
      const blocks = groupKeys.map((k) =>
        `<div style="margin-top:8px"><span style="font-size:.74rem;text-transform:uppercase;color:var(--muted);letter-spacing:.05em">${esc(k)}</span>` +
        `<div style="margin-top:4px;display:flex;gap:4px;flex-wrap:wrap">${groups[k].join("")}</div></div>`
      ).join("");
      parts.push(`<div style="margin-top:12px">${blocks}</div>`);
    }
    kgDetail.innerHTML = parts.join("");
  }

  let network = null;

  function nodeVisual(n) {
    const layer = LAYERS[n.type] || { color: "#8b98ab" };
    let size = 12;
    if (n.type === "entity") size = 10 + (n.importance || 3) * 3 + Math.min((n.mentions || 1), 6);
    else if (n.type === "topic") size = 22;
    else if (n.type === "synthesis") size = 18;
    else if (n.type === "source") size = 16;
    else if (n.type === "chunk") size = 9;
    return { color: layer.color, size };
  }

  function edgeVisual(e) {
    switch (e.kind) {
      case "relation":
        return { color: "#7c5bff", width: 2, dashes: e.confidence === "INFERRED" || e.confidence === "AMBIGUOUS", showLabel: true };
      case "belongs_to":
        return { color: "#fbbf24", width: 2, dashes: false, showLabel: false };
      case "subtopic_of":
        return { color: "#f59e0b", width: 3, dashes: false, showLabel: false };
      case "from_source":
        return { color: "#2f5d4a", width: 1, dashes: true, showLabel: false };
      case "covers":
        return { color: "#f472b6", width: 2, dashes: true, showLabel: false };
      case "mentions":
      default:
        return { color: "#2d3a52", width: 1, dashes: false, showLabel: false };
    }
  }

  function renderLegend() {
    if (!kgLegendEl) return;
    kgLegendEl.innerHTML = Object.entries(LAYERS).map(([key, v]) => `
      <label class="legend-chip ${enabledLayers.has(key) ? "" : "off"}" data-layer="${key}">
        <span class="layer-dot" style="background:${v.color}"></span>${v.label}
      </label>
    `).join("");
    kgLegendEl.querySelectorAll(".legend-chip").forEach((el) => {
      el.addEventListener("click", () => {
        const key = el.dataset.layer;
        if (enabledLayers.has(key)) enabledLayers.delete(key);
        else enabledLayers.add(key);
        el.classList.toggle("off", !enabledLayers.has(key));
        renderGraph(lastGraph);
      });
    });
  }

  function renderGraph(graph) {
    renderLegend();
    if (typeof vis === "undefined") {
      kgGraphEl.innerHTML = `<div style="padding:24px;color:var(--muted);text-align:center">Graph viz library couldn't load (offline?). The data is still saved — use the lists on the left.</div>`;
      return;
    }
    const visibleIds = new Set(
      graph.nodes.filter((n) => enabledLayers.has(n.type)).map((n) => n.id)
    );
    const visNodes = graph.nodes
      .filter((n) => visibleIds.has(n.id))
      .map((n) => {
        const v = nodeVisual(n);
        return {
          id: n.id,
          label: (n.name || n.title || (n.preview || "").slice(0, 30) || n.id).replace(/\s+/g, " "),
          group: n.type,
          title: (n.summary || n.abstract || n.text || n.name || n.title || n.id || "").slice(0, 400),
          value: v.size,
          size: v.size,
          color: { background: v.color, border: v.color },
        };
      });
    const visEdges = graph.edges
      .filter((e) => visibleIds.has(e.from) && visibleIds.has(e.to))
      .map((e) => {
        const v = edgeVisual(e);
        return {
          id: e.id, from: e.from, to: e.to,
          label: v.showLabel ? e.label : undefined,
          arrows: "to",
          dashes: v.dashes,
          color: { color: v.color, highlight: "#9d83ff", opacity: 0.9 },
          width: v.width,
          font: { color: "#c4b5ff", size: 11, strokeWidth: 0, align: "middle" },
        };
      });

    const data = { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) };
    const options = {
      nodes: {
        shape: "dot",
        font: { color: "#e6edf3", size: 12, face: "system-ui" },
        borderWidth: 2,
        scaling: { min: 8, max: 34 },
      },
      groups: {
        source:    { color: { background: LAYERS.source.color,    border: LAYERS.source.color } },
        chunk:     { color: { background: LAYERS.chunk.color,     border: LAYERS.chunk.color } },
        entity:    { color: { background: LAYERS.entity.color,    border: LAYERS.entity.color } },
        topic:     { color: { background: LAYERS.topic.color,     border: LAYERS.topic.color }, shape: "diamond" },
        synthesis: { color: { background: LAYERS.synthesis.color, border: LAYERS.synthesis.color }, shape: "star" },
      },
      edges: {
        smooth: { type: "continuous", roundness: 0.2 },
        font: { color: "#8b98ab", size: 10, strokeWidth: 0, align: "middle" },
      },
      physics: {
        stabilization: { iterations: 150 },
        barnesHut: { gravitationalConstant: -4000, springLength: 130, centralGravity: 0.25 },
      },
      interaction: { hover: true, tooltipDelay: 200 },
    };
    if (network) network.destroy();
    network = new vis.Network(kgGraphEl, data, options);
    network.on("click", (params) => {
      if (params.nodes.length) {
        const node = graph.nodes.find((n) => n.id === params.nodes[0]);
        if (node) renderDetail(node, graph);
      }
    });
  }

  integrateBtn.addEventListener("click", async () => {
    integrateBtn.disabled = true;
    const prev = integrateBtn.textContent;
    integrateBtn.textContent = "Integrating…";
    try {
      const provider = document.querySelector("#provider")?.value || "auto";
      const model = document.querySelector("#model")?.value?.trim() || "";
      const res = await fetch("/api/kg/integrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model, use_ai: true }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Integrate failed (${res.status})`);
      await refresh();
      flash(
        `Integrated ${data.chunks_integrated} chunk(s), ` +
        `${data.entities_added} new entities (${data.entities_reinforced} reinforced), ` +
        `${data.sources_added} sources, ${data.relation_edges_added} relations. ` +
        `Engine: ${data.provider_used}${data.model_used ? " · " + data.model_used : ""}. ` +
        `Tip: click Maintain ✨ to de-duplicate and build the topic map.`,
      );
    } catch (e) {
      flash(e.message, "error");
    } finally {
      integrateBtn.disabled = false;
      integrateBtn.textContent = prev;
    }
  });

  // --- Agent: Ask + Maintain -------------------------------------------------
  function providerModel() {
    return {
      provider: document.querySelector("#provider")?.value || "auto",
      model: document.querySelector("#model")?.value?.trim() || "",
    };
  }

  function showAnswer(html) {
    askAnswer.classList.remove("hidden");
    askAnswer.innerHTML = html;
  }

  async function doAsk() {
    const question = askInput.value.trim();
    if (!question) return;
    askBtn.disabled = true;
    const prev = askBtn.textContent;
    askBtn.textContent = "Thinking…";
    showAnswer(`<div class="ask-thinking"><span class="spinner"></span>Reasoning over your library…</div>`);
    try {
      const res = await fetch("/api/agent/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, ...providerModel() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Ask failed (${res.status})`);
      let html = `<div class="ask-body">${mdLite(data.answer || "")}</div>`;
      if (data.citations && data.citations.length) {
        html += `<div class="ask-citations"><h4>Citations</h4>` + data.citations.map((c) => {
          const label = c.source_title || c.source_url || c.chunk_id;
          const link = c.source_url
            ? `<a href="${esc(c.source_url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`
            : esc(label);
          return `<div class="cite"><span class="cite-id">${esc(c.chunk_id)}</span> ${link}` +
                 (c.preview ? `<div class="cite-prev">${esc(c.preview)}</div>` : "") + `</div>`;
        }).join("") + `</div>`;
      } else {
        html += `<div class="ask-note">No grounding citations were found — the library may not contain this yet.</div>`;
      }
      const used = [data.provider, data.model].filter(Boolean).join(" · ");
      if (used) html += `<div class="ask-foot">Answered by ${esc(used)}${data.filed ? ` · saved to wiki/${esc(data.filed)}` : ""}</div>`;
      showAnswer(html);
      await refresh();
    } catch (e) {
      showAnswer(`<div style="color:var(--error)">${esc(e.message)}</div>`);
    } finally {
      askBtn.disabled = false;
      askBtn.textContent = prev;
    }
  }

  async function doMaintain() {
    maintainBtn.disabled = true;
    const prev = maintainBtn.textContent;
    maintainBtn.textContent = "Maintaining…";
    showAnswer(`<div class="ask-thinking"><span class="spinner"></span>De-duplicating, building topics, and synthesizing…</div>`);
    try {
      const res = await fetch("/api/agent/maintain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...providerModel() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Maintain failed (${res.status})`);
      let html = `<div class="ask-body">${mdLite(data.report || "Maintenance complete.")}</div>`;
      if (data.before && data.after) {
        html += `<div class="ask-foot">Entities ${data.before.entities} → ${data.after.entities} · ` +
                `topics ${data.after.topics} · syntheses ${data.after.syntheses}</div>`;
      }
      showAnswer(html);
      await refresh();
    } catch (e) {
      showAnswer(`<div style="color:var(--error)">${esc(e.message)}</div>`);
    } finally {
      maintainBtn.disabled = false;
      maintainBtn.textContent = prev;
    }
  }

  async function loadAgentStatus() {
    try {
      const s = await fetch("/api/agent/status").then((r) => r.json());
      if (!s.agent_available) {
        agentReady = false;
        agentStatusEl.textContent = "Agent not installed — pip install deepagents";
        agentStatusEl.className = "provider-status warn";
      } else if (!s.provider_ready) {
        agentReady = false;
        agentStatusEl.textContent = "Set an API key to ask & maintain";
        agentStatusEl.className = "provider-status warn";
      } else {
        agentReady = true;
        agentStatusEl.textContent = `Ready: ${s.provider_ready}`;
        agentStatusEl.className = "provider-status ok";
      }
    } catch (e) {
      agentStatusEl.textContent = "";
    }
    askBtn.disabled = !agentReady;
    maintainBtn.disabled = !agentReady;
  }

  if (askBtn) askBtn.addEventListener("click", doAsk);
  if (maintainBtn) maintainBtn.addEventListener("click", doMaintain);
  if (askInput) askInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); if (!askBtn.disabled) doAsk(); }
  });

  async function searchKg() {
    const q = kgQueryEl.value.trim();
    const res = await fetch("/api/kg/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, where: "overall" }),
    });
    const data = await res.json();
    const chunks = data.results.filter((n) => n.type === "chunk");
    const entities = data.results.filter((n) => n.type === "entity");
    renderList(overallList, chunks, "overall", `No chunks matched "${q}".`);
    if (entities.length) {
      flash(`Found ${chunks.length} chunk(s) and ${entities.length} entit${entities.length === 1 ? "y" : "ies"} matching "${q}". Entities: ${entities.map((e) => e.name).join(", ")}`);
    }
  }

  kgSearchBtn.addEventListener("click", searchKg);
  kgRefreshBtn.addEventListener("click", refresh);
  kgQueryEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); searchKg(); }
  });

  const modeTabs = document.querySelectorAll(".ingest-mode-tab");
  const modePanels = {
    files: document.getElementById("ingestFiles"),
    urls: document.getElementById("ingestUrls"),
    text: document.getElementById("ingestText"),
  };
  let activeMode = "files";
  modeTabs.forEach((t) => {
    t.addEventListener("click", () => {
      activeMode = t.dataset.mode;
      modeTabs.forEach((x) => x.classList.toggle("active", x === t));
      Object.entries(modePanels).forEach(([m, el]) => el.classList.toggle("active", m === activeMode));
    });
  });

  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const fileListEl = document.getElementById("fileList");
  let pendingFiles = [];

  function renderFileList() {
    fileListEl.innerHTML = pendingFiles.map((f, i) => `
      <div class="file-row">
        <span><span class="file-name">${esc(f.name)}</span><span class="file-meta">${(f.size / 1024).toFixed(1)} KB</span></span>
        <button data-i="${i}" title="Remove">✕</button>
      </div>
    `).join("");
    fileListEl.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => {
        pendingFiles.splice(Number(b.dataset.i), 1);
        renderFileList();
      })
    );
  }

  function addFiles(list) {
    for (const f of list) pendingFiles.push(f);
    renderFileList();
  }

  fileInput.addEventListener("change", (e) => addFiles(e.target.files));
  ["dragenter", "dragover"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("dragover"); })
  );
  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
  });

  const ingestBtn = document.getElementById("ingestBtn");
  const ingestProgress = document.getElementById("ingestProgress");

  function setIngestStatus(msg, kind = "info") {
    ingestProgress.className = "ingest-progress" + (kind === "error" ? " error" : kind === "ok" ? " ok" : "");
    ingestProgress.textContent = msg;
  }

  function getOptions() {
    return {
      chunk_size: parseInt(document.getElementById("chunkSize").value, 10) || 800,
      overlap: parseInt(document.getElementById("overlap").value, 10) || 120,
      tags: document.getElementById("ingestTags").value.split(",").map((t) => t.trim()).filter(Boolean),
    };
  }

  async function ingestText() {
    const text = document.getElementById("textInput").value.trim();
    if (text.length < 50) throw new Error("Paste at least 50 characters.");
    const opts = getOptions();
    const body = { ...opts, text, source_title: document.getElementById("textTitle").value.trim() || "Pasted document" };
    const res = await fetch("/api/kg/ingest/text", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Ingest failed");
    document.getElementById("textInput").value = "";
    document.getElementById("textTitle").value = "";
    return `Created ${data.chunks_created} chunk(s) from "${data.source}".`;
  }

  async function ingestUrls() {
    const urls = document.getElementById("urlInput").value.split("\n").map((u) => u.trim()).filter(Boolean);
    if (!urls.length) throw new Error("Paste one or more URLs.");
    const opts = getOptions();
    const res = await fetch("/api/kg/ingest/urls", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...opts, urls }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Ingest failed");
    document.getElementById("urlInput").value = "";
    const errs = data.results.filter((r) => r.error);
    const okCount = data.results.length - errs.length;
    let msg = `Fetched ${okCount}/${data.results.length} URLs, created ${data.total_chunks} chunks.`;
    if (errs.length) msg += " Errors: " + errs.map((e) => `${e.url} (${e.error})`).join("; ");
    return msg;
  }

  async function ingestFiles() {
    if (!pendingFiles.length) throw new Error("Add at least one file.");
    const opts = getOptions();
    const fd = new FormData();
    pendingFiles.forEach((f) => fd.append("files", f));
    fd.append("tags", opts.tags.join(","));
    fd.append("chunk_size", String(opts.chunk_size));
    fd.append("overlap", String(opts.overlap));
    const res = await fetch("/api/kg/ingest/files", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Ingest failed");
    pendingFiles = [];
    renderFileList();
    const errs = data.results.filter((r) => r.error);
    let msg = `Parsed ${data.results.length - errs.length}/${data.results.length} files, created ${data.total_chunks} chunks.`;
    if (errs.length) msg += " Errors: " + errs.map((e) => `${e.filename} (${e.error})`).join("; ");
    return msg;
  }

  ingestBtn.addEventListener("click", async () => {
    ingestBtn.disabled = true;
    const prev = ingestBtn.textContent;
    ingestBtn.textContent = "Ingesting…";
    setIngestStatus("Working…");
    try {
      const fn = activeMode === "files" ? ingestFiles : activeMode === "urls" ? ingestUrls : ingestText;
      const msg = await fn();
      setIngestStatus(msg, "ok");
      await refresh();
    } catch (e) {
      setIngestStatus(e.message, "error");
    } finally {
      ingestBtn.disabled = false;
      ingestBtn.textContent = prev;
    }
  });

  window.kg = { openModal, refresh };
  loadAgentStatus();
  refresh();
})();
