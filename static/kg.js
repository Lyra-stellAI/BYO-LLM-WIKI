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
  const kgDetail = $("#kgDetail");
  const tabCount = $("#tabCount");
  const integrateBtn = $("#integrateBtn");
  const kgQueryEl = $("#kgQuery");
  const kgSearchBtn = $("#kgSearchBtn");
  const kgRefreshBtn = $("#kgRefresh");

  function esc(s = "") {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
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

  function closeModal() {
    modal.classList.add("hidden");
  }

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
    $("#currentChunks").textContent = stats.current.chunks;
    $("#overallChunks").textContent = stats.overall.chunks;
    $("#overallEntities").textContent = stats.overall.entities;
    $("#overallEdges").textContent = stats.overall.edges;
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
    if (!confirm("Remove this chunk?")) return;
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
    const linkedEdges = graph.edges.filter((e) => e.from === node.id || e.to === node.id);
    const linked = linkedEdges.map((e) => {
      const otherId = e.from === node.id ? e.to : e.from;
      const other = graph.nodes.find((n) => n.id === otherId);
      if (!other) return "";
      const label = other.name || other.preview || other.id;
      return `<span class="kg-tag">${esc(label)} <em style="opacity:.6">· ${esc(e.label)}${e.confidence ? " · " + esc(e.confidence) : ""}</em></span>`;
    }).join(" ");

    const parts = [];
    parts.push(`<strong style="font-size:1rem">${esc(node.source_title || node.name || node.id)}</strong>`);
    if (node.source_url) {
      parts.push(`<div style="margin-top:4px"><a href="${esc(node.source_url)}" target="_blank" rel="noopener noreferrer">${esc(node.source_url)}</a></div>`);
    }
    if (node.kind) parts.push(`<div style="margin-top:4px;color:var(--muted)">Kind: ${esc(node.kind)} · Mentions: ${esc(String(node.mentions || 1))}</div>`);
    if (node.tags?.length) {
      parts.push(`<div style="margin:8px 0">${node.tags.map((t) => `<span class="kg-tag">${esc(t)}</span>`).join(" ")}</div>`);
    }
    if (node.note) parts.push(`<div style="margin-top:6px;color:var(--muted);font-style:italic">${esc(node.note)}</div>`);
    if (node.text) parts.push(`<div style="white-space:pre-wrap;margin-top:10px">${esc(node.text)}</div>`);
    if (linked) parts.push(`<div style="margin-top:12px"><strong style="font-size:.78rem;text-transform:uppercase;color:var(--muted);letter-spacing:.05em">Linked</strong><div style="margin-top:6px">${linked}</div></div>`);
    if (node.integrated_at) parts.push(`<div style="margin-top:10px;color:var(--muted);font-size:.8rem">Integrated ${esc(new Date(node.integrated_at).toLocaleString())}</div>`);

    kgDetail.innerHTML = parts.join("");
  }

  let network = null;

  function renderGraph(graph) {
    if (typeof vis === "undefined") {
      kgGraphEl.innerHTML = `<div style="padding:24px;color:var(--muted);text-align:center">Graph viz library couldn't load (offline?). The data is still saved — use the lists on the left.</div>`;
      return;
    }
    const visNodes = graph.nodes.map((n) => ({
      id: n.id,
      label: (n.name || (n.preview || "").slice(0, 36) || n.id).replace(/\s+/g, " "),
      group: n.type,
      title: (n.text || n.name || n.id || "").slice(0, 400),
      value: n.mentions || 1,
    }));
    const visEdges = graph.edges.map((e) => {
      const isRel = e.kind === "relation";
      return {
        id: e.id, from: e.from, to: e.to, label: e.label,
        arrows: "to",
        dashes: !isRel && (e.confidence === "INFERRED" || e.confidence === "AMBIGUOUS"),
        color: isRel ? { color: "#7c5bff", highlight: "#9d83ff" } : { color: "#2d3a52", highlight: "#5b9dff" },
        width: isRel ? 2 : 1,
        font: isRel
          ? { color: "#c4b5ff", size: 11, strokeWidth: 0, align: "middle" }
          : { color: "#8b98ab", size: 10, strokeWidth: 0, align: "middle" },
      };
    });
    const data = { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) };
    const options = {
      nodes: {
        shape: "dot",
        size: 14,
        font: { color: "#e6edf3", size: 12, face: "system-ui" },
        borderWidth: 1,
        scaling: { min: 10, max: 30 },
      },
      groups: {
        chunk: { color: { background: "#5b9dff", border: "#7ab2ff" } },
        entity: { color: { background: "#7c5bff", border: "#9d83ff" } },
      },
      edges: {
        color: { color: "#2d3a52", highlight: "#5b9dff" },
        font: { color: "#8b98ab", size: 10, strokeWidth: 0, align: "middle" },
        smooth: { type: "continuous", roundness: 0.2 },
        width: 1,
      },
      physics: {
        stabilization: { iterations: 120 },
        barnesHut: { gravitationalConstant: -3000, springLength: 120 },
      },
      interaction: { hover: true, tooltipDelay: 200 },
    };
    if (network) network.destroy();
    network = new vis.Network(kgGraphEl, data, options);
    network.on("click", (params) => {
      if (params.nodes.length) {
        const id = params.nodes[0];
        const node = graph.nodes.find((n) => n.id === id);
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
        `${data.edges_added} edges. Engine: ${data.provider_used}${data.model_used ? " · " + data.model_used : ""}.`,
      );
    } catch (e) {
      flash(e.message, "error");
    } finally {
      integrateBtn.disabled = false;
      integrateBtn.textContent = prev;
    }
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
    if (e.key === "Enter") {
      e.preventDefault();
      searchKg();
    }
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
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    })
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
    const body = {
      ...opts,
      text,
      source_title: document.getElementById("textTitle").value.trim() || "Pasted document",
    };
    const res = await fetch("/api/kg/ingest/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Ingest failed");
    document.getElementById("textInput").value = "";
    document.getElementById("textTitle").value = "";
    return `Created ${data.chunks_created} chunk(s) from "${data.source}".`;
  }

  async function ingestUrls() {
    const urls = document.getElementById("urlInput").value
      .split("\n").map((u) => u.trim()).filter(Boolean);
    if (!urls.length) throw new Error("Paste one or more URLs.");
    const opts = getOptions();
    const res = await fetch("/api/kg/ingest/urls", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...opts, urls }),
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
  refresh();
})();
