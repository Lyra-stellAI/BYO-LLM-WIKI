(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s = "") => String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  const buildBtn = $("#skillBuildBtn");
  const buildProg = $("#skillBuildProg");
  const buildOut = $("#skillBuildOut");
  const listEl = $("#skillList");
  const filterEl = $("#skillFilter");
  const statusEl = $("#skillStatus");
  const tabCount = $("#skillTabCount");

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
    $("#skillTotal").textContent = s.total ?? 0;
    $("#skillAccepted").textContent = s.accepted ?? 0;
    $("#skillPending").textContent = s.pending_review ?? 0;
    $("#skillRevision").textContent = s.needs_revision ?? 0;
    $("#skillRejected").textContent = s.rejected ?? 0;
    const align = s.gate_human_alignment;
    $("#skillAlign").textContent = (align == null)
      ? "gate↔human —" : `gate↔human ${align}`;
    if (s.pending_review) { tabCount.textContent = s.pending_review; tabCount.classList.remove("hidden"); }
    else { tabCount.classList.add("hidden"); }
    const recall = s.embeddings ? "semantic recall" : "keyword recall (set OPENAI_API_KEY for semantic)";
    statusEl.textContent = `${s.accepted ?? 0} live · ${s.pending_review ?? 0} to review · ${recall}`;
    statusEl.className = "provider-status " + (s.embeddings ? "ok" : "warn");
  }

  function gateBadge(gate) {
    return `<span class="skill-badge skill-gate-${esc(gate || "review")}">gate: ${esc(gate || "—")}</span>`;
  }

  function statusBadge(st) {
    return `<span class="skill-badge skill-st-${esc(st)}">${esc((st || "").replace(/_/g, " "))}</span>`;
  }

  function evalSummary(ev) {
    if (!ev) return "";
    const det = ev.deterministic || {};
    const rm = ev.rubric_mean ?? (ev.rubric && ev.rubric.mean);
    const trig = ev.triggering;
    return `<span class="kg-tag">checks ${det.passed ?? "?"}/${det.total ?? "?"}</span>` +
      (rm != null ? `<span class="kg-tag">rubric ${esc(String(rm))}</span>` : "") +
      (trig && trig.f1 != null ? `<span class="kg-tag">trigger F1 ${esc(String(trig.f1))}</span>` : "");
  }

  async function loadBackends() {
    try {
      const data = await fetch("/api/skill/backends").then((r) => r.json());
      const sel = $("#skillBackend");
      const cc = (data.backends || {}).claude_code || {};
      const opt = sel.querySelector('option[value="claude_code"]');
      if (opt) {
        opt.disabled = !cc.available;
        opt.textContent = cc.available
          ? "Claude Code (subprocess)"
          : "Claude Code (CLI not found)";
      }
      sel.value = (data.default === "claude_code" && cc.available) ? "claude_code" : "pipeline";
    } catch (e) { /* ignore */ }
  }

  async function loadObservability() {
    try {
      const b = (await fetch("/api/skill/observability").then((r) => r.json())).benchmark || {};
      $("#obsBuilds").textContent = b.builds ?? 0;
      $("#obsPass").textContent = b.gate_pass_rate == null ? "—" : b.gate_pass_rate;
      $("#obsRubric").textContent = b.avg_rubric_mean == null ? "—" : b.avg_rubric_mean;
      $("#obsTokens").textContent = b.avg_tokens == null ? "—" : Math.round(b.avg_tokens);
      $("#obsLatency").textContent = b.avg_duration_ms == null ? "—" : Math.round(b.avg_duration_ms);
      $("#obsTotalTokens").textContent = b.total_tokens ?? 0;
    } catch (e) { /* ignore */ }
  }

  function actionsFor(s) {
    const id = esc(s.id);
    const btn = (act, label, cls = "") =>
      `<span class="kg-tag skill-act ${cls}" data-act="${act}" data-id="${id}">${label}</span>`;
    let acts = "";
    if (s.status === "pending_review" || s.status === "draft" || s.status === "evaluated") {
      acts += btn("accept", "✓ accept", "ok") + btn("revise", "✎ revise") + btn("reject", "✕ reject", "danger");
      acts += btn("refine", "✦ refine");
    } else if (s.status === "needs_revision") {
      acts += btn("rebuild", "↻ rebuild") + btn("refine", "✦ refine");
    } else if (s.status === "accepted") {
      acts += btn("export", "⤓ SKILL.md", "ok") + btn("refine", "✦ refine");
    }
    acts += btn("view", "view");
    acts += btn("forget", "forget", "danger");
    return acts;
  }

  function skillItem(s) {
    const tools = (s.tools || []).map((t) => `<span class="kg-tag">${esc(t)}</span>`).join("");
    const needsNote = (s.status === "pending_review" || s.status === "needs_revision"
      || s.status === "draft" || s.status === "evaluated");
    const reviewRow = needsNote
      ? `<div class="skill-review-row">
           <input class="skill-notes" data-id="${esc(s.id)}" placeholder="reviewer notes / revision guidance (optional)" />
           <input class="skill-score" data-id="${esc(s.id)}" type="number" min="0" max="1" step="0.1" placeholder="score 0-1" />
         </div>` : "";
    const thread = s.graph_thread_id || "";
    const durableBadge = thread
      ? `<span class="skill-badge skill-st-pending_review" title="LangGraph thread ${esc(thread)} — decisions resume the durable build">⛓ durable</span>`
      : "";
    return `<div class="kg-item skill-card" data-id="${esc(s.id)}" data-thread="${esc(thread)}">
      <div class="kg-preview"><strong>${esc(s.name)}</strong> — ${esc(s.description || "")}</div>
      <div class="kg-meta">
        ${statusBadge(s.status)}
        ${durableBadge}
        ${s.eval ? gateBadge(s.eval.gate) : ""}
        ${evalSummary(s.eval)}
        ${s.version ? `<span class="kg-tag">v${esc(String(s.version))}</span>` : ""}
        ${tools}
      </div>
      ${reviewRow}
      <div class="kg-meta skill-actions">${actionsFor(s)}</div>
      <div class="skill-detail hidden" data-detail="${esc(s.id)}"></div>
    </div>`;
  }

  async function loadList() {
    try {
      const f = filterEl.value;
      const data = await fetch(`/api/skill/list${f ? `?status=${encodeURIComponent(f)}` : ""}`)
        .then((r) => r.json());
      applyStats(data.stats);
      const rows = data.skills || [];
      listEl.innerHTML = rows.length
        ? rows.map(skillItem).join("")
        : `<div class="empty">No skills yet. Build one from context above.</div>`;
      loadObservability();
    } catch (e) {
      listEl.innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    }
  }

  async function loadStats() {
    try { applyStats(await fetch("/api/skill/stats").then((r) => r.json())); }
    catch (e) { /* ignore */ }
  }

  function renderBuildResult(res) {
    const s = res.skill || {};
    const ev = res.eval || {};
    const det = ev.deterministic || {};
    const rubric = ev.rubric || {};
    const dims = rubric.per_dimension || {};
    const dimLine = Object.keys(dims).length
      ? Object.entries(dims).map(([k, v]) => `${esc(k)} ${esc(String(v))}`).join(" · ")
      : (rubric.error ? esc(rubric.error) : "no rubric panel");
    const failures = (det.failures || []).length
      ? `<div class="cite-prev">failed checks: ${esc((det.failures || []).join(", "))}</div>` : "";
    const obs = res.observability || {};
    const trig = ev.triggering;
    const trigLine = trig
      ? `<div class="cite"><strong>Triggering.</strong> precision ${esc(String(trig.precision))} ·
         recall ${esc(String(trig.recall))} · F1 ${esc(String(trig.f1))}
         <span style="color:var(--muted)">(${esc(String(trig.judge || ""))})</span></div>` : "";
    buildOut.classList.remove("hidden");
    buildOut.innerHTML = `
      <h4>Built “${esc(s.name)}” ${statusBadge(s.status)} ${gateBadge(res.gate)}</h4>
      <div class="cite"><strong>Author.</strong>
        <span class="kg-tag">${esc(res.backend || "pipeline")}</span>
        ${esc(res.provider || "")}/${esc(res.model || "")}
        ${obs.tool_mode ? `<span class="kg-tag">tools ×${esc(String(obs.tools_used || 0))}</span>` : ""}
        <span class="kg-tag">${esc(String(obs.duration_ms ?? "?"))} ms</span>
        <span class="kg-tag">${esc(String(obs.tokens ?? "?"))} tokens</span>
        ${obs.cost_usd != null ? `<span class="kg-tag">$${esc(String(obs.cost_usd))}</span>` : ""}</div>
      <div class="cite"><strong>Description.</strong> ${esc(s.description || "")}</div>
      <div class="cite"><strong>Eval.</strong> deterministic ${esc(String(det.passed))}/${esc(String(det.total))}
        · rubric mean ${esc(String(ev.rubric_mean ?? "—"))} <span style="color:var(--muted)">(${dimLine})</span>
        ${failures}</div>
      ${trigLine}
      <div class="cite"><strong>Gate.</strong> ${esc((ev.gate_reasons || []).join("; "))}</div>
      <div class="ask-note">${s.status === "pending_review"
        ? "Passed the gate — review it below to accept, revise, or reject."
        : (s.status === "rejected" ? "Rejected by the gate — revise the context/goal or rebuild."
          : "Awaiting your review below.")}</div>`;
  }

  function renderDurableResult(res) {
    const s = res.skill || {};
    const ev = res.eval || {};
    const det = ev.deterministic || {};
    buildOut.classList.remove("hidden");
    buildOut.innerHTML = `
      <h4>⛓ Durable build paused for review — “${esc(s.name)}” ${statusBadge(res.status || s.status)} ${gateBadge(res.gate)}</h4>
      <div class="cite"><strong>Thread.</strong> <code>${esc(res.thread_id || "")}</code>
        <span class="kg-tag">checkpoint: ${esc(res.checkpoint || "sqlite")}</span></div>
      <div class="cite"><strong>Description.</strong> ${esc(s.description || "")}</div>
      <div class="cite"><strong>Eval.</strong> deterministic ${esc(String(det.passed))}/${esc(String(det.total))}
        · rubric mean ${esc(String(ev.rubric_mean ?? "—"))}</div>
      <div class="ask-note">Checkpointed and waiting. Decide in the review queue below
        (accept / revise / reject) — even later, or after a restart.</div>`;
  }

  async function doBuild() {
    const text = $("#skillText").value.trim();
    const query = $("#skillQuery").value.trim();
    const tags = $("#skillTags").value.trim();
    if (!text && !query && !tags) {
      buildProg.className = "ingest-progress error";
      buildProg.textContent = "Provide context: paste text, a query, or tags.";
      return;
    }
    const durable = $("#skillDurable").checked;
    buildBtn.disabled = true;
    const prev = buildBtn.textContent;
    buildBtn.textContent = "Building…";
    buildProg.className = "ingest-progress";
    buildProg.innerHTML = `<span class="spinner"></span>${durable ? "⛓ " : ""}understand → analyze → codeact → eval → gate…`;
    buildOut.classList.add("hidden");
    try {
      const res = await postJSON(durable ? "/api/skill/graph/build" : "/api/skill/build", {
        text: text || undefined,
        query: query || undefined,
        tags: tags || undefined,
        goal: $("#skillGoal").value.trim() || undefined,
        run_rubric: $("#skillRubric").checked,
        use_tools: $("#skillTools").checked,
        backend: $("#skillBackend").value || undefined,
      });
      buildProg.className = "ingest-progress ok";
      if (durable && res.awaiting_review) {
        buildProg.textContent = `Paused for review (checkpoint: ${res.checkpoint || "sqlite"}).`;
        renderDurableResult(res);
      } else {
        buildProg.textContent = `Drafted and evaluated (gate: ${res.gate}).`;
        renderBuildResult(res);
      }
      await loadList();
    } catch (e) {
      buildProg.className = "ingest-progress error";
      buildProg.textContent = e.message;
    } finally {
      buildBtn.disabled = false;
      buildBtn.textContent = prev;
    }
  }

  function detailHTML(skill, markdown) {
    const det = (skill.eval && skill.eval.deterministic && skill.eval.deterministic.checks) || [];
    const checkRows = det.map((c) =>
      `<div class="cite-prev">${c.passed ? "✓" : "✕"} ${esc(c.key)} <span style="color:var(--muted)">(${esc(c.detail || "")})</span></div>`
    ).join("");
    return `<div class="skill-detail-inner">
      ${checkRows ? `<h4>Deterministic checks</h4>${checkRows}` : ""}
      <h4>SKILL.md</h4>
      <pre class="skill-md">${esc(markdown || "")}</pre>
    </div>`;
  }

  async function toggleDetail(id, panel) {
    if (!panel.classList.contains("hidden")) { panel.classList.add("hidden"); panel.innerHTML = ""; return; }
    panel.classList.remove("hidden");
    panel.innerHTML = `<div class="ask-thinking"><span class="spinner"></span>Loading…</div>`;
    try {
      const data = await fetch(`/api/skill/${encodeURIComponent(id)}`).then((r) => r.json());
      panel.innerHTML = detailHTML(data.skill || {}, data.markdown);
    } catch (e) {
      panel.innerHTML = `<div class="cite-prev" style="color:var(--error)">${esc(e.message)}</div>`;
    }
  }

  listEl.addEventListener("click", async (e) => {
    const act = e.target.closest(".skill-act");
    if (!act) return;
    const id = act.dataset.id;
    const card = act.closest(".skill-card");
    const action = act.dataset.act;
    const notesEl = card.querySelector(".skill-notes");
    const scoreEl = card.querySelector(".skill-score");
    const notes = notesEl ? notesEl.value.trim() : "";
    const score = scoreEl && scoreEl.value !== "" ? parseFloat(scoreEl.value) : undefined;
    try {
      if (action === "view") {
        await toggleDetail(id, card.querySelector(".skill-detail"));
      } else if (action === "forget") {
        await postJSON(`/api/skill/${encodeURIComponent(id)}`, null, "DELETE");
        await loadList();
      } else if (action === "export") {
        await postJSON(`/api/skill/${encodeURIComponent(id)}/export`, {});
        act.textContent = "written ✓";
      } else if (action === "rebuild") {
        act.textContent = "rebuilding…";
        await postJSON(`/api/skill/${encodeURIComponent(id)}/rebuild`, { extra_guidance: notes });
        await loadList();
      } else if (action === "refine") {
        act.textContent = "refining…";
        await postJSON(`/api/skill/${encodeURIComponent(id)}/refine`, {});
        await loadList();
      } else if (action === "accept" || action === "reject" || action === "revise") {
        const thread = card.dataset.thread;
        if (thread) {
          // Durable build: resume the LangGraph thread (pauses again on revise).
          act.textContent = action + "…";
          await postJSON("/api/skill/graph/resume",
            { thread_id: thread, decision: action, notes, score });
        } else {
          await postJSON(`/api/skill/${encodeURIComponent(id)}/review`,
            { decision: action, notes, score });
        }
        await loadList();
      }
    } catch (err) {
      const detail = card.querySelector(".skill-detail");
      detail.classList.remove("hidden");
      detail.innerHTML = `<div class="cite-prev" style="color:var(--error)">${esc(err.message)}</div>`;
    }
  });

  buildBtn.addEventListener("click", doBuild);
  $("#skillRefresh").addEventListener("click", loadList);
  $("#skillObsRefresh").addEventListener("click", loadObservability);
  filterEl.addEventListener("change", loadList);

  document.querySelectorAll('.tab[data-tab="skill"]').forEach((t) =>
    t.addEventListener("click", loadList));

  loadBackends();
  loadStats();
  loadList();
})();
