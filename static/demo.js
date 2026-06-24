// Public-demo UI lock. Cosmetic only — the server (DEMO_MODE) is the real
// enforcement: it overrides provider/model on every route, blocks disabled
// features with 403, rate-limits, and caps spend. This just makes the UI honest:
// pin the model selector, surface the budget, and hide controls that 403.
(function () {
  "use strict";

  function pinModelSelector(general) {
    const prov = document.querySelector("#provider");
    const model = document.querySelector("#model");
    if (prov) {
      prov.innerHTML = `<option value="auto">Demo — cheapest model (locked)</option>`;
      prov.value = "auto";
      prov.disabled = true;
      prov.title = "Model is fixed in the public demo";
    }
    if (model) {
      model.value = general || "";
      model.placeholder = general || "demo model";
      model.readOnly = true;
      model.disabled = true;
      model.title = "Model is fixed in the public demo";
    }
  }

  // Selectors for controls that hit demo-disabled endpoints (agent, eval,
  // crossdoc, skill eval/review/rebuild/refine, delete buttons). Best-effort by
  // id/data-attr/text; anything missed still fails safe (server returns 403).
  const HIDE_SELECTORS = [
    "#kgAskBtn", "#kgMaintainBtn",                 // KG agent ask / maintain
    "#ragEvalBtn", "#ragCrossdocBtn", "#ragRagasBtn", "#ragExperimentBtn",
    "[data-demo-hide]",
  ];

  function hideDisabled() {
    HIDE_SELECTORS.forEach((sel) =>
      document.querySelectorAll(sel).forEach((el) => { el.style.display = "none"; }));
    // Delete buttons across cache/memory/skill/kg lists.
    document.querySelectorAll(
      "[data-ingest-delete], .cache-del, .mem-del, .skill-del, [data-kg-del]"
    ).forEach((el) => { el.style.display = "none"; });
  }

  function fmt(usd) { return "$" + (Number(usd) || 0).toFixed(3); }

  function renderBanner(st) {
    const b = document.getElementById("demoBanner");
    if (!b) return;
    b.classList.remove("hidden");
    b.innerHTML =
      `<strong>Live demo.</strong> Forced onto cheap models ` +
      `(<code>${st.general_model || "?"}</code> · code <code>${st.code_model || "?"}</code>) ` +
      `under a shared budget. <span id="demoRemain"></span> ` +
      `Agent runs &amp; evaluations are disabled, and data is sandboxed. ` +
      `<a href="https://github.com/Lyra-stellAI/BYO-WIKI" target="_blank" rel="noopener noreferrer">Run your own &rarr;</a>`;
    updateRemain(st);
  }

  function updateRemain(st) {
    const el = document.getElementById("demoRemain");
    if (!el) return;
    const v = st.visitor_remaining_usd, g = st.global_remaining_usd;
    el.textContent = `Your remaining: ${fmt(v)} · shared pool: ${fmt(g)}.`;
    el.classList.toggle("demo-low", (Number(v) || 0) <= 0 || (Number(g) || 0) <= 0);
  }

  async function refresh() {
    try {
      const st = await fetch("/api/demo-status").then((r) => r.json());
      if (st && st.demo) updateRemain(st);
      return st;
    } catch (e) { return null; }
  }

  async function init() {
    let st = null;
    try { st = await fetch("/api/demo-status").then((r) => r.json()); } catch (e) { return; }
    if (!st || !st.demo) return;            // not a demo build — do nothing
    document.body.classList.add("demo-mode");
    pinModelSelector(st.general_model);
    renderBanner(st);
    hideDisabled();
    // Re-hide after tab switches / async list renders, and refresh the budget.
    document.querySelectorAll(".tab").forEach((t) =>
      t.addEventListener("click", () => setTimeout(hideDisabled, 50)));
    setInterval(refresh, 15000);
    // Surface 429/403 from any fetch as a friendly toast via a global hook.
    window.__demoRefresh = refresh;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
