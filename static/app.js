/* Chroma — frontend logic */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const progressEl = $("#progress");
  const progressSteps = [...document.querySelectorAll("#progressSteps .step")];
  const resultsEl = $("#results");
  const errorEl = $("#error");
  const errorText = $("#errorText");
  const sampleGrid = $("#sampleGrid");
  const shuffleBtn = $("#samplesShuffle");

  // Normalisation: pct = value / max * 100 (clamped). Max = typical upper bound.
  const FEATURE_META = {
    color_diversity: { label: "Color diversity", max: 512 },
    colorfulness: { label: "Colorfulness", max: 160 },
    brightness: { label: "Brightness", max: 255 },
    saturation: { label: "Saturation", max: 255 },
    edge_chaos: { label: "Edge chaos", max: 0.3 },
    symmetry_score: { label: "Symmetry", max: 1 },
    texture_complexity: { label: "Texture", max: 100 },
    contrast: { label: "Contrast", max: 128 },
  };

  const STAGE_ORDER = ["features", "concepts", "caption", "heatmap", "interpret"];

  // ---- Health ------------------------------------------------------------
  async function checkHealth() {
    const dot = $("#healthDot");
    const text = $("#healthText");
    try {
      const r = await fetch("/health");
      const j = await r.json();
      dot.classList.add("ok");
      text.textContent = "ready";
      const badge = $("#engineBadge");
      if (j.models && !j.models.groq_configured) {
        badge.hidden = false;
        badge.textContent = "local interpretation";
      }
    } catch {
      dot.classList.add("err");
      text.textContent = "offline";
    }
  }

  // ---- Samples -----------------------------------------------------------
  async function loadSamples() {
    try {
      const r = await fetch("/api/samples");
      const items = await r.json();
      if (!Array.isArray(items) || !items.length) {
        sampleGrid.parentElement.hidden = true;
        return;
      }
      sampleGrid.parentElement.hidden = false;
      sampleGrid.innerHTML = "";
      items.forEach((it) => {
        const card = document.createElement("button");
        card.className = "sample-card";
        card.setAttribute("aria-label", `Analyse ${it.name}`);
        card.innerHTML = `<img src="${it.url}" alt="${it.name}" loading="lazy" /><span class="sample-overlay">Analyse</span>`;
        card.addEventListener("click", () => analyzeUrl(it.url, it.name));
        sampleGrid.appendChild(card);
      });
    } catch {
      sampleGrid.parentElement.hidden = true;
    }
  }

  function shuffleSamples() {
    const cards = [...sampleGrid.children];
    cards.sort(() => Math.random() - 0.5);
    cards.forEach((c) => sampleGrid.appendChild(c));
  }

  // ---- Crypto helpers ----------------------------------------------------
  async function sha256(buffer) {
    try {
      const digest = await crypto.subtle.digest("SHA-256", buffer);
      return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
    } catch {
      return null;
    }
  }

  function cacheGet(key) {
    try {
      const raw = localStorage.getItem("chroma:" + key);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }
  function cacheSet(key, value) {
    try {
      localStorage.setItem("chroma:" + key, JSON.stringify(value));
    } catch { /* quota / privacy */ }
  }

  // ---- Progress ----------------------------------------------------------
  function resetProgress() {
    progressEl.hidden = false;
    progressSteps.forEach((s) => {
      s.classList.remove("active", "done");
    });
  }
  function setStage(name) {
    const idx = STAGE_ORDER.indexOf(name);
    progressSteps.forEach((s, i) => {
      s.classList.toggle("active", i === idx);
      s.classList.toggle("done", i < idx);
    });
  }
  function finishProgress() {
    progressSteps.forEach((s) => {
      s.classList.remove("active");
      s.classList.add("done");
    });
    setTimeout(() => (progressEl.hidden = true), 500);
  }

  // A lightweight animated sweep so the user sees life while the models run.
  function startProgressAnim(stop) {
    let i = 0;
    setStage(STAGE_ORDER[0]);
    const timer = setInterval(() => {
      if (stop.done) return clearInterval(timer);
      i = Math.min(i + 1, STAGE_ORDER.length - 1);
      setStage(STAGE_ORDER[i]);
    }, 700);
  }

  // ---- Analysis ----------------------------------------------------------
  function showError(msg) {
    errorEl.hidden = false;
    errorText.textContent = msg;
    progressEl.hidden = true;
  }

  function readBuffer(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = reject;
      fr.readAsArrayBuffer(file);
    });
  }

  async function analyzeBuffer(buffer, name, srcUrl) {
    errorEl.hidden = true;
    resetProgress();

    const stop = { done: false };
    startProgressAnim(stop);

    const hash = await sha256(buffer);
    const cached = hash ? cacheGet(hash) : null;
    if (cached) {
      stop.done = true;
      finishProgress();
      renderResult(cached, true, srcUrl);
      return;
    }

    const form = new FormData();
    form.append("file", new Blob([buffer]), name);

    let res;
    try {
      res = await fetch("/api/analyze", { method: "POST", body: form });
    } catch {
      stop.done = true;
      showError("Could not reach the server. Is it running?");
      return;
    }

    if (!res.ok) {
      stop.done = true;
      const detail = await res.json().catch(() => ({}));
      showError(detail.detail || `Server error (${res.status})`);
      return;
    }

    const data = await res.json();
    stop.done = true;
    finishProgress();
    if (hash) cacheSet(hash, data);
    renderResult(data, false, srcUrl);
  }

  async function analyzeFile(file) {
    if (!file) return;
    const buffer = await readBuffer(file);
    const srcUrl = URL.createObjectURL(file);
    analyzeBuffer(buffer, file.name, srcUrl);
  }

  async function analyzeUrl(url, name) {
    errorEl.hidden = true;
    const res = await fetch(url);
    const buffer = await res.arrayBuffer();
    analyzeBuffer(buffer, name, url);
  }

  // ---- Rendering ---------------------------------------------------------
  function renderResult(d, cached, srcUrl) {
    resultsEl.hidden = false;
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });

    $("#resultName").textContent = d.name || "painting";
    const enginePill = $("#resultEngine");
    enginePill.className = "pill pill--accent";
    enginePill.textContent = d.engine === "groq" ? "✦ Groq interpretation" : "local interpretation";
    const timePill = $("#resultTime");
    timePill.textContent = cached
      ? `instant · cached`
      : `~${(d.elapsed_ms / 1000).toFixed(1)}s`;

    // Original image
    const orig = $("#originalImg");
    if (srcUrl) orig.src = srcUrl;
    orig.hidden = false;

    // Reset view tabs to "Original" and re-enable the heatmap tab.
    document.querySelectorAll(".viewer-tab").forEach((t) => {
      t.classList.toggle("is-active", t.dataset.view === "original");
      t.disabled = false;
    });

    const hm = $("#heatmapImg");
    if (d.heatmap) {
      hm.src = "data:image/png;base64," + d.heatmap;
      hm.hidden = true;
    } else {
      hm.hidden = true;
      hm.removeAttribute("src");
      document.querySelector('[data-view="heatmap"]').disabled = true;
    }
    $("#compareToggle").disabled = !d.heatmap;

    // Interpretation
    $("#interpretationText").textContent = d.interpretation || "";
    $("#captionText").textContent = d.caption || "";

    // Concepts
    const conceptsList = $("#conceptsList");
    conceptsList.innerHTML = "";
    (d.concepts || []).forEach((c, i) => {
      const li = document.createElement("li");
      li.className = "concept";
      li.style.animationDelay = `${i * 60}ms`;
      const pct = Math.round((c.score || 0) * 100);
      li.innerHTML = `
        <div class="concept-row">
          <span>${escapeHtml(c.label)}</span>
          <span class="concept-score">${pct}%</span>
        </div>
        <div class="concept-bar"><div class="concept-fill" data-w="${pct}"></div></div>`;
      conceptsList.appendChild(li);
    });

    // Features
    const grid = $("#featuresGrid");
    grid.innerHTML = "";
    const feats = d.features || {};
    Object.entries(FEATURE_META).forEach(([key, meta], i) => {
      if (feats[key] === undefined) return;
      const val = feats[key];
      const pct = Math.max(2, Math.min(100, (val / meta.max) * 100));
      const el = document.createElement("div");
      el.className = "feature";
      el.style.animationDelay = `${i * 50}ms`;
      el.innerHTML = `
        <div class="feature-label">
          <span>${meta.label}</span>
          <span class="feature-value">${fmt(val)}</span>
        </div>
        <div class="feature-track"><div class="feature-fill" data-w="${pct}"></div></div>`;
      grid.appendChild(el);
    });

    // Palette
    const swatches = $("#paletteSwatches");
    swatches.innerHTML = "";
    (feats.dominant_colors || []).forEach((c) => {
      const el = document.createElement("div");
      el.className = "swatch";
      el.style.background = c.hex;
      el.dataset.share = Math.round((c.share || 0) * 100) + "%";
      el.title = c.hex;
      swatches.appendChild(el);
    });

    // Animate bars after a frame so CSS transitions fire.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document.querySelectorAll(".concept-fill, .feature-fill").forEach((b) => {
          b.style.width = b.dataset.w + "%";
        });
      });
    });
  }

  function fmt(v) {
    if (typeof v === "number") return v >= 100 ? Math.round(v) : v.toFixed(2);
    return v;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // ---- Compare (hold-to-compare) ----------------------------------------
  function setupCompare() {
    const btn = $("#compareToggle");
    const orig = $("#originalImg");
    const hm = $("#heatmapImg");
    const show = () => { orig.hidden = true; hm.hidden = false; };
    const hide = () => { orig.hidden = false; hm.hidden = true; };
    btn.addEventListener("mousedown", show);
    btn.addEventListener("mouseup", hide);
    btn.addEventListener("mouseleave", hide);
    btn.addEventListener("touchstart", (e) => { e.preventDefault(); show(); }, { passive: false });
    btn.addEventListener("touchend", hide);

    // Tabs switch permanently.
    document.querySelectorAll(".viewer-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".viewer-tab").forEach((t) => t.classList.remove("is-active"));
        tab.classList.add("is-active");
        const view = tab.dataset.view;
        if (view === "heatmap" && hm.src) { show(); }
        else hide();
      });
    });
  }

  // ---- Events ------------------------------------------------------------
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => analyzeFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) analyzeFile(file);
  });

  shuffleBtn.addEventListener("click", shuffleSamples);
  $("#analyzeAnother").addEventListener("click", () => {
    resultsEl.hidden = true;
    errorEl.hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // ---- Init --------------------------------------------------------------
  checkHealth();
  loadSamples();
  setupCompare();
  setInterval(checkHealth, 15000);
})();
