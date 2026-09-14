/* ============================================================
   Shadow AI — live dashboard logic
   ============================================================ */

const $  = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);
const API_BASE = "http://127.0.0.1:8000";
const apiUrl = (path) => API_BASE + path;
/* ---------- helpers ---------- */
const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString());
const fmtBytes = (n) => {
  if (!n) return "0 B";
  if (n > 1e6) return (n / 1e6).toFixed(2) + " MB";
  if (n > 1e3) return (n / 1e3).toFixed(1) + " KB";
  return n + " B";
};
const sevClass = (s) =>
  ({ Critical: "Critical", High: "High", Medium: "Medium", Low: "Low" }[s] || "Unknown");

/* ---------- tabs ---------- */
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    const t = btn.dataset.tab;
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + t));
    if (t === "findings") loadFindings();
    if (t === "review")   loadReview();
    if (t === "domains")  loadDomains();
  });
});

/* ============================================================
   LIVE STATE — driven entirely by SSE
   ============================================================ */
const state = {
  rows: 0, chunks: 0, review: 0, ai: 0, findings: 0,
  classifyCount: 0, chunkCount: 0,
  startedAt: null, hasFindings: false,
};

function updateKpis() {
  $("kpi-rows").textContent     = fmt(state.rows);
  $("kpi-chunks").textContent   = fmt(state.chunks);
  $("kpi-review").textContent   = fmt(state.review);
  $("kpi-ai").textContent       = fmt(state.ai);
  $("kpi-findings").textContent = fmt(state.findings);
}
setInterval(() => {
  if (!state.startedAt) return;
  const s = (Date.now() - state.startedAt) / 1000;
  $("kpi-elapsed").textContent = s < 60 ? s.toFixed(0) + "s" : (s / 60).toFixed(1) + "m";
}, 500);

function setPhase(name) {
  $$(".phase").forEach((el) =>
    el.classList.toggle("active", el.dataset.phase === name));
}

function pushFeed(elId, html) {
  const el = $(elId);
  const ph = el.querySelector(".placeholder");
  if (ph) ph.remove();
  const div = document.createElement("div");
  div.className = "row";
  div.innerHTML = html;
  el.prepend(div);
  while (el.children.length > 250) el.removeChild(el.lastChild);
}

function resetRun() {
  state.rows = state.chunks = state.review = state.ai = 0;
  state.findings = state.classifyCount = state.chunkCount = 0;
  state.startedAt = Date.now();
  state.hasFindings = false;

  $("classify-feed").innerHTML = '<div class="placeholder">waiting…</div>';
  $("chunk-feed").innerHTML    = '<div class="placeholder">waiting…</div>';
  $("findings-body").innerHTML = '<tr><td colspan="8" class="placeholder">no findings yet</td></tr>';
  $("classify-count").textContent = "0";
  $("chunk-count").textContent    = "0";
  $("finding-count").textContent  = "0";
  updateKpis();
}

/* ============================================================
   SSE handler
   ============================================================ */
function connect() {
  const es = new EventSource(apiUrl("/stream"));

  es.onopen = () => {
    $("conn-status").textContent = "connected";
    $("conn-status").className = "conn ok";
    $("live-dot").className = "dot on";
  };
  es.onerror = () => {
    $("conn-status").textContent = "reconnecting…";
    $("conn-status").className = "conn warn";
    $("live-dot").className = "dot";
  };

  es.onmessage = (ev) => {
    let e;
    try { e = JSON.parse(ev.data); } catch { return; }
    handleEvent(e);
  };
}

function handleEvent(e) {
  switch (e.kind) {

    case "run_started":
      resetRun();
      $("run-id").textContent = e.csv ? "· " + e.csv : "";
      setPhase("1");
      pushFeed("chunk-feed",
        `<span class="badge AI">▶ run</span> <span class="muted">${e.csv || ""}</span>`);
      break;

    case "phase1_progress":
      state.rows = e.rows; updateKpis();
      break;

    case "phase1_done":
      state.rows = e.rows; updateKpis();
      setPhase("2");
      pushFeed("chunk-feed",
        `<span class="muted">✓ phase 1 — ${fmt(e.rows)} rows, ${fmt(e.failed)} failed, ${fmt(e.ooo)} out-of-order</span>`);
      break;

    case "phase2_done":
      setPhase("3");
      pushFeed("chunk-feed",
        `<span class="muted">✓ phase 2 — sorted ${fmt(e.rows)} rows</span>`);
      break;

    case "chunk": {
      state.chunks++; state.chunkCount++;
      if (e.needs_review) state.review++;
      if (e.is_ai) state.ai++;
      updateKpis();
      $("chunk-count").textContent = state.chunkCount + " chunks";

      const kind = e.is_ai ? "AI" : (e.needs_review ? "REVIEW" : "drop");
      const apps = (e.apps || []).join(", ") || "—";
      pushFeed("chunk-feed",
        `<span class="badge ${kind}">[${kind}]</span> ` +
        `<span class="muted">#${e.chunk_id}</span> ` +
        `<span>${e.tier}</span> ` +
        `<span class="muted">${apps}</span> ` +
        `<span class="muted">${e.events} ev · ${fmtBytes(e.bytes)}${e.user ? " · " + e.user : ""}</span>`);
      break;
    }

    case "classify": {
      state.classifyCount++;
      $("classify-count").textContent = state.classifyCount + " verdicts";
      const status = (e.status || "unknown").toLowerCase();
      pushFeed("classify-feed",
        `<span class="badge ${status}">${status.toUpperCase()}</span> ` +
        `<span>${e.domain}</span> ` +
        (e.app ? `<span class="muted">→ ${e.app}</span> ` : "") +
        `<span class="muted">conf=${(e.confidence || 0).toFixed(2)}</span>`);
      break;
    }

    case "finding": {
      if (!state.hasFindings) {
        $("findings-body").innerHTML = "";
        state.hasFindings = true;
      }
      state.findings++;
      updateKpis();
      $("finding-count").textContent = state.findings;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">#${e.chunk_id ?? "—"}</td>
        <td>${e.user || "—"}</td>
        <td>${e.application || "—"}</td>
        <td class="muted">${e.match_tier || "—"}</td>
        <td><span class="pill ${sevClass(e.severity)}">${e.severity || "—"}</span></td>
        <td class="conf muted">${e.confidence != null ? Number(e.confidence).toFixed(2) : "—"}</td>
        <td class="conf muted">${fmtBytes(e.bytes)}</td>
        <td class="muted">${e.reason || ""}</td>`;
      tr.style.cursor = "pointer";
      tr.addEventListener("click", () => openEvidence(e.chunk_id));
      $("findings-body").prepend(tr);
      break;
    }

    case "run_finished":
      setPhase("done");
      updateKpis();
      pushFeed("chunk-feed",
        `<span class="badge non_ai">■ finished</span> ` +
        `<span class="muted">${fmt(e.rows)} rows · ${fmt(e.chunks)} chunks · ` +
        `${fmt(e.ai)} AI · ${fmt(e.findings)} findings · ${(e.elapsed || 0).toFixed(1)}s</span>`);
      loadReview();
      loadDomains();
      break;

    case "stdout":
      // Uncomment to see raw analyser output in the chunk feed.
      // pushFeed("chunk-feed", `<span class="muted mono">${e.line}</span>`);
      break;
  }
}

connect();
setPhase("1");

/* ============================================================
   TAB: Findings (historical / paginated)
   ============================================================ */
async function loadFindings() {
  const sev = $("f-severity").value;
  const app = $("f-app").value.trim();
  const user = $("f-user").value.trim();
  const conf = $("f-conf").value;

  const qs = new URLSearchParams({ limit: "200" });
  if (sev) qs.set("severity", sev);
  if (app) qs.set("application", app);
  if (user) qs.set("user", user);
  if (conf && Number(conf) > 0) qs.set("min_confidence", conf);

  const body = $("all-findings-body");
  body.innerHTML = '<tr><td colspan="8" class="placeholder">loading…</td></tr>';
  try {
    const r = await fetch(apiUrl("/findings?" + qs));
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    body.innerHTML = "";
    if (!data.items.length) {
      body.innerHTML = '<tr><td colspan="8" class="placeholder">no matches</td></tr>';
      return;
    }
    data.items.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">#${row._chunk_id}</td>
        <td>${row._user || "—"}</td>
        <td>${row.application || "—"}</td>
        <td class="muted">${row._match_tier || "—"}</td>
        <td><span class="pill ${sevClass(row.severity)}">${row.severity || "—"}</span></td>
        <td class="conf muted">${row.confidence != null ? Number(row.confidence).toFixed(2) : "—"}</td>
        <td class="conf muted">${fmtBytes(row._bytes)}</td>
        <td><a class="link" data-chunk="${row._chunk_id}">view evidence</a></td>`;
      tr.querySelector("a").addEventListener("click", () => openEvidence(row._chunk_id));
      body.appendChild(tr);
    });
  } catch (err) {
    body.innerHTML = `<tr><td colspan="8" class="placeholder">error: ${err}</td></tr>`;
  }
}
$("f-apply").addEventListener("click", loadFindings);

/* ============================================================
   TAB: Review queue
   ============================================================ */
async function loadReview() {
  const body = $("review-body");
  body.innerHTML = '<tr><td colspan="5" class="placeholder">loading…</td></tr>';
  try {
    const r = await fetch(apiUrl("/review?limit=500"));
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    $("review-count").textContent = data.total + " chunks";
    body.innerHTML = "";
    if (!data.items.length) {
      body.innerHTML = '<tr><td colspan="5" class="placeholder">nothing to review</td></tr>';
      return;
    }
    data.items.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">#${row.chunk_id}</td>
        <td class="muted">${row.sensor || "—"}</td>
        <td>${row.user || "—"}</td>
        <td class="mono muted">${(row.unknown_domains || []).join("<br>")}</td>
        <td class="conf muted">${row.event_count || 0}</td>`;
      body.appendChild(tr);
    });
  } catch (err) {
    body.innerHTML = `<tr><td colspan="5" class="placeholder">error: ${err}</td></tr>`;
  }
}

/* ============================================================
   TAB: Domain cache
   ============================================================ */
async function loadDomains() {
  const status = $("d-status").value;
  const qs = new URLSearchParams({ limit: "1000" });
  if (status) qs.set("status", status);

  const body = $("domains-body");
  body.innerHTML = '<tr><td colspan="5" class="placeholder">loading…</td></tr>';
  try {
    const r = await fetch(apiUrl("/domains?" + qs));
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    body.innerHTML = "";
    if (!data.items.length) {
      body.innerHTML = '<tr><td colspan="5" class="placeholder">no domains cached</td></tr>';
      return;
    }
    data.items.forEach((row) => {
      const tr = document.createElement("tr");
      const t = row.last_attempt ? new Date(row.last_attempt * 1000).toLocaleString() : "—";
      tr.innerHTML = `
        <td class="mono">${row.domain}</td>
        <td><span class="badge ${row.status}">${(row.status || "unknown").toUpperCase()}</span></td>
        <td>${row.app || "—"}</td>
        <td class="conf muted">${(row.confidence || 0).toFixed(2)}</td>
        <td class="muted">${t}</td>`;
      body.appendChild(tr);
    });
  } catch (err) {
    body.innerHTML = `<tr><td colspan="5" class="placeholder">error: ${err}</td></tr>`;
  }
}
$("d-apply").addEventListener("click", loadDomains);

/* ============================================================
   Evidence drawer
   ============================================================ */
async function openEvidence(chunkId) {
  if (chunkId == null) return;
  const drawer = $("drawer"), backdrop = $("backdrop");
  drawer.classList.add("open");
  backdrop.classList.add("on");
  drawer.setAttribute("aria-hidden", "false");

  $("drawer-title").textContent = "Evidence — chunk #" + chunkId;
  $("drawer-body").innerHTML = '<div class="muted">loading…</div>';

  try {
    const r = await fetch(apiUrl(`/findings/${chunkId}/evidence`));
    if (!r.ok) throw new Error("HTTP " + r.status);
    const ev = await r.json();
    $("drawer-body").innerHTML = renderEvidence(ev);
  } catch (err) {
    $("drawer-body").innerHTML = `<div class="muted">Failed to load evidence: ${err}</div>`;
  }
}

function closeDrawer() {
  $("drawer").classList.remove("open");
  $("backdrop").classList.remove("on");
  $("drawer").setAttribute("aria-hidden", "true");
}
$("drawer-close").addEventListener("click", closeDrawer);
$("backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderEvidence(ev) {
  const apps = ev.matched_applications || {};
  const domains = ev.domains || [];
  const samples = ev.sample_events || [];

  const appRows = Object.entries(apps).map(([name, m]) => `
    <tr>
      <td><strong>${esc(name)}</strong></td>
      <td><span class="badge ${m.approved ? "non_ai" : "AI"}">${m.approved ? "approved" : "not approved"}</span></td>
      <td class="muted">${esc(m.source)}</td>
      <td class="conf">${(m.confidence || 0).toFixed(2)}</td>
    </tr>`).join("");

  const domainRows = domains.map((d) => `
    <tr>
      <td class="mono">${esc(d.domain)}</td>
      <td class="conf muted">${d.event_count}</td>
      <td class="conf muted">${fmtBytes(d.total_bytes)}</td>
      <td class="muted">${esc(d.first_seen) || "—"}</td>
      <td class="muted">${esc(d.last_seen)  || "—"}</td>
      <td class="muted">${esc((d.users || []).join(", ")) || "—"}</td>
    </tr>`).join("");

  const sampleRows = samples.map((s) => `
    <tr>
      <td class="muted">${esc(s.timestamp) || "—"}</td>
      <td class="mono">${esc(s.domain) || "—"}</td>
      <td>${esc(s.user) || "—"}</td>
      <td class="conf muted">${fmtBytes(s.bytes)}</td>
      <td class="muted">${esc(s.log_format) || "—"}</td>
    </tr>`).join("");

  return `
    <div class="muted" style="margin-bottom:12px">
      <strong>${esc(ev.sensor)}</strong> · ${esc(ev.source)} · user <strong>${esc(ev.user) || "—"}</strong><br>
      ${esc(ev.start_ts)} → ${esc(ev.end_ts)} · ${ev.event_count} events · ${fmtBytes(ev.total_bytes)}
      ${ev.reanalysis ? '<br><span class="badge REVIEW">reanalysis</span> ' + esc(ev.reanalysis_reason || "") : ""}
    </div>

    ${appRows ? `<h3>Matched applications</h3>
      <table>
        <thead><tr><th>App</th><th>Status</th><th>Source</th><th>Conf</th></tr></thead>
        <tbody>${appRows}</tbody>
      </table>` : ""}

    ${domainRows ? `<h3>Domains in this chunk</h3>
      <table>
        <thead><tr><th>Domain</th><th>Events</th><th>Bytes</th><th>First</th><th>Last</th><th>Users</th></tr></thead>
        <tbody>${domainRows}</tbody>
      </table>` : ""}

    ${sampleRows ? `<h3>Sample events</h3>
      <table>
        <thead><tr><th>Timestamp</th><th>Domain</th><th>User</th><th>Bytes</th><th>Format</th></tr></thead>
        <tbody>${sampleRows}</tbody>
      </table>` : ""}

    <h3>Raw</h3>
    <pre>${esc(JSON.stringify(ev, null, 2))}</pre>
  `;
}

/* ============================================================
   TAB: Upload & Run
   ============================================================ */
const fileInput  = $("file-input");
const fileDrop   = $("file-drop");
const fileLabel  = $("file-label");
const upForm     = $("upload-form");
const upSubmit   = $("up-submit");
const upCancel   = $("up-cancel");
const upResult   = $("upload-result");
const upStatus   = $("upload-status");

let uploadInFlight = false;

["dragenter", "dragover"].forEach((ev) => {
  fileDrop.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    fileDrop.classList.add("hover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  fileDrop.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    fileDrop.classList.remove("hover");
  });
});
fileDrop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    updateFileLabel();
  }
});
fileInput.addEventListener("change", updateFileLabel);

function updateFileLabel() {
  const f = fileInput.files[0];
  fileLabel.textContent = f ? f.name : "Click to choose a CSV, or drag & drop";
}

upForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (uploadInFlight) return;

  const f = fileInput.files[0];
  if (!f) {
    showUploadResult("err", "Please choose a CSV file first.");
    return;
  }

  const fd = new FormData();
  fd.append("file", f);
  fd.append("classify_model", $("up-classify").value.trim() || "qwen2.5:14b");
  fd.append("reset_cache", $("up-reset-cache").checked ? "true" : "false");
  const ly = $("up-log-year").value.trim();
  if (ly) fd.append("log_year", ly);

  uploadInFlight = true;
  upSubmit.disabled = true;
  upCancel.hidden = false;
  upStatus.textContent = "uploading…";
  upResult.hidden = true;

  try {
    const r = await fetch(apiUrl("/upload"), {
      method: "POST",
      body: fd
    });

    // Read body as text first so an empty or non-JSON response gives a real error.
    const raw = await r.text();

    if (!r.ok) {
      let detail = raw;
      try { detail = JSON.parse(raw).detail || raw; } catch { /* keep raw */ }
      throw new Error(detail || `HTTP ${r.status} ${r.statusText} (empty body — check uvicorn console)`);
    }
    if (!raw) {
      throw new Error("server returned an empty 200 response");
    }

    const data = JSON.parse(raw);

    upStatus.textContent = "running";
    showUploadResult("ok",
      `Started analysis on <strong>${esc(data.csv)}</strong> — pid ${data.pid}.<br>` +
      `<span class="muted">Switch to the <a class="link" id="goto-live">Live</a> tab to watch progress.</span>`);
    $("goto-live").addEventListener("click", () => {
      document.querySelector('.tab[data-tab="live"]').click();
    });

    pollRunUntilDone();

  } catch (err) {
    showUploadResult("err", `Upload failed: ${err.message}`);
    upSubmit.disabled = false;
    upCancel.hidden = true;
    upStatus.textContent = "idle";
    uploadInFlight = false;
  }
});

upCancel.addEventListener("click", () => {
  upSubmit.disabled = false;
  upCancel.hidden = true;
  upStatus.textContent = "idle";
  uploadInFlight = false;
});

async function pollRunUntilDone() {
  try {
    const r = await fetch(apiUrl("/run-status"));
    const s = await r.json();
    if (s.running) {
      setTimeout(pollRunUntilDone, 3000);
      return;
    }
    upStatus.textContent = s.exit_code === 0 ? "finished" : `exited (${s.exit_code})`;
    upSubmit.disabled = false;
    upCancel.hidden = true;
    uploadInFlight = false;
    showUploadResult(
      s.exit_code === 0 ? "ok" : "err",
      `Run on <strong>${esc(s.csv)}</strong> finished with exit code ${s.exit_code}. ` +
      `Findings are in the <a class="link" id="goto-findings">Findings</a> tab.`
    );
    const link = $("goto-findings");
    if (link) link.addEventListener("click", () => {
      document.querySelector('.tab[data-tab="findings"]').click();
    });
    loadFindings();
    loadReview();
    loadDomains();
  } catch {
    setTimeout(pollRunUntilDone, 5000);
  }
}

function showUploadResult(kind, html) {
  upResult.className = "upload-result " + kind;
  upResult.innerHTML = html;
  upResult.hidden = false;
}

/* ============================================================
   Initial tab loads
   ============================================================ */
loadReview();
loadDomains();
