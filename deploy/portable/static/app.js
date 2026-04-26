// CWformer Portable -- single-page UI logic.

const sock = io({ transports: ["websocket", "polling"] });

const $ = (id) => document.getElementById(id);

const stateBadge   = $("state-badge");
const syncBadge    = $("sync-badge");
const gpioBadge    = $("gpio-badge");

const modelSelect  = $("model-select");
const modelHint    = $("model-hint");
const sourceKind   = $("source-kind");
const deviceSelect = $("device-select");
const fileSelect   = $("file-select");
const wavHint      = $("wav-hint");
const commandInput = $("command-input");

const btnRefreshModels  = $("btn-refresh-models");
const btnRefreshDevices = $("btn-refresh-devices");
const btnRefreshWavs    = $("btn-refresh-wavs");

const btnStart     = $("btn-start");
const btnStop      = $("btn-stop");
const btnClear     = $("btn-clear");
const btnSpec      = $("btn-spec-toggle");
const specImg      = $("spectrogram-img");

const rxText       = $("rx-text");
const recentDiv    = $("recent-callsigns");

const myCall       = $("mycall");
const wpm          = $("wpm");
const hisCall      = $("hiscall");
const rstSent      = $("rst-sent");
const rstRcvd      = $("rst-rcvd");
const opName       = $("op-name");
const opQth        = $("op-qth");
const macroDiv     = $("macro-buttons");
const freetext     = $("freetext");
const btnSendtext  = $("btn-sendtext");
const btnCancel    = $("btn-cancel");
const txLog        = $("tx-log");

const logFilename  = $("log-filename");
const btnRenameLog = $("btn-rename-log");
const btnDownloadLog = $("btn-download-log");
const logPath      = $("log-path");

let macros = [];
let specEnabled = true;

const RX_MAX_CHARS = 50000;
const recentCallsigns = new Map();   // call -> button element
const CALLSIGN_RE = /\b(?:[A-Z]{1,2}\d[A-Z]{1,3}|\d[A-Z]\d[A-Z]{1,3})\b/g;

// ---- Rest API helpers ----

async function getJSON(path) {
  const r = await fetch(path);
  return r.json();
}

function fmtSize(bytes) {
  const n = Number(bytes) || 0;
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + " KB";
  return n + " B";
}

async function refreshModels(preferredPath) {
  const data = await getJSON("/api/models");
  const items = data.models || [];
  // Sort by relative path so subdir grouping reads naturally.
  items.sort((a, b) =>
    (a.rel || a.name).localeCompare(b.rel || b.name));
  modelSelect.innerHTML = items.length
    ? items.map((m) => {
        const label = (m.rel && m.rel !== m.name) ? m.rel : m.name;
        return `<option value="${escapeAttr(m.path)}">${escapeHtml(label)} (${fmtSize(m.size)})</option>`;
      }).join("")
    : `<option value="">(no .onnx files under ${escapeHtml(data.scan_root || "")})</option>`;
  if (preferredPath) {
    // setting .value to a path that doesn't exist in the list is a no-op,
    // which is the right fallback (first option remains selected).
    modelSelect.value = preferredPath;
  }
  if (modelHint) {
    modelHint.textContent = data.scan_root
      ? `Scanning ${data.scan_root} (depth ${data.scan_depth})`
      : "";
  }
}

async function refreshDevices(preferredId) {
  const data = await getJSON("/api/devices");
  const devices = data.devices || [];
  deviceSelect.innerHTML = `<option value="">(default)</option>` +
    devices.map((d) =>
      `<option value="${d.id}">${d.id}: ${escapeHtml(d.name)}${d.default ? " *" : ""}</option>`
    ).join("");
  if (preferredId != null && preferredId !== "") {
    deviceSelect.value = String(preferredId);
  }
}

async function refreshWavs(preferredPath) {
  const data = await getJSON("/api/wavs");
  const items = data.wavs || [];
  items.sort((a, b) =>
    (a.rel || a.name).localeCompare(b.rel || b.name));
  fileSelect.innerHTML = items.length
    ? items.map((w) => {
        const label = (w.rel && w.rel !== w.name) ? w.rel : w.name;
        return `<option value="${escapeAttr(w.path)}">${escapeHtml(label)} (${fmtSize(w.size)})</option>`;
      }).join("")
    : `<option value="">(no .wav/.flac/.ogg files under ${escapeHtml(data.scan_root || data.dir || "")})</option>`;
  if (preferredPath) fileSelect.value = preferredPath;
  if (wavHint) {
    wavHint.textContent = data.scan_root
      ? `Scanning ${data.scan_root} (depth ${data.scan_depth})`
      : "";
  }
}

async function refreshLists() {
  let status = {};
  try { status = await getJSON("/api/status"); }
  catch (e) { console.error("status:", e); }

  await Promise.all([
    refreshModels(status.model_path).catch((e) => console.error("models:", e)),
    refreshDevices(status.source ? status.source.device_id : null)
      .catch((e) => console.error("devices:", e)),
    refreshWavs(status.source ? status.source.file_path : "")
      .catch((e) => console.error("wavs:", e)),
  ]);

  if (status.source) {
    sourceKind.value = status.source.kind || "device";
    commandInput.value = status.source.command || "";
    updateSourceVisibility();
  }
  if (status.log_path) logPath.textContent = status.log_path;
  if (status.log_filename) logFilename.value = status.log_filename;

  if (typeof status.running === "boolean") {
    setDecoderState(status.running ? "running" : "idle", "");
  }
  if (typeof status.gpio_enabled === "boolean") {
    setGpioBadge(status.gpio_enabled);
  }
  if (status.time_sync) updateSync(status.time_sync);
}

function updateSourceVisibility() {
  document.querySelectorAll(".source-device").forEach((el) =>
    el.hidden = sourceKind.value !== "device");
  document.querySelectorAll(".source-file").forEach((el) =>
    el.hidden = sourceKind.value !== "file");
  document.querySelectorAll(".source-command").forEach((el) =>
    el.hidden = sourceKind.value !== "command");
}
sourceKind.addEventListener("change", updateSourceVisibility);

// ---- Helpers ----

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

function appendRxText(t) {
  const cur = rxText.textContent;
  let next = cur + t;
  if (next.length > RX_MAX_CHARS) next = next.slice(-RX_MAX_CHARS);
  rxText.textContent = next;
  rxText.scrollTop = rxText.scrollHeight;

  // Detect new callsigns from accumulated text. Cheap regex on the full
  // text each update -- fine at the rate text arrives.
  const found = next.toUpperCase().match(CALLSIGN_RE) || [];
  for (const c of found) {
    if (!recentCallsigns.has(c)) addRecentCallsign(c);
  }
}

function addRecentCallsign(call) {
  const btn = document.createElement("button");
  btn.textContent = call;
  btn.className = "callsign-btn";
  btn.title = "Click to fill 'His call'";
  btn.addEventListener("click", () => { hisCall.value = call; });
  recentDiv.appendChild(btn);
  recentCallsigns.set(call, btn);
  // Cap recent list to last 30 calls
  while (recentDiv.children.length > 30) {
    const first = recentDiv.firstChild;
    recentCallsigns.delete(first.textContent);
    recentDiv.removeChild(first);
  }
}

function setDecoderState(state, msg) {
  stateBadge.textContent = state + (msg ? ` -- ${msg}` : "");
  stateBadge.className = `badge state-${state}`;
  btnStart.disabled = (state === "running");
  btnStop.disabled  = (state !== "running");
}

function setGpioBadge(active) {
  if (active) {
    gpioBadge.textContent = "TX active";
    gpioBadge.className = "badge gpio-active";
    gpioBadge.title = "GPIO is driving the keyer pin";
  } else {
    gpioBadge.textContent = "TX stub (no GPIO)";
    gpioBadge.className = "badge gpio-stub";
    gpioBadge.title = "GPIO disabled — Send buttons run the keyer "
      + "logic but do not drive any output. Messages will not be transmitted.";
  }
}

function updateSync(s) {
  if (!s) {
    syncBadge.textContent = "sync ?";
    syncBadge.className = "badge sync-unknown";
    syncBadge.title = "";
    return;
  }
  const src = s.source && s.source !== "none" ? s.source : "no ref";
  syncBadge.textContent = `${s.state} · ${src}`;
  syncBadge.className = `badge sync-${s.state}`;
  syncBadge.title = s.detail || "";
}

// ---- Socket events ----

sock.on("init", (data) => {
  myCall.value = data.config.callsign || "";
  wpm.value = data.config.wpm || 20;
  logFilename.value = data.config.log_filename || "";
  specEnabled = !!data.config.spectrogram_enabled;
  btnSpec.textContent = specEnabled ? "Hide spectrogram" : "Show spectrogram";
  specImg.style.display = specEnabled ? "" : "none";
  macros = data.macros || [];
  renderMacros();
  setDecoderState(data.decoder_running ? "running" : "idle", "");
  setGpioBadge(data.gpio_enabled);
  if (data.time_sync) updateSync(data.time_sync);
});

sock.on("decoded_text", (m) => appendRxText(m.text));
sock.on("decoder_state", (m) => setDecoderState(m.state, m.message));
sock.on("text_cleared", () => { rxText.textContent = ""; });
sock.on("spectrogram", (m) => { if (specEnabled) specImg.src = m.img; });
sock.on("time_sync", updateSync);

sock.on("config_updated", (m) => {
  if (m.callsign != null) myCall.value = m.callsign;
  if (m.wpm != null) wpm.value = m.wpm;
  if (m.log_filename != null) {
    logFilename.value = m.log_filename;
  }
  if (m.log_path != null) logPath.textContent = m.log_path;
});

sock.on("tx_text", (m) => {
  const ts = new Date().toISOString().replace("T", " ").slice(0, 19);
  txLog.textContent += `[${ts}] ${m.text}\n`;
  txLog.scrollTop = txLog.scrollHeight;
});

sock.on("key_state", (m) => {
  document.body.classList.toggle("keying", !!m.on);
});

// ---- Buttons ----

btnStart.addEventListener("click", () => {
  const payload = {
    kind: sourceKind.value,
    model_path: modelSelect.value,
    device_id: deviceSelect.value || null,
    file_path: fileSelect.value || "",
    command: commandInput.value || "",
  };
  sock.emit("start_decode", payload, (resp) => {
    if (resp && !resp.ok) alert("Start failed: " + (resp.error || ""));
  });
});

btnStop.addEventListener("click", () => sock.emit("stop_decode", {}));
btnClear.addEventListener("click", () => {
  sock.emit("clear_text", {});
  rxText.textContent = "";
  recentDiv.innerHTML = "";
  recentCallsigns.clear();
});

btnSpec.addEventListener("click", () => {
  specEnabled = !specEnabled;
  btnSpec.textContent = specEnabled ? "Hide spectrogram" : "Show spectrogram";
  specImg.style.display = specEnabled ? "" : "none";
  sock.emit("toggle_spectrogram", { on: specEnabled });
});

myCall.addEventListener("change", () =>
  sock.emit("set_callsign", { callsign: myCall.value }));
wpm.addEventListener("change", () =>
  sock.emit("set_wpm", { wpm: parseInt(wpm.value, 10) || 20 }));

btnRenameLog.addEventListener("click", () => {
  const name = (logFilename.value || "").trim();
  if (!name) return;
  sock.emit("set_log_filename", { name }, (r) => {
    if (r && !r.ok) alert("Rename failed: " + (r.error || ""));
  });
});

btnDownloadLog.addEventListener("click", () => {
  const name = (logFilename.value || "").trim();
  btnDownloadLog.href = "/api/log/download?name=" + encodeURIComponent(name);
});

btnSendtext.addEventListener("click", () => {
  const t = (freetext.value || "").trim();
  if (!t) return;
  sock.emit("send_text", { text: t }, (r) => {
    if (r && !r.ok) alert("Send failed: " + (r.error || ""));
  });
  freetext.value = "";
});

btnCancel.addEventListener("click", () => sock.emit("cancel_send", {}));

if (btnRefreshModels) {
  btnRefreshModels.addEventListener("click", () => {
    refreshModels(modelSelect.value).catch((e) => console.error("models:", e));
  });
}
if (btnRefreshDevices) {
  btnRefreshDevices.addEventListener("click", () => {
    refreshDevices(deviceSelect.value).catch((e) => console.error("devices:", e));
  });
}
if (btnRefreshWavs) {
  btnRefreshWavs.addEventListener("click", () => {
    refreshWavs(fileSelect.value).catch((e) => console.error("wavs:", e));
  });
}

function renderMacros() {
  macroDiv.innerHTML = "";
  for (const m of macros) {
    const btn = document.createElement("button");
    btn.textContent = m.name;
    btn.title = m.text;
    btn.addEventListener("click", () => {
      sock.emit("send_macro", {
        text: m.text,
        his_call: hisCall.value,
        rst_sent: rstSent.value || "599",
        rst_rcvd: rstRcvd.value || "599",
        name: opName.value,
        qth: opQth.value,
      }, (r) => {
        if (r && !r.ok) alert("Macro failed: " + (r.error || ""));
      });
    });
    macroDiv.appendChild(btn);
  }
}

// Init
refreshLists().catch((e) => console.error("refreshLists:", e));
