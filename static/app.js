// Sidekick frontend: talks to the FastAPI backend in app.py.
// One session per tab; while a turn runs, the page polls for the plan, live activity and files.

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"),
  chat: $("chat"),
  empty: $("empty"),
  form: $("composer"),
  message: $("message"),
  criteria: $("criteria"),
  hint: $("hint"),
  go: $("go"),
  stop: $("stop"),
  reset: $("reset"),
  planBody: $("plan-body"),
  planCount: $("plan-count"),
  progress: $("progress-bar"),
  filesBody: $("files-body"),
  filesCount: $("files-count"),
  toast: $("toast"),
};

const ICONS = {
  bolt: '<svg viewBox="0 0 24 24"><path d="M13.5 2 5 14h6l-1 8 8.5-12h-6z"/></svg>',
  alert: '<svg viewBox="0 0 24 24"><path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12.5 10 17 19 7"/></svg>',
  file: '<svg class="file-icon" viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>',
  download: '<svg class="file-download" viewBox="0 0 24 24"><path d="M12 4v11m-5-5 5 5 5-5M5 20h14"/></svg>',
};

const VERDICTS = {
  met: "Met",
  needs_input: "Needs you",
  not_met: "Not fully met",
};

let sessionId = null;
let history = [];
let activity = [];
let paused = false;
let busy = false;
let starting = true;
let pollTimer = null;
let renderedCount = 0; // entries already on screen, so only new ones animate in
let knownFiles = null; // file paths seen when the page loaded, to highlight new ones

// ---------- API ----------

async function api(path, { method = "POST", body } = {}) {
  const response = await fetch(`/api${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

async function startSession() {
  starting = true;
  setStatus("connecting", "Starting tools…");
  updateControls();
  try {
    const data = await api("/sessions");
    sessionId = data.session_id;
    setStatus("ready", "Ready");
  } catch (error) {
    setStatus("error", "Tools failed to start");
    showToast(error.message);
  } finally {
    starting = false;
    updateControls();
  }
}

function closeSession() {
  if (sessionId) navigator.sendBeacon(`/api/sessions/${sessionId}/close`);
  sessionId = null;
}

// ---------- Rendering helpers ----------

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) {
    return DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }));
  }
  return `<p>${escapeHtml(text).replace(/\n/g, "<br>")}</p>`;
}

function stepsList(steps) {
  return `<ul class="steps">${steps.map((s) =>
    `<li class="${escapeHtml(s.state || "done")}"><span class="step-mark"></span><span>${escapeHtml(s.label ?? s)}</span></li>`
  ).join("")}</ul>`;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function timeAgo(iso) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(iso).toLocaleDateString("en-NZ", { day: "numeric", month: "short" });
}

// ---------- Conversation entries ----------

function userNode(entry) {
  const node = document.createElement("div");
  node.className = "msg msg-user";
  const criteria = entry.criteria
    ? `<div class="criteria-tag"><span class="criteria-label">Success criteria</span>${escapeHtml(entry.criteria)}</div>`
    : "";
  node.innerHTML = `<div class="user-block"><div class="bubble">${escapeHtml(entry.content)}</div>${criteria}</div>`;
  return node;
}

function assistantNode(entry) {
  const node = document.createElement("div");
  node.className = "msg msg-assistant";
  const steps = entry.steps || [];
  const stepsHtml = steps.length
    ? `<details><summary>${steps.length} step${steps.length === 1 ? "" : "s"}</summary>${stepsList(steps)}</details>`
    : "";
  node.innerHTML = `
    <div class="avatar" aria-hidden="true">${ICONS.bolt}</div>
    <div class="reply">
      <div class="bubble">${renderMarkdown(entry.content)}</div>
      <div class="reply-tools">
        <button type="button" class="copy-btn" aria-label="Copy reply">${ICONS.copy}<span>Copy</span></button>
        ${stepsHtml}
      </div>
    </div>`;
  node.querySelectorAll(".bubble a").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
  node.querySelector(".copy-btn").addEventListener("click", (e) => copyText(entry.content, e.currentTarget));
  return node;
}

function evaluatorNode(entry) {
  const node = document.createElement("div");
  node.className = "note";
  const status = VERDICTS[entry.status] ? entry.status : "met";
  const attempts = entry.status === "not_met" && entry.attempts
    ? ` <span class="note-attempts">after ${entry.attempts} attempts</span>` : "";
  node.innerHTML = `
    <span class="note-label">Evaluator</span>
    <span class="badge badge-${status}">${VERDICTS[status]}</span>
    <span>${escapeHtml(entry.content)}${attempts}</span>`;
  return node;
}

function describeAction(action) {
  const args = action.args || {};
  if (action.name === "request_human_help") {
    return { title: "Sidekick needs your help in the browser", body: args.instructions, help: true };
  }
  if (action.name === "send_push_notification") {
    return { title: "Send this notification to your phone?", body: args.text, help: false };
  }
  return { title: `Run ${action.name}?`, body: JSON.stringify(args, null, 2), help: false };
}

function approvalNode(entry) {
  const node = document.createElement("div");
  const actions = (entry.actions || []).map(describeAction);
  const help = actions.some((a) => a.help);
  const pending = entry.status === "pending";
  node.className = `approval${pending ? "" : " resolved"}`;

  const body = actions.map((a) => `
    <div class="approval-action">
      <div class="approval-title">${ICONS.alert}${escapeHtml(a.title)}</div>
      <div class="approval-body">${escapeHtml(a.body ?? "")}</div>
    </div>`).join("");

  if (!pending) {
    const outcome = entry.status === "approved"
      ? (help ? "You said you had done it." : "You approved this.")
      : (help ? "You said you could not do it." : "You declined this.");
    node.innerHTML = `${body}<div class="approval-outcome">${outcome}</div>`;
    return node;
  }

  node.innerHTML = `${body}
    <div class="approval-controls">
      <input type="text" class="approval-note" placeholder="${help ? "Anything Sidekick should know? (optional)" : "Reason if declining (optional)"}">
      <button type="button" class="btn btn-ghost btn-sm" data-approve="false">${help ? "I can’t do it" : "Decline"}</button>
      <button type="button" class="btn btn-approve btn-sm" data-approve="true">${help ? "I’ve done it" : "Approve"}</button>
    </div>`;
  node.querySelectorAll("[data-approve]").forEach((button) => {
    button.disabled = busy;
    button.addEventListener("click", () =>
      decide(button.dataset.approve === "true", node.querySelector(".approval-note").value));
  });
  return node;
}

function noticeNode(entry) {
  const node = document.createElement("div");
  node.className = "notice";
  node.textContent = entry.content;
  return node;
}

const RENDERERS = {
  user: userNode,
  assistant: assistantNode,
  evaluator: evaluatorNode,
  approval: approvalNode,
  notice: noticeNode,
};

function workingNode() {
  const node = document.createElement("div");
  node.className = "working";
  node.id = "working";
  node.innerHTML = `
    <div class="avatar" aria-hidden="true">${ICONS.bolt}</div>
    <div class="working-body">
      <div class="working-title"><span>Working on it</span><span class="dots"><i></i><i></i><i></i></span></div>
      <div class="working-steps"></div>
    </div>`;
  return node;
}

function renderActivity() {
  const holder = document.querySelector("#working .working-steps");
  if (!holder) return;
  const nearBottom = els.chat.scrollHeight - els.chat.scrollTop - els.chat.clientHeight < 80;
  holder.innerHTML = activity.length ? stepsList(activity.slice(-8)) : "";
  if (nearBottom) els.chat.scrollTop = els.chat.scrollHeight;
}

function renderChat({ pendingUser = null, error = null } = {}) {
  els.chat.querySelectorAll(":scope > :not(#empty)").forEach((n) => n.remove());
  const entries = pendingUser ? [...history, pendingUser] : history;
  els.empty.hidden = entries.length > 0;

  entries.forEach((entry, i) => {
    const node = (RENDERERS[entry.role] || noticeNode)(entry);
    if (i < renderedCount) node.classList.add("settled");
    els.chat.appendChild(node);
  });
  renderedCount = entries.length;

  if (error) {
    const node = document.createElement("div");
    node.className = "error-card";
    node.textContent = error;
    els.chat.appendChild(node);
  }
  if (busy) {
    els.chat.appendChild(workingNode());
    renderActivity();
  }
  requestAnimationFrame(() => (els.chat.scrollTop = els.chat.scrollHeight));
}

function renderPlan(todos = []) {
  if (!todos.length) {
    els.planBody.innerHTML = '<p class="plan-empty">Sidekick will write its plan here as it works.</p>';
    els.planCount.textContent = "";
    els.progress.style.width = "0";
    return;
  }
  const done = todos.filter((t) => t.status === "completed").length;
  els.planCount.textContent = `${done} / ${todos.length}`;
  els.progress.style.width = `${(done / todos.length) * 100}%`;
  els.planBody.innerHTML = "<ul>" + todos.map((t) =>
    `<li class="${escapeHtml(t.status)}"><span class="mark"></span><span>${escapeHtml(t.content)}</span></li>`
  ).join("") + "</ul>";
}

async function refreshFiles() {
  let files;
  try {
    ({ files } = await api("/files", { method: "GET" }));
  } catch {
    return; // the list refreshes again on the next poll
  }
  if (knownFiles === null) knownFiles = new Set(files.map((f) => f.path));
  els.filesCount.textContent = files.length ? String(files.length) : "";
  if (!files.length) {
    els.filesBody.innerHTML = '<p class="plan-empty">Files Sidekick saves to its sandbox will appear here.</p>';
    return;
  }
  els.filesBody.innerHTML = '<ul class="file-list">' + files.map((f) => `
    <li class="${knownFiles.has(f.path) ? "" : "fresh"}">
      <a href="/api/files/${f.path.split("/").map(encodeURIComponent).join("/")}" download title="Download ${escapeHtml(f.path)}">
        ${ICONS.file}
        <span class="file-main">
          <span class="file-name">${escapeHtml(f.path)}</span>
          <span class="file-meta">${formatSize(f.size)} · ${timeAgo(f.modified)}</span>
        </span>
        ${ICONS.download}
      </a>
    </li>`).join("") + "</ul>";
}

function setStatus(state, text) {
  els.status.dataset.state = state;
  els.status.querySelector(".status-text").textContent = text;
}

function updateControls() {
  // While an action waits for a decision the agent is mid tool call, so the only ways on are the card or Reset
  els.go.disabled = busy || starting || !sessionId || paused;
  els.go.hidden = busy;
  els.stop.hidden = !busy;
  els.reset.disabled = busy || starting;
  els.hint.innerHTML = paused
    ? "Respond to the request above to continue"
    : "<kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line";
  els.chat.querySelectorAll(".approval [data-approve]").forEach((b) => (b.disabled = busy));
}

let toastTimer;
function showToast(text) {
  els.toast.textContent = text;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), 6000);
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    button.classList.add("copied");
    button.innerHTML = `${ICONS.check}<span>Copied</span>`;
    setTimeout(() => {
      button.classList.remove("copied");
      button.innerHTML = `${ICONS.copy}<span>Copy</span>`;
    }, 1600);
  } catch {
    showToast("Could not copy to the clipboard.");
  }
}

// ---------- Polling while working ----------

function startPolling() {
  stopPolling();
  let tick = 0;
  pollTimer = setInterval(async () => {
    if (!sessionId) return;
    try {
      const state = await api(`/sessions/${sessionId}/state`, { method: "GET" });
      renderPlan(state.todos);
      activity = state.activity || [];
      renderActivity();
    } catch { /* a missed poll is harmless */ }
    if (++tick % 3 === 0) refreshFiles();
  }, 1000);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

// ---------- Actions ----------

async function runWork(path, body, pendingUser = null) {
  busy = true;
  activity = [];
  setStatus("working", "Working…");
  updateControls();
  renderChat({ pendingUser });
  startPolling();

  try {
    const state = await api(`/sessions/${sessionId}${path}`, { body });
    history = state.history;
    paused = state.paused;
    busy = false;
    renderPlan(state.todos);
    renderChat();
    setStatus(paused ? "waiting" : "ready", paused ? "Waiting for you" : "Ready");
    if (paused) els.chat.querySelector(".approval:not(.resolved) .btn-approve")?.focus();
    return true;
  } catch (error) {
    busy = false;
    renderChat({ pendingUser, error: error.message });
    setStatus("ready", "Ready");
    return false;
  } finally {
    stopPolling();
    updateControls();
    refreshFiles();
  }
}

async function send() {
  const message = els.message.value.trim();
  if (!message || busy || paused || !sessionId) return;
  const success_criteria = els.criteria.value.trim();
  els.message.value = "";
  els.criteria.value = "";
  autoGrow();
  const ok = await runWork("/turn", { message, success_criteria }, { role: "user", content: message, criteria: success_criteria });
  if (!ok && !els.message.value && !els.criteria.value) {
    // give the user their text back so they can retry
    els.message.value = message;
    els.criteria.value = success_criteria;
    autoGrow();
  }
  els.message.focus();
}

async function decide(approve, note) {
  if (busy || !sessionId) return;
  await runWork("/decision", { approve, note: note.trim() });
}

async function stop() {
  if (!busy || !sessionId) return;
  els.stop.disabled = true;
  setStatus("working", "Stopping…");
  try {
    await api(`/sessions/${sessionId}/stop`);
  } catch (error) {
    showToast(error.message);
  } finally {
    els.stop.disabled = false;
  }
}

async function reset() {
  if (busy) return;
  closeSession();
  history = [];
  activity = [];
  paused = false;
  renderedCount = 0;
  els.message.value = "";
  els.criteria.value = "";
  autoGrow();
  renderChat();
  renderPlan([]);
  updateControls();
  await startSession();
  els.message.focus();
}

// ---------- Wiring ----------

function autoGrow() {
  els.message.style.height = "auto";
  els.message.style.height = `${Math.min(els.message.scrollHeight, 180)}px`;
}

els.form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
els.message.addEventListener("input", autoGrow);
els.message.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
els.criteria.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); send(); }
});
els.stop.addEventListener("click", stop);
els.reset.addEventListener("click", reset);

$("suggestions").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  els.message.value = chip.dataset.message;
  els.criteria.value = chip.dataset.criteria;
  autoGrow();
  els.message.focus();
});

window.addEventListener("pagehide", closeSession);

startSession();
refreshFiles();
els.message.focus();
