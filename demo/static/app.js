let sessionId = null;
let pollTimer = null;
let activeTab = "facts";

const els = {
  settingsToggle: document.getElementById("settingsToggle"),
  settingsDrawer: document.getElementById("settingsDrawer"),
  settingsOverlay: document.getElementById("settingsOverlay"),
  settingsClose: document.getElementById("settingsClose"),
  applySettings: document.getElementById("applySettings"),
  settingsStatus: document.getElementById("settingsStatus"),
  generalApiKey: document.getElementById("generalApiKey"),
  generalApiBase: document.getElementById("generalApiBase"),
  generalModel: document.getElementById("generalModel"),
  factApiKey: document.getElementById("factApiKey"),
  factApiBase: document.getElementById("factApiBase"),
  factModel: document.getElementById("factModel"),
  messages: document.getElementById("messages"),
  chatForm: document.getElementById("chatForm"),
  messageInput: document.getElementById("messageInput"),
  sendButton: document.getElementById("sendButton"),
  sessionInfo: document.getElementById("sessionInfo"),
  memoryStatus: document.getElementById("memoryStatus"),
  pendingProfiles: document.getElementById("pendingProfiles"),
  facts: document.getElementById("facts"),
  events: document.getElementById("events"),
  profiles: document.getElementById("profiles"),
  flushProfiles: document.getElementById("flushProfiles"),
};

function settingsPayload() {
  return {
    general_api_key: els.generalApiKey.value.trim(),
    general_api_base: els.generalApiBase.value.trim(),
    general_model: els.generalModel.value.trim(),
    fact_api_key: els.factApiKey.value.trim(),
    fact_api_base: els.factApiBase.value.trim(),
    fact_model: els.factModel.value.trim(),
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_err) {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = payload.detail || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

async function loadDefaults() {
  const payload = await api("/api/defaults");
  const settings = payload.settings || {};
  els.generalApiKey.value = settings.general_api_key || "";
  els.generalApiBase.value = settings.general_api_base || "";
  els.generalModel.value = settings.general_model || "";
  els.factApiKey.value = settings.fact_api_key || "";
  els.factApiBase.value = settings.fact_api_base || "";
  els.factModel.value = settings.fact_model || "";
  setText(els.pendingProfiles, `0 / ${payload.profile_batch_trigger || 10}`);
  if (!settings.general_api_key || !settings.fact_api_key) {
    openSettings();
    setText(els.settingsStatus, "Enter API settings to start.");
  }
}

async function createSession() {
  els.applySettings.disabled = true;
  setText(els.settingsStatus, "Creating session...");
  try {
    const snapshot = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({ settings: settingsPayload() }),
    });
    sessionId = snapshot.session_id;
    setText(els.settingsStatus, "Session ready.");
    setText(els.sessionInfo, snapshot.conversation_id || sessionId);
    els.messages.innerHTML = "";
    renderSnapshot(snapshot);
    startPolling();
    closeSettings();
  } catch (err) {
    els.settingsStatus.innerHTML = `<span class="error-text">${escapeHtml(err.message)}</span>`;
  } finally {
    els.applySettings.disabled = false;
  }
}

function startPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
  pollTimer = setInterval(refreshMemory, 1200);
}

async function refreshMemory() {
  if (!sessionId) {
    return;
  }
  try {
    const snapshot = await api(`/api/memory/${sessionId}`);
    renderSnapshot(snapshot);
  } catch (err) {
    setText(els.memoryStatus, err.message);
    els.memoryStatus?.classList.add("error-text");
  }
}

async function sendMessage(event) {
  event.preventDefault();
  if (!sessionId) {
    openSettings();
    setText(els.settingsStatus, "Apply settings before chatting.");
    return;
  }
  const message = els.messageInput.value.trim();
  if (!message) {
    return;
  }
  appendMessage("user", message);
  els.messageInput.value = "";
  els.sendButton.disabled = true;
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    });
    appendMessage("assistant", payload.assistant_message || "");
    refreshMemory();
  } catch (err) {
    appendMessage("assistant", `Error: ${err.message}`);
  } finally {
    els.sendButton.disabled = false;
    els.messageInput.focus();
  }
}

async function flushProfiles() {
  if (!sessionId) {
    return;
  }
  els.flushProfiles.disabled = true;
  try {
    const snapshot = await api("/api/flush-profiles", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
    renderSnapshot(snapshot);
  } catch (err) {
    setText(els.memoryStatus, err.message);
    els.memoryStatus?.classList.add("error-text");
  } finally {
    els.flushProfiles.disabled = false;
  }
}

function appendMessage(role, content) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = content;
  els.messages.appendChild(node);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function renderSnapshot(snapshot) {
  setText(els.sessionInfo, snapshot.conversation_id || snapshot.session_id || "No session");
  const status = snapshot.status_detail || snapshot.status || "Unknown";
  setText(els.memoryStatus, status);
  els.memoryStatus?.classList.toggle("error-text", snapshot.status === "error");
  const pending = snapshot.pending_profile_facts || 0;
  const trigger = snapshot.profile_batch_trigger || 10;
  setText(els.pendingProfiles, `${pending} / ${trigger}`);
  renderFacts(snapshot.facts || []);
  renderEvents(snapshot.events || []);
  renderProfiles(snapshot.profiles || []);
}

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function renderFacts(facts) {
  if (!facts.length) {
    els.facts.innerHTML = emptyState("No facts yet.");
    return;
  }
  els.facts.innerHTML = facts.map((fact) => {
    return card(
      fact.fact_id || "F?",
      fact.fact || "",
      [
        ...(fact.people || []).map((item) => `person: ${item}`),
        ...(fact.keywords || []).map((item) => `kw: ${item}`),
        ...(fact.event_ids || []).map((item) => `event: ${item}`),
      ].filter(Boolean),
    );
  }).join("");
}

function renderEvents(events) {
  if (!events.length) {
    els.events.innerHTML = emptyState("No events yet.");
    return;
  }
  els.events.innerHTML = events.map((event) => card(
    event.event_id || "E?",
    event.summary || "",
    [
      ...(event.keywords || []).map((item) => `kw: ${item}`),
    ],
  )).join("");
}

function renderProfiles(profiles) {
  if (!profiles.length) {
    els.profiles.innerHTML = emptyState("No profiles yet.");
    return;
  }
  els.profiles.innerHTML = profiles.map((profile) => {
    const historyCount = Array.isArray(profile.history) ? profile.history.length : 0;
    return card(
      profile.profile_id || "P?",
      profile.content || "",
      [
        profile.person ? `person: ${profile.person}` : "",
        profile.valid_from ? `valid from: ${profile.valid_from}` : "",
        historyCount ? `history: ${historyCount}` : "",
        ...(profile.evidence || []).map((item) => `evidence: ${item}`),
      ].filter(Boolean),
    );
  }).join("");
}

function card(id, body, chips) {
  const chipHtml = chips.map((chip) => `<span class="chip">${escapeHtml(String(chip))}</span>`).join("");
  return `
    <article class="memory-card">
      <header><span class="badge">${escapeHtml(String(id))}</span></header>
      <p>${escapeHtml(String(body || "(empty)"))}</p>
      <div class="meta">${chipHtml}</div>
    </article>
  `;
}

function emptyState(text) {
  return `<div class="empty-state">${escapeHtml(text)}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function openSettings() {
  els.settingsDrawer.classList.add("open");
  els.settingsOverlay.classList.add("open");
}

function closeSettings() {
  els.settingsDrawer.classList.remove("open");
  els.settingsOverlay.classList.remove("open");
}

els.settingsToggle.addEventListener("click", openSettings);
els.settingsClose.addEventListener("click", closeSettings);
els.settingsOverlay.addEventListener("click", closeSettings);
els.applySettings.addEventListener("click", createSession);
els.chatForm.addEventListener("submit", sendMessage);
els.flushProfiles.addEventListener("click", flushProfiles);

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    activeTab = button.dataset.tab;
    document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item.dataset.tab === activeTab));
    document.querySelectorAll(".memory-list").forEach((item) => item.classList.toggle("active", item.id === activeTab));
  });
});

els.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.chatForm.requestSubmit();
  }
});

loadDefaults().catch((err) => {
  els.settingsStatus.innerHTML = `<span class="error-text">${escapeHtml(err.message)}</span>`;
});
