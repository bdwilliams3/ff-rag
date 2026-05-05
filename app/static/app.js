const form = document.querySelector("#chatForm");
const messages = document.querySelector("#messages");
const statusEl = document.querySelector("#status");

function value(id) {
  return document.querySelector(id).value.trim();
}

function addMessage(role, content) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = content;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderMarkdownish(text) {
  return escapeHtml(text).replace(/\n/g, "<br>");
}

function renderTables(tables) {
  return (tables || []).map((table) => {
    const columns = table.columns || [];
    const rows = table.rows || [];
    return `
      <section class="block">
        <h2>${escapeHtml(table.title || "Table")}</h2>
        <div class="tableWrap">
          <table>
            <thead><tr>${columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}</tr></thead>
            <tbody>
              ${rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}
            </tbody>
          </table>
        </div>
      </section>`;
  }).join("");
}

function renderLists(lists) {
  return (lists || []).map((list) => `
    <section class="block">
      <h2>${escapeHtml(list.title || "List")}</h2>
      <ol>${(list.items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
    </section>
  `).join("");
}

function renderCitations(citations) {
  if (!citations || citations.length === 0) return "";
  return `
    <section class="chips">
      ${citations.map((c) => `<span>${escapeHtml(c.data_source || "source")}: ${escapeHtml(c.detail || "")}</span>`).join("")}
    </section>
  `;
}

function renderActions(results) {
  if (!results || results.length === 0) return "";
  return `
    <section class="actions">
      ${results.map((r) => `<p><strong>${escapeHtml(r.status)}</strong> ${escapeHtml(r.message || `${r.type} ${r.recipient_alias || ""}`)}</p>`).join("")}
    </section>
  `;
}

async function loadStatus() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    statusEl.textContent = `Ready · recipients: ${data.recipients.join(", ") || "none"}`;
  } catch {
    statusEl.textContent = "Backend unavailable";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = value("#message");
  if (!message) return;

  addMessage("user", `<p>${renderMarkdownish(message)}</p>`);
  document.querySelector("#message").value = "";
  const pending = addMessage("assistant", `<p>Thinking...</p>`);

  const payload = {
    message,
    top: Number(value("#top") || 14),
    position: value("#position") || null,
    season_min: value("#seasonMin") ? Number(value("#seasonMin")) : null,
    season_max: value("#seasonMax") ? Number(value("#seasonMax")) : null,
    source: value("#source") || null,
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    pending.innerHTML = `
      <div class="answer">${renderMarkdownish(data.answer)}</div>
      ${renderTables(data.tables)}
      ${renderLists(data.lists)}
      ${renderCitations(data.citations)}
      ${renderActions(data.action_results)}
    `;
  } catch (error) {
    pending.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
});

loadStatus();
