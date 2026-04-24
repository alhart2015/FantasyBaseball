// Live-draft dashboard: polls state every 500ms, renders picks,
// posts picks on player click, posts undos on undo-click.

const POLL_INTERVAL_MS = 500;
let lastVersion = 0;
let fullBoard = [];  // cached once from /api/board

async function fetchBoard() {
  const r = await fetch("/api/board");
  if (!r.ok) return [];
  return r.json();
}

async function fetchState(since = null) {
  const url = since == null ? "/api/state" : `/api/state?since=${since}`;
  const r = await fetch(url);
  if (!r.ok) return null;
  return r.json();
}

function renderState(state) {
  document.getElementById("round").textContent = state.on_the_clock ? currentRound(state) : "done";
  document.getElementById("pick").textContent = (state.picks?.length ?? 0) + 1;
  document.getElementById("otc-btn").textContent = state.on_the_clock ?? "—";
  document.getElementById("picks-to-next").textContent = picksUntilNext(state);
  renderAvailablePlayers(state);
  renderRecentPicks(state);
}

function currentRound(state) {
  const numTeams = 10;  // TODO wire from /api/state meta if this ever varies
  return Math.floor((state.picks?.length ?? 0) / numTeams) + 1;
}

function picksUntilNext(state) {
  // Placeholder — Phase 5 wires real "your next pick" calculation.
  return "—";
}

function renderAvailablePlayers(state) {
  const drafted = new Set([
    ...(state.keepers ?? []).map((p) => p.player_id),
    ...(state.picks ?? []).map((p) => p.player_id),
  ]);
  const available = fullBoard.filter((p) => !drafted.has(p.player_id));
  const ul = document.getElementById("player-list");
  ul.innerHTML = available.slice(0, 200).map((p) => `
    <li data-pid="${p.player_id}" data-pname="${p.name}" data-pos="${p.best_position || p.positions?.[0] || ''}">
      ${p.name} <span class="pos">${(p.positions || []).join("/")}</span>
    </li>
  `).join("");
  ul.onclick = (e) => {
    const li = e.target.closest("li");
    if (!li) return;
    recordPick({
      player_id: li.dataset.pid,
      player_name: li.dataset.pname,
      position: li.dataset.pos,
      team: document.getElementById("otc-btn").textContent,
    });
  };
}

function renderRecentPicks(state) {
  const recent = (state.picks ?? []).slice(-6);
  document.getElementById("recent-picks").innerHTML = recent.map((p) => `
    <li>${p.team}: ${p.player_name}</li>
  `).join("");
  document.getElementById("undo-btn").disabled = (state.picks?.length ?? 0) === 0;
}

async function recordPick(payload) {
  const r = await fetch("/api/pick", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const err = await r.json();
    alert(`Pick rejected: ${err.error}`);
    return;
  }
  const state = await r.json();
  lastVersion = state.version ?? lastVersion;
  renderState(state);
}

async function undo() {
  const r = await fetch("/api/undo", { method: "POST" });
  if (!r.ok) return;
  const state = await r.json();
  renderState(state);
}

async function newDraft() {
  const r = await fetch("/api/new-draft", { method: "POST" });
  if (!r.ok) {
    const err = await r.json();
    alert(`New draft failed: ${err.error}`);
    return;
  }
  renderState(await r.json());
}

async function reset() {
  const input = prompt("Type RESET to confirm");
  if (input !== "RESET") return;
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: "RESET" }),
  });
  window.location.reload();
}

async function poll() {
  const state = await fetchState(lastVersion || null);
  if (state) {
    if (state.full_state || state.version !== lastVersion) {
      lastVersion = state.version ?? lastVersion;
      renderState(state);
    }
  }
  setTimeout(poll, POLL_INTERVAL_MS);
}

(async () => {
  fullBoard = await fetchBoard();
  const initial = await fetchState();
  if (initial) renderState(initial);
  document.getElementById("undo-btn").onclick = undo;
  document.getElementById("new-draft-btn").onclick = newDraft;
  document.getElementById("reset-btn").onclick = reset;
  setTimeout(poll, POLL_INTERVAL_MS);
})();
