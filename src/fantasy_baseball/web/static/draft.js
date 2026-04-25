// Live-draft dashboard: polls state every 500ms, renders picks,
// posts picks on player click, posts undos on undo-click.

const POLL_INTERVAL_MS = 500;
let lastVersion = 0;
let lastOnTheClock = null;
let lastState = null;  // cached so filter/sort UI changes re-render without refetching
let fullBoard = [];  // cached once from /api/board
let recsPrimarySort = "immediate"; // or "vopn"
// Increments on every recs fetch so a slow response from a stale request
// doesn't overwrite a newer one. lastRecsRows caches the most recent
// response so the sort toggle can re-render without re-fetching.
let recsRequestId = 0;
let lastRecsRows = [];

// Available-players panel sort + filter state.
// playerSort=null means "use board order" (var-desc as written by build_draft_board).
let playerSort = null;        // null | "name" | "adp" | "sgp"
let playerSortDir = "desc";   // "asc" | "desc"
let playerNameFilter = "";    // case-insensitive substring of player.name
let playerPosFilter = "";     // canonical position string, "" means no filter

// League meta cached on page load (teams + snake pick_order + user_team).
let leagueMeta = { teams: [], user_team: null, pick_order: [] };

async function fetchBoard() {
  const r = await fetch("/api/board");
  if (!r.ok) return [];
  return r.json();
}

async function fetchMeta() {
  const r = await fetch("/api/meta");
  if (!r.ok) return { teams: [], user_team: null, pick_order: [] };
  return r.json();
}

function populateTeamPicker(meta) {
  const select = document.getElementById("team-picker");
  select.innerHTML = meta.teams.map((t) => `<option value="${t}">${t}</option>`).join("");
  if (meta.user_team) select.value = meta.user_team;
}

async function fetchState(since = null) {
  const url = since == null ? "/api/state" : `/api/state?since=${since}`;
  const r = await fetch(url);
  if (!r.ok) return null;
  return r.json();
}

function renderState(state) {
  lastState = state;
  document.getElementById("round").textContent = state.on_the_clock ? currentRound(state) : "done";
  document.getElementById("pick").textContent = (state.picks?.length ?? 0) + 1;
  document.getElementById("otc-btn").textContent = state.on_the_clock ?? "—";
  document.getElementById("picks-to-next").textContent = picksUntilNext(state);
  renderAvailablePlayers(state);
  renderRecentPicks(state);
  // /api/recs is the slow endpoint (~200ms even capped). Only refetch when
  // the team on the clock changes — otherwise the cached rows still apply.
  if (state.on_the_clock && state.on_the_clock !== lastOnTheClock) {
    lastOnTheClock = state.on_the_clock;
    loadAndRenderRecs(state.on_the_clock);
  }
  // Roster/standings panels need to refresh on every pick — otherwise
  // the user sees stale data until they manually flip tabs.
  refreshInspectorPanel();
}

function currentRound(state) {
  const numTeams = leagueMeta.teams?.length || 10;
  return Math.floor((state.picks?.length ?? 0) / numTeams) + 1;
}

function picksUntilNext(state) {
  // Number of picks between the current overall pick and the user's
  // next turn in the snake order. Returns "—" if we don't have meta yet
  // or the user team isn't in the order.
  if (!leagueMeta.user_team || !leagueMeta.pick_order?.length) return "—";
  const picksSoFar = state.picks?.length ?? 0;
  const userIdx = leagueMeta.pick_order.findIndex(
    (t, i) => i >= picksSoFar && t === leagueMeta.user_team
  );
  if (userIdx === -1) return "—";
  return userIdx - picksSoFar;
}

// 999 is the projection-CSV sentinel for "no ADP data" (e.g. NPB
// imports). Treat any value at or above the sentinel as missing so
// boards written before the sentinel-normalization fix still render
// "—" instead of "999.0".
function fmtAdp(v) { return v == null || v >= 999 ? "—" : v.toFixed(1); }
function fmtSgp(v) { return v == null ? "—" : v.toFixed(2); }

// Match a player's positions against the dropdown's canonical filter.
// "P" expands to {P, SP, RP} so "any pitcher" works; everything else is
// a case-insensitive literal match. The board emits Yahoo casing
// ("Util") but the filter values are canonical ("UTIL").
function playerMatchesPosition(player, posFilter) {
  if (!posFilter) return true;
  const positions = (player.positions || []).map((p) => p.toUpperCase());
  if (posFilter === "P") {
    return positions.some((p) => p === "P" || p === "SP" || p === "RP");
  }
  return positions.includes(posFilter);
}

// Sort key plucker. ADP-missing rows sort last regardless of direction
// (they're effectively "unranked"); same for SGP-missing.
function playerSortValue(player, key) {
  if (key === "name") return (player.name || "").toLowerCase();
  // Treat 999+ as missing — same sentinel fmtAdp handles for display.
  if (key === "adp") return player.adp == null || player.adp >= 999 ? Infinity : player.adp;
  if (key === "sgp") return player.total_sgp == null ? -Infinity : player.total_sgp;
  return 0;
}

function applyPlayerSort(players) {
  if (playerSort == null) return players;  // board order (var-desc)
  // Numeric default for ADP is asc (lower = earlier pick); SGP and most
  // others default to desc (higher = better). The dir state overrides.
  const ascending = playerSortDir === "asc";
  return [...players].sort((a, b) => {
    const va = playerSortValue(a, playerSort);
    const vb = playerSortValue(b, playerSort);
    if (va < vb) return ascending ? -1 : 1;
    if (va > vb) return ascending ? 1 : -1;
    return 0;
  });
}

function renderAvailablePlayers(state) {
  const drafted = new Set([
    ...(state.keepers ?? []).map((p) => p.player_id),
    ...(state.picks ?? []).map((p) => p.player_id),
  ]);
  const nameQuery = playerNameFilter.toLowerCase();
  let available = fullBoard.filter((p) =>
    !drafted.has(p.player_id) &&
    playerMatchesPosition(p, playerPosFilter) &&
    (!nameQuery || (p.name || "").toLowerCase().includes(nameQuery))
  );
  available = applyPlayerSort(available);
  const ul = document.getElementById("player-list");
  ul.innerHTML = available.slice(0, 200).map((p) => `
    <li data-pid="${p.player_id}" data-pname="${p.name}" data-pos="${p.best_position || p.positions?.[0] || ''}">
      <span class="name">${p.name}</span>
      <span class="pos">${(p.positions || []).join("/")}</span>
      <span class="adp">${fmtAdp(p.adp)}</span>
      <span class="sgp">${fmtSgp(p.total_sgp)}</span>
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
  updateSortIndicators();
}

function updateSortIndicators() {
  document.querySelectorAll(".player-list-header [data-sort]").forEach((btn) => {
    const isActive = playerSort === btn.dataset.sort;
    btn.classList.toggle("sort-active", isActive);
    btn.classList.toggle("sort-asc", isActive && playerSortDir === "asc");
    btn.classList.toggle("sort-desc", isActive && playerSortDir === "desc");
  });
}

// Re-render the available-players panel from cached state. Used by the
// filter/sort UI listeners — saves a poll-cycle round-trip when the
// state hasn't changed.
function rerenderAvailablePlayers() {
  if (lastState) renderAvailablePlayers(lastState);
}

function renderRecentPicks(state) {
  const recent = (state.picks ?? []).slice(-6);
  document.getElementById("recent-picks").innerHTML = recent.map((p) => `
    <li>${p.team}: ${p.player_name}</li>
  `).join("");
  document.getElementById("undo-btn").disabled = (state.picks?.length ?? 0) === 0;
}

async function fetchRecs(team) {
  const r = await fetch(`/api/recs?team=${encodeURIComponent(team)}`);
  if (!r.ok) return null;  // null distinguishes failure from "0 recs"
  return r.json();
}

function setRecsStatus(text, loading) {
  const status = document.getElementById("recs-status");
  if (status) status.textContent = text;
  document.getElementById("rec-list").classList.toggle("loading", !!loading);
}

// Fetch + render with a stale-response guard. If a newer fetch starts
// before this one returns, the older response is discarded — prevents
// the panel from flashing stale recs on top of fresh ones.
async function loadAndRenderRecs(team) {
  const myReq = ++recsRequestId;
  setRecsStatus(`loading for ${team}…`, true);
  const rows = await fetchRecs(team);
  if (myReq !== recsRequestId) return;  // superseded
  if (rows == null) {
    lastRecsRows = [];
    document.getElementById("rec-list").innerHTML = "";
    setRecsStatus(`no data for ${team} (board not loaded?)`, false);
    return;
  }
  lastRecsRows = rows;
  renderRecs(rows);
  setRecsStatus(rows.length ? `for ${team}` : `no recs for ${team}`, false);
}

function renderRecs(rows) {
  const sortKey = recsPrimarySort === "immediate" ? "immediate_delta" : "value_of_picking_now";
  rows = [...rows].sort((a, b) => b[sortKey] - a[sortKey]);
  const ol = document.getElementById("rec-list");
  ol.innerHTML = rows.map((r, i) => `
    <li data-pid="${r.player_id}" data-pname="${r.name}" data-pos="${r.positions[0] || ''}">
      <div class="row">
        <span class="rank">${i + 1}.</span>
        <span class="name">${r.name}</span>
        <span class="pos">${r.positions.join("/")}</span>
        <span class="delta ${r.immediate_delta >= 0 ? "positive" : "negative"}">
          ${r.immediate_delta.toFixed(2)}
        </span>
        <span class="vopn">${r.value_of_picking_now.toFixed(2)}</span>
        <button class="detail-toggle" aria-label="expand">▾</button>
      </div>
      <div class="detail">
        <table>
          <thead><tr><th>Cat</th><th>Δ</th></tr></thead>
          <tbody>
            ${Object.entries(r.per_category).map(([cat, delta]) => `
              <tr><td>${cat}</td><td class="${delta >= 0 ? 'positive' : 'negative'}">${delta.toFixed(2)}</td></tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </li>
  `).join("");
  ol.onclick = (e) => {
    const toggle = e.target.closest(".detail-toggle");
    if (toggle) {
      toggle.closest("li").classList.toggle("expanded");
      return;
    }
    const li = e.target.closest("li");
    if (li) {
      recordPick({
        player_id: li.dataset.pid,
        player_name: li.dataset.pname,
        position: li.dataset.pos,
        team: document.getElementById("otc-btn").textContent,
      });
    }
  };
}

async function fetchRoster(team) {
  const r = await fetch(`/api/roster?team=${encodeURIComponent(team)}`);
  if (!r.ok) return [];
  return r.json();
}

function renderRoster(rows) {
  document.getElementById("roster-panel").innerHTML = `
    <ul class="roster-list">
      ${rows.map((row) => `
        <li class="${row.replacement ? 'replacement-slot' : ''}">
          <span class="slot">${row.slot}</span>
          <span class="name">${row.name}</span>
        </li>
      `).join("")}
    </ul>
  `;
}

async function fetchStandings() {
  const r = await fetch("/api/standings");
  if (!r.ok) return [];
  return r.json();
}

function renderStandings(rows) {
  document.getElementById("standings-panel").innerHTML = `
    <table class="standings">
      <thead><tr><th>Team</th><th>ERoto</th><th>±</th></tr></thead>
      <tbody>
        ${rows.map((r) => `
          <tr>
            <td>${r.team}</td>
            <td>
              ${r.total.toFixed(1)}
              <span class="uncertainty-bar" style="width:${Math.max(4, r.sd * 2)}px"></span>
            </td>
            <td>±${r.sd.toFixed(1)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
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

async function refreshInspectorPanel() {
  const team = document.getElementById("team-picker").value || document.getElementById("otc-btn").textContent;
  const activeTab = document.querySelector(".team-inspector .tabs button.active")?.dataset.tab;
  if (activeTab === "standings") {
    renderStandings(await fetchStandings());
  } else {
    renderRoster(await fetchRoster(team));
  }
}

(async () => {
  fullBoard = await fetchBoard();
  leagueMeta = await fetchMeta();
  populateTeamPicker(leagueMeta);
  const initial = await fetchState();
  if (initial) renderState(initial);
  // Initial Roster-tab fill (the tab is .active by default in dashboard.html).
  refreshInspectorPanel();
  document.getElementById("undo-btn").onclick = undo;
  document.getElementById("new-draft-btn").onclick = newDraft;
  document.getElementById("team-picker").addEventListener("change", refreshInspectorPanel);

  document.querySelectorAll(".sort-toggle button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sort-toggle button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      recsPrimarySort = btn.dataset.sort;
      // Sort is client-side — re-render the cached rows instead of refetching.
      if (lastRecsRows.length) renderRecs(lastRecsRows);
    });
  });

  // Available-players name filter: re-render on every keystroke.
  document.getElementById("player-filter").addEventListener("input", (e) => {
    playerNameFilter = e.target.value;
    rerenderAvailablePlayers();
  });

  // Position filter dropdown.
  document.getElementById("position-filter").addEventListener("change", (e) => {
    playerPosFilter = e.target.value;
    rerenderAvailablePlayers();
  });

  // Column-header sort: clicking the active column flips direction;
  // clicking a different column starts in that column's natural direction
  // (asc for ADP / name, desc for SGP).
  document.querySelectorAll(".player-list-header [data-sort]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.sort;
      if (playerSort === key) {
        playerSortDir = playerSortDir === "asc" ? "desc" : "asc";
      } else {
        playerSort = key;
        playerSortDir = key === "sgp" ? "desc" : "asc";
      }
      rerenderAvailablePlayers();
    });
  });

  document.querySelectorAll(".team-inspector .tabs button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll(".team-inspector .tabs button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.getElementById("roster-panel").classList.toggle("hidden", tab !== "roster");
      document.getElementById("standings-panel").classList.toggle("hidden", tab !== "standings");
      const team = document.getElementById("team-picker").value || document.getElementById("otc-btn").textContent;
      if (tab === "roster") {
        renderRoster(await fetchRoster(team));
      } else {
        renderStandings(await fetchStandings());
      }
    });
  });

  setTimeout(poll, POLL_INTERVAL_MS);
})();
