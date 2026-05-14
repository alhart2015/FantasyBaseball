// /trends — Chart.js line graphs for actual standings + projected ERoto.
//
// Loads /api/trends/series once, builds two charts, and handles tab
// switching (swap dataset.data, then chart.update()), hover-to-highlight
// (dim non-hovered datasets), and click-to-toggle (Chart.js legend
// default).

(function () {
  // 12-color qualitative palette; user team gets its own bold color.
  const PALETTE = [
    "#4e79a7", "#f28e2c", "#76b041", "#bab0ac",
    "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
    "#9c755f", "#3a9da3", "#86bc4f", "#b07aa1",
  ];
  const USER_COLOR = "#e15759";
  // Lower-is-better stats: flip the y-axis so the leader sits on top.
  const INVERSE_STATS = new Set(["ERA", "WHIP"]);

  let payload = null;
  // Populated from payload.counting_stats on fetch — the API is the
  // source of truth for which categories arrive as "distance from leader".
  let countingStats = new Set();

  function yAxisTitle(tab) {
    if (tab === "roto") return "Roto points";
    if (countingStats.has(tab)) return tab + " - distance from 1st";
    return tab;
  }

  function yAxisReversed(tab) {
    return INVERSE_STATS.has(tab);
  }

  function colorForTeam(name, userTeam, sortedNames) {
    if (name === userTeam) return USER_COLOR;
    const otherNames = sortedNames.filter((n) => n !== userTeam);
    const i = otherNames.indexOf(name);
    return PALETTE[i % PALETTE.length];
  }

  function buildDatasets(seriesTeams, dates, userTeam, tab) {
    const sortedNames = Object.keys(seriesTeams).sort();
    return sortedNames.map((name) => {
      const series = seriesTeams[name];
      const data = tab === "roto" ? series.roto_points : (series.stats[tab] || []);
      const color = colorForTeam(name, userTeam, sortedNames);
      return {
        label: name,
        data: data.slice(),
        borderColor: color,
        backgroundColor: color,
        // _origColor is the immutable source of truth for this team's
        // color. dimOthers / resetAlpha read it to compute live
        // borderColor / backgroundColor without ever mutating it.
        _origColor: color,
        borderWidth: name === userTeam ? 4 : 2,
        pointRadius: 2,
        pointHoverRadius: 5,
        spanGaps: false,
        tension: 0.2,
      };
    });
  }

  function buildChart(canvasId, dates, datasets) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    const chart = new Chart(ctx, {
      type: "line",
      data: { labels: dates, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        // axis: "xy" picks the dataset closest to the cursor by 2D
        // distance. With axis: "x" every dataset ties at any cursor x
        // (they all have a point at every x-index), so Chart.js falls
        // back to dataset order and the first one stays "focused" no
        // matter where you hover.
        interaction: { mode: "nearest", axis: "xy", intersect: false },
        plugins: {
          legend: { position: "right", onHover: undefined },
          tooltip: { mode: "nearest", axis: "xy", intersect: false },
        },
        scales: {
          y: {
            beginAtZero: false,
            reverse: yAxisReversed("roto"),
            title: { display: true, text: yAxisTitle("roto") },
          },
          x: { ticks: { autoSkip: true, maxTicksLimit: 10 } },
        },
        onHover: (evt, activeEls) => {
          if (!activeEls || activeEls.length === 0) {
            resetAlpha(chart);
            return;
          }
          const focused = activeEls[0].datasetIndex;
          dimOthers(chart, focused);
        },
      },
    });
    // `interaction.mode: "nearest"` with `intersect: false` always picks a
    // nearest dataset while the cursor is inside the plot area, so onHover
    // never sees an empty activeEls there. The dim state would persist
    // forever once the cursor leaves the canvas — clear it explicitly on
    // mouseleave.
    canvas.addEventListener("mouseleave", () => resetAlpha(chart));
    return chart;
  }

  function withAlpha(hex, alpha) {
    // hex like "#abcdef" → "rgba(r,g,b,a)"
    const m = /^#?([a-f\d]{6})$/i.exec(hex);
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function dimOthers(chart, focusedIdx) {
    chart.data.datasets.forEach((ds, i) => {
      const baseColor = ds._origColor;
      if (i === focusedIdx) {
        ds.borderColor = baseColor;
        ds.backgroundColor = baseColor;
      } else {
        ds.borderColor = withAlpha(baseColor, 0.15);
        ds.backgroundColor = withAlpha(baseColor, 0.15);
      }
    });
    chart.update("none");
  }

  function resetAlpha(chart) {
    chart.data.datasets.forEach((ds) => {
      ds.borderColor = ds._origColor;
      ds.backgroundColor = ds._origColor;
    });
    chart.update("none");
  }

  function applyTab(chart, target, tab) {
    const series = payload[target];
    chart.data.datasets.forEach((ds) => {
      const team = series.teams[ds.label];
      if (!team) {
        ds.data = [];
        return;
      }
      ds.data = (tab === "roto" ? team.roto_points : team.stats[tab] || []).slice();
    });
    chart.options.scales.y.title.text = yAxisTitle(tab);
    chart.options.scales.y.reverse = yAxisReversed(tab);
    resetAlpha(chart);
    chart.update();
  }

  function wireTabs(navSelector, chart, target) {
    const nav = document.querySelector(navSelector);
    if (!nav) return;
    nav.addEventListener("click", (evt) => {
      const btn = evt.target.closest("button[data-tab]");
      if (!btn) return;
      nav.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyTab(chart, target, btn.dataset.tab);
    });
  }

  function showError(canvasId, msg) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const wrapper = canvas.parentElement;
    if (wrapper) {
      wrapper.innerHTML = '<div style="padding: 24px; color: var(--text-secondary); font-size: 13px;">' + msg + '</div>';
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    fetch("/api/trends/series")
      .then((r) => {
        if (!r.ok) {
          throw new Error("HTTP " + r.status);
        }
        return r.json();
      })
      .then((data) => {
        payload = data;
        countingStats = new Set(data.counting_stats || []);
        const userTeam = data.user_team;

        if (!data.actual || !data.actual.dates || data.actual.dates.length === 0) {
          showError("chart-actual", "No standings history yet. Run a refresh.");
        } else {
          const actualDatasets = buildDatasets(
            data.actual.teams, data.actual.dates, userTeam, "roto"
          );
          const actualChart = buildChart(
            "chart-actual", data.actual.dates, actualDatasets
          );
          wireTabs('.tab-strip[data-target="actual"]', actualChart, "actual");
        }

        if (!data.projected || !data.projected.dates || data.projected.dates.length === 0) {
          showError("chart-projected", "No projected history yet. Run a refresh or run the backfill script.");
        } else {
          const projectedDatasets = buildDatasets(
            data.projected.teams, data.projected.dates, userTeam, "roto"
          );
          const projectedChart = buildChart(
            "chart-projected", data.projected.dates, projectedDatasets
          );
          wireTabs('.tab-strip[data-target="projected"]', projectedChart, "projected");
        }
      })
      .catch((err) => {
        showError("chart-actual", "Failed to load trends: " + err.message);
        showError("chart-projected", "Failed to load trends: " + err.message);
      });
  });
})();
