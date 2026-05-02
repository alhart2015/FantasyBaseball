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

  let payload = null;

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
        borderWidth: name === userTeam ? 4 : 2,
        pointRadius: 2,
        pointHoverRadius: 5,
        spanGaps: false,
        tension: 0.2,
      };
    });
  }

  function buildChart(canvasId, dates, datasets) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    return new Chart(ctx, {
      type: "line",
      data: { labels: dates, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        plugins: {
          legend: { position: "right", onHover: undefined },
          tooltip: { mode: "nearest", intersect: false },
        },
        scales: {
          y: { beginAtZero: false },
          x: { ticks: { autoSkip: true, maxTicksLimit: 10 } },
        },
        onHover: (evt, activeEls, chart) => {
          if (!activeEls || activeEls.length === 0) {
            resetAlpha(chart);
            return;
          }
          const focused = activeEls[0].datasetIndex;
          dimOthers(chart, focused);
        },
      },
    });
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
      if (!ds._origColor) ds._origColor = ds.borderColor;
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
      if (ds._origColor) {
        ds.borderColor = ds._origColor;
        ds.backgroundColor = ds._origColor;
      }
    });
    chart.update("none");
  }

  function applyTab(chart, target, tab) {
    const series = payload[target];
    chart.data.datasets.forEach((ds) => {
      delete ds._origColor;
      const team = series.teams[ds.label];
      if (!team) {
        ds.data = [];
        return;
      }
      ds.data = (tab === "roto" ? team.roto_points : team.stats[tab] || []).slice();
    });
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

  document.addEventListener("DOMContentLoaded", () => {
    fetch("/api/trends/series")
      .then((r) => r.json())
      .then((data) => {
        payload = data;
        const userTeam = data.user_team;

        const actualDatasets = buildDatasets(
          data.actual.teams, data.actual.dates, userTeam, "roto"
        );
        const actualChart = buildChart(
          "chart-actual", data.actual.dates, actualDatasets
        );

        const projectedDatasets = buildDatasets(
          data.projected.teams, data.projected.dates, userTeam, "roto"
        );
        const projectedChart = buildChart(
          "chart-projected", data.projected.dates, projectedDatasets
        );

        wireTabs('.tab-strip[data-target="actual"]', actualChart, "actual");
        wireTabs('.tab-strip[data-target="projected"]', projectedChart, "projected");
      });
  });
})();
