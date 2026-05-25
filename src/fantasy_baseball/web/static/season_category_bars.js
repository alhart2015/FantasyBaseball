/* Category Bars: ranked dot-plot of all teams for one roto category, with
 * +/-1 SD error bars. Reads the JSON embedded by standings.html, renders into
 * #category-bars-canvas, and re-renders on category/projection change.
 * Data shape: { preseason: {CAT: {rows: [{team, value, sd, is_user}, ...],
 *   odds: {first_pct, top3_pct, wins, opponents} | null}}, current: {...} }
 */
(function () {
  "use strict";

  var RATE_CATS = { AVG: 3, ERA: 2, WHIP: 2 };
  var INVERSE_CATS = { ERA: true, WHIP: true };
  var USER_COLOR = "#e15759";
  var OTHER_COLOR = "#4e79a7";

  var state = { projection: "current", category: "R" };
  var chart = null;
  var payload = null;

  // Inline plugin: draw full-height dashed vertical lines at the user team's
  // lower/upper bound (value -/+ sd) so every other team's dot and whiskers
  // read directly against the user's band. Bounds are precomputed per render
  // and read off chart.options so they track the active category/projection.
  var userBoundsPlugin = {
    id: "userBounds",
    afterDatasetsDraw: function (chart) {
      var ub = chart.options.plugins.userBounds;
      if (!ub) return;
      var xScale = chart.scales.x;
      var area = chart.chartArea;
      var c = chart.ctx;
      c.save();
      c.setLineDash([5, 4]);
      c.lineWidth = 1.5;
      c.strokeStyle = ub.color;
      [ub.lo, ub.hi].forEach(function (v) {
        var px = xScale.getPixelForValue(v);
        c.beginPath();
        c.moveTo(px, area.top);
        c.lineTo(px, area.bottom);
        c.stroke();
      });
      c.restore();
    }
  };

  function loadPayload() {
    var node = document.getElementById("category-bars-data");
    if (!node) return null;
    try {
      return JSON.parse(node.textContent);
    } catch (e) {
      return null;
    }
  }

  function fmt(value, cat) {
    if (RATE_CATS[cat] != null) return value.toFixed(RATE_CATS[cat]);
    return String(Math.round(value));
  }

  function entryFor() {
    if (!payload) return null;
    var flavor = payload[state.projection];
    if (!flavor) return null;
    return flavor[state.category] || null;
  }

  function rowsFor() {
    var entry = entryFor();
    return entry && entry.rows ? entry.rows : [];
  }

  function oddsFor() {
    var entry = entryFor();
    return entry ? entry.odds : null;
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function updateOdds() {
    var box = document.getElementById("catbars-odds");
    if (!box) return;
    var odds = oddsFor();
    if (!odds) {
      box.style.display = "none";
      return;
    }
    box.style.display = "";
    setText("catbars-first", odds.first_pct + "%");
    setText("catbars-top3", odds.top3_pct + "%");
    setText("catbars-wins", odds.wins + "/" + odds.opponents);
  }

  function render() {
    if (payload == null) payload = loadPayload();
    updateOdds();
    var rows = rowsFor();
    var canvas = document.getElementById("category-bars-canvas");
    var empty = document.getElementById("catbars-empty");
    if (!canvas) return;

    if (!rows.length) {
      if (chart) { chart.destroy(); chart = null; }
      canvas.style.display = "none";
      if (empty) empty.style.display = "";
      return;
    }
    canvas.style.display = "";
    if (empty) empty.style.display = "none";

    // rows arrive sorted best-on-top; the category y-axis lists top->bottom in
    // the order of `labels`, so use the rows as-is.
    var labels = rows.map(function (r) { return r.team; });
    var points = rows.map(function (r) {
      return { x: r.value, y: r.team, xMin: r.value - r.sd, xMax: r.value + r.sd };
    });
    var colors = rows.map(function (r) { return r.is_user ? USER_COLOR : OTHER_COLOR; });
    var cat = state.category;

    var hint = document.getElementById("catbars-hint");
    if (hint) hint.style.display = INVERSE_CATS[cat] ? "" : "none";

    var userRow = null;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].is_user) {
        userRow = rows[i];
        break;
      }
    }
    var userBounds =
      userRow && userRow.sd > 0
        ? { lo: userRow.value - userRow.sd, hi: userRow.value + userRow.sd, color: USER_COLOR }
        : null;

    var config = {
      type: "scatterWithErrorBars",
      plugins: [userBoundsPlugin],
      data: {
        labels: labels,
        datasets: [{
          label: cat,
          data: points,
          backgroundColor: colors,
          borderColor: colors,
          pointRadius: 6,
          pointHoverRadius: 8,
          errorBarColor: "#888",
          errorBarWhiskerColor: "#888",
          errorBarLineWidth: 1.5,
          errorBarWhiskerSize: 8
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          userBounds: userBounds,
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var r = rows[ctx.dataIndex];
                return r.team + ": " + fmt(r.value, cat) +
                  " \u00b1 " + fmt(r.sd, cat);
              }
            }
          }
        },
        scales: {
          x: { title: { display: true, text: cat } },
          y: { type: "category", labels: labels }
        }
      }
    };

    if (chart) chart.destroy();
    chart = new Chart(canvas.getContext("2d"), config);
  }

  // Exposed so toggleTopView can (re)render when the tab is shown (the canvas
  // has zero size while its view is display:none).
  window.renderCategoryBars = render;

  // Mark the clicked pill active within its group, update the matching state
  // field, and re-render.
  function setActivePill(groupSelector, stateKey, dataAttr, el) {
    document.querySelectorAll(groupSelector + " .pill").forEach(function (p) {
      p.classList.remove("active");
    });
    el.classList.add("active");
    state[stateKey] = el.dataset[dataAttr];
    render();
  }

  window.catBarsSetProjection = function (el) {
    setActivePill("#catbars-proj-toggle", "projection", "cbproj", el);
  };

  window.catBarsSetCategory = function (el) {
    setActivePill("#catbars-cat-toggle", "category", "cbcat", el);
  };
})();
