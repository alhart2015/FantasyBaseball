/* Distributions: ridgeline of per-team Monte Carlo outcome distributions for the
 * standings "Distributions" view. Reads the JSON embedded by standings.html
 * (#distributions-data) and draws into #distributions-canvas with the raw 2D
 * canvas API -- one density row per team on a shared x-axis, user row highlighted,
 * a central-tendency tick per row. Re-renders on metric/mode change and when the
 * tab is shown (the canvas has zero size while its view is display:none).
 *
 * Formatted data shape (from format_distributions_for_display):
 *   { overall: {x:[...], rows:[{team,is_user,y:[...],median}]},
 *     category_totals: {CAT: {x:[...], rows:[{team,is_user,y:[...],median}]}},
 *     category_points: {CAT: {x:[...], rows:[{team,is_user,p:[...],mean}]}} }
 */
(function () {
  "use strict";

  var USER_COLOR = "#e15759";
  var OTHER_COLOR = "#4e79a7";
  // Decimal precision for the rate categories' tick labels (not a direction set).
  var RATE_PREC = { AVG: 3, ERA: 2, WHIP: 2 };

  // Ridgeline layout: fixed vertical spacing per team row so the chart height
  // grows with the league size (no clipping), plus headroom for the top row's
  // peak and the x-axis label. Curves overlap upward by OVERLAP rows.
  var ROW_H = 48;
  var PAD_TOP = 44;
  var PAD_BOTTOM = 40;
  var OVERLAP = 1.6;
  // Left label gutter: fit the longest team name (+LABEL_PAD), clamped so a very
  // long name can't eat the plot and a short one still leaves a readable margin.
  var GUTTER_MIN = 90;
  var GUTTER_MAX = 220;
  var LABEL_PAD = 16;

  var state = { metric: "overall", mode: "totals" };
  var payload = null;

  function loadPayload() {
    var node = document.getElementById("distributions-data");
    if (!node) return null;
    try { return JSON.parse(node.textContent); } catch (e) { return null; }
  }

  function fmtTick(value) {
    var cat = state.metric;
    if (cat !== "overall" && state.mode === "totals" && RATE_PREC[cat] != null) {
      return value.toFixed(RATE_PREC[cat]);
    }
    return String(Math.round(value * 10) / 10);
  }

  // Pixel y of a curve sample within a row (baseline minus normalized height).
  function curveY(curve, idx, cMax, amp, baseY) {
    return baseY - (curve[idx] / cMax) * amp;
  }

  // Resolve the active metric into a uniform shape:
  // {x:[...], rows:[{team,is_user,curve:[...],center:float}], discrete:bool, label}
  function currentMetric() {
    if (!payload) return null;
    if (state.metric === "overall") {
      return adapt(payload.overall, "y", "median", false, "Total roto points");
    }
    var cat = state.metric;
    if (state.mode === "points") {
      var cp = (payload.category_points || {})[cat];
      return adapt(cp, "p", "mean", true, cat + " roto points");
    }
    var ct = (payload.category_totals || {})[cat];
    return adapt(ct, "y", "median", false, cat + " total");
  }

  function adapt(metric, curveKey, centerKey, discrete, label) {
    if (!metric || !metric.rows || !metric.rows.length) return null;
    return {
      x: metric.x,
      discrete: discrete,
      label: label,
      rows: metric.rows.map(function (r) {
        return { team: r.team, is_user: r.is_user, curve: r[curveKey], center: r[centerKey] };
      })
    };
  }

  function showEmpty(canvas, empty, isEmpty) {
    if (canvas) canvas.style.display = isEmpty ? "none" : "";
    if (empty) empty.style.display = isEmpty ? "" : "none";
  }

  function render() {
    if (payload == null) payload = loadPayload();
    var canvas = document.getElementById("distributions-canvas");
    var empty = document.getElementById("dist-empty");
    if (!canvas) return;

    var data = currentMetric();
    if (!data) { showEmpty(canvas, empty, true); return; }
    showEmpty(canvas, empty, false);

    // Size the backing store from the WRAPPER box (responsive width x fixed
    // height) times devicePixelRatio for crisp lines. Do NOT read the canvas's
    // own clientWidth/Height: the canvas is sized to 100% of the wrapper via CSS
    // and has no intrinsic size, so reading its layout size and writing it back
    // into the backing store compounds on every render -- that is why the chart
    // started tiny and grew on each tab click. The wrapper is the stable box.
    var dpr = window.devicePixelRatio || 1;
    var wrap = canvas.parentNode;
    var cssW = wrap.clientWidth || 800;
    // Height grows with the team count (fixed per-row spacing) so no row or the
    // x-axis label is clipped, whatever the league size. The wrapper height is
    // driven from here; the canvas fills it via CSS.
    var cssH = data.rows.length * ROW_H + PAD_TOP + PAD_BOTTOM;
    wrap.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    drawRidgeline(ctx, cssW, cssH, data);
  }

  function drawRidgeline(ctx, W, H, data) {
    var rows = data.rows;
    var n = rows.length;

    // Left gutter fits the longest team name (measured) so labels never clip;
    // clamped so a very long name can't eat the whole plot.
    ctx.font = "bold 12px system-ui, sans-serif";
    var maxLabelW = 0;
    for (var li = 0; li < n; li++) {
      var lw = ctx.measureText(rows[li].team).width;
      if (lw > maxLabelW) maxLabelW = lw;
    }
    var padL = Math.min(GUTTER_MAX, Math.max(GUTTER_MIN, Math.ceil(maxLabelW) + LABEL_PAD));
    var padR = 24, padT = PAD_TOP, padB = PAD_BOTTOM;
    var plotW = W - padL - padR;
    var plotH = H - padT - padB;

    var xs = data.x;
    var xMin = xs[0], xMax = xs[xs.length - 1];
    var xSpan = xMax - xMin || 1;
    function sx(v) { return padL + ((v - xMin) / xSpan) * plotW; }

    // Per-row vertical band (== ROW_H, since H was sized from it); curves overlap
    // upward by OVERLAP bands for the classic ridgeline look. Peak height is
    // normalized per row (each row's own max), so a tight row reads as
    // tall-and-narrow, a wide row as low-and-broad.
    var band = plotH / n;

    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";

    // x-axis label.
    ctx.fillStyle = "#888";
    ctx.textAlign = "center";
    ctx.fillText(data.label, padL + plotW / 2, H - 12);

    // Draw back-to-front (bottom rows first) so upper rows overlap correctly.
    for (var i = n - 1; i >= 0; i--) {
      var row = rows[i];
      var baseY = padT + (i + 1) * band;
      var curve = row.curve;
      var cMax = 0;
      for (var k = 0; k < curve.length; k++) if (curve[k] > cMax) cMax = curve[k];
      if (cMax <= 0) cMax = 1;
      var amp = band * OVERLAP;

      var stroke = row.is_user ? USER_COLOR : OTHER_COLOR;
      var fill = row.is_user ? "rgba(225,87,89,0.35)" : "rgba(78,121,167,0.22)";

      if (data.discrete) {
        // Stems at each support value.
        ctx.strokeStyle = stroke;
        ctx.lineWidth = row.is_user ? 2.5 : 1.5;
        for (var s = 0; s < xs.length; s++) {
          if (curve[s] <= 0) continue;
          var px = sx(xs[s]);
          ctx.beginPath();
          ctx.moveTo(px, baseY);
          ctx.lineTo(px, curveY(curve, s, cMax, amp, baseY));
          ctx.stroke();
        }
      } else {
        // Filled density path.
        ctx.beginPath();
        ctx.moveTo(sx(xs[0]), baseY);
        for (var j = 0; j < xs.length; j++) ctx.lineTo(sx(xs[j]), curveY(curve, j, cMax, amp, baseY));
        ctx.lineTo(sx(xs[xs.length - 1]), baseY);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = row.is_user ? 2.5 : 1.5;
        ctx.stroke();
      }

      // Central-tendency tick (median for continuous, mean for discrete).
      var tx = sx(row.center);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(tx, baseY);
      ctx.lineTo(tx, baseY - amp);
      ctx.stroke();
      ctx.setLineDash([]);

      // Row label (team name, bold for user) + center value.
      ctx.fillStyle = row.is_user ? USER_COLOR : "#ccc";
      ctx.font = (row.is_user ? "bold " : "") + "12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(row.team, padL - 8, baseY - band * 0.35);
      ctx.fillStyle = "#888";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(fmtTick(row.center), padL - 8, baseY - band * 0.35 + 14);
    }
  }

  window.renderDistributions = render;

  function setActivePill(groupSelector, stateKey, dataAttr, el) {
    // Toggles are `.tab-strip` <button> elements (matching the Trends view);
    // select plain buttons rather than the old `.pill` class.
    document.querySelectorAll(groupSelector + " button").forEach(function (p) {
      p.classList.remove("active");
    });
    el.classList.add("active");
    state[stateKey] = el.dataset[dataAttr];
  }

  window.distSetMetric = function (el) {
    setActivePill("#dist-metric-toggle", "metric", "distmetric", el);
    // Totals|Points only applies to a specific category, not Overall.
    var modeToggle = document.getElementById("dist-mode-toggle");
    if (modeToggle) modeToggle.style.display = state.metric === "overall" ? "none" : "";
    render();
  };

  window.distSetMode = function (el) {
    setActivePill("#dist-mode-toggle", "mode", "distmode", el);
    render();
  };

  // Re-render on resize so the canvas stays crisp and correctly sized.
  window.addEventListener("resize", function () {
    var view = document.getElementById("view-distributions");
    if (view && view.style.display !== "none") render();
  });
})();
