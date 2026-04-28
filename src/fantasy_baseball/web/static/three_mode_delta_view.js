// Renders a 3-mode (Roto Points / eROTO / Stat Totals) before/after/delta
// table for a single team. Used by the trade builder.
//
// payload shape:
//   {
//     roto:        {delta_total: number, categories: {CAT: {before, after, delta}}},
//     ev_roto:     {delta_total: number, categories: {CAT: {before, after, delta}}},
//     stat_totals: {delta_total: number, categories: {CAT: {before, after, delta}}},
//   }
//
// opts:
//   initialMode: "roto" | "ev_roto" | "stat_totals"  (default "ev_roto")
//   inverseCats: array of cat codes where lower is better (default ["ERA","WHIP"])
//   rateCats:    array of cat codes that need 3-decimal formatting
//                (default ["AVG","ERA","WHIP"])
(function () {
  const DEFAULT_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"];

  function esc(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fmtVal(val, cat, isRate, isStatTotals) {
    if (typeof val !== "number" || !isFinite(val)) return "—";
    if (isStatTotals) {
      return isRate ? val.toFixed(3) : (Number.isInteger(val) ? String(val) : val.toFixed(1));
    }
    // roto / ev_roto are points
    return val.toFixed(2);
  }

  function fmtDelta(d, cat, isRate, isStatTotals) {
    if (typeof d !== "number" || !isFinite(d)) return "";
    const sign = d > 0 ? "+" : "";
    if (isStatTotals) {
      return sign + (isRate ? d.toFixed(3) : (Math.abs(d) >= 1 ? d.toFixed(1) : d.toFixed(2)));
    }
    return sign + d.toFixed(2);
  }

  function deltaClass(d, cat, isInverse, isStatTotals) {
    if (typeof d !== "number" || Math.abs(d) < 1e-6) return "";
    if (isStatTotals && isInverse) {
      // For ERA/WHIP: lower is better, so negative delta = improvement.
      return d < 0 ? "cat-gain" : "cat-loss";
    }
    return d > 0 ? "cat-gain" : "cat-loss";
  }

  function renderTable(view, opts) {
    const cats = opts.cats;
    const isStatTotals = opts.mode === "stat_totals";
    const inverse = new Set(opts.inverseCats);
    const rate = new Set(opts.rateCats);

    const rows = cats.map((c) => {
      const cv = (view.categories || {})[c] || {before: 0, after: 0, delta: 0};
      const isRate = rate.has(c);
      const isInverse = inverse.has(c);
      const cls = deltaClass(cv.delta, c, isInverse, isStatTotals);
      return (
        '<tr><th>' + esc(c) + '</th>' +
        '<td>' + fmtVal(cv.before, c, isRate, isStatTotals) + '</td>' +
        '<td>' + fmtVal(cv.after, c, isRate, isStatTotals) + '</td>' +
        '<td class="' + cls + '">' + fmtDelta(cv.delta, c, isRate, isStatTotals) + '</td></tr>'
      );
    }).join("");

    let totalRow = "";
    if (!isStatTotals) {
      const t = view.delta_total || 0;
      const cls = t > 0 ? "cat-gain" : t < 0 ? "cat-loss" : "";
      const sign = t >= 0 ? "+" : "";
      totalRow =
        '<tr><th>Total</th><td></td><td></td>' +
        '<td class="' + cls + '">' + sign + t.toFixed(2) + '</td></tr>';
    }

    return (
      '<table class="trade-details open">' +
      '<thead><tr><th>Cat</th><th>Before</th><th>After</th><th>&Delta;</th></tr></thead>' +
      '<tbody>' + rows + totalRow + '</tbody></table>'
    );
  }

  function renderToggle(currentMode) {
    const modes = [
      {id: "roto", label: "Roto Points"},
      {id: "ev_roto", label: "eROTO"},
      {id: "stat_totals", label: "Stat Totals"},
    ];
    return (
      '<div class="three-mode-toggle" role="tablist">' +
      modes.map((m) => {
        const cls = m.id === currentMode ? "pill active" : "pill";
        return '<button type="button" class="' + cls + '" data-mode="' + m.id + '">' +
               esc(m.label) + '</button>';
      }).join("") +
      '</div>'
    );
  }

  function render(targetEl, payload, opts) {
    if (!targetEl || !payload) return;
    const cats = (opts && opts.cats) || DEFAULT_CATS;
    const inverseCats = (opts && opts.inverseCats) || ["ERA", "WHIP"];
    const rateCats = (opts && opts.rateCats) || ["AVG", "ERA", "WHIP"];
    let mode = (opts && opts.initialMode) || "ev_roto";
    if (!["roto", "ev_roto", "stat_totals"].includes(mode)) mode = "ev_roto";

    function paint() {
      const view = payload[mode] || {delta_total: 0, categories: {}};
      const renderOpts = {mode, cats, inverseCats, rateCats};
      targetEl.innerHTML = renderToggle(mode) + renderTable(view, renderOpts);
      const buttons = targetEl.querySelectorAll(".three-mode-toggle .pill");
      buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
          const next = btn.dataset.mode;
          if (next && next !== mode) {
            mode = next;
            paint();
          }
        });
      });
    }
    paint();
  }

  window.renderThreeModeDeltaView = render;
})();
