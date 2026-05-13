/* Streaks page client-side helpers. Small datasets — pure DOM ops. */
(function () {
  "use strict";

  function _cellValue(row, key) {
    // Map column key → comparable value. Keys correspond to the
    // onclick="sortStreaksTable(this, '<key>', ...)" arguments in the
    // streaks.html template.
    const cells = row.children;
    switch (key) {
      case "name": return cells[0].textContent.trim().toLowerCase();
      case "pos":  return cells[1].textContent.trim().toLowerCase();
      case "avg":  return _toneOrder(cells[2]);
      case "hr":   return _toneOrder(cells[3]);
      case "r":    return _toneOrder(cells[4]);
      case "rbi":  return _toneOrder(cells[5]);
      case "sb":   return _toneOrder(cells[6]);
      case "cmp":  return parseFloat(cells[7].textContent) || 0;
      default:     return 0;
    }
  }

  function _toneOrder(cell) {
    // HOT > NEUTRAL > COLD, so HOT sorts highest in desc order.
    if (cell.querySelector(".streak-hot")) return 1;
    if (cell.querySelector(".streak-cold")) return -1;
    return 0;
  }

  window.sortStreaksTable = function (th, key, defaultDir) {
    const table = th.closest("table");
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const currentDir = th.getAttribute("data-sort-dir");
    const dir = currentDir === "asc" ? "desc" : "asc";
    rows.sort((a, b) => {
      const av = _cellValue(a, key);
      const bv = _cellValue(b, key);
      if (av < bv) return dir === "asc" ? -1 : 1;
      if (av > bv) return dir === "asc" ? 1 : -1;
      return 0;
    });
    // Reset all headers, mark this one.
    Array.from(table.querySelectorAll("th")).forEach(h => h.removeAttribute("data-sort-dir"));
    th.setAttribute("data-sort-dir", dir);
    rows.forEach(r => tbody.appendChild(r));
  };

  window.filterFaRows = function (count) {
    const limit = parseInt(count, 10);
    const rows = document.querySelectorAll("#fa-table tbody tr");
    rows.forEach(r => {
      const rank = parseInt(r.dataset.rank, 10);
      r.style.display = rank <= limit ? "" : "none";
    });
  };

  // Apply default FA-count on load.
  document.addEventListener("DOMContentLoaded", () => {
    const sel = document.getElementById("fa-count");
    if (sel) window.filterFaRows(sel.value);
  });
})();
