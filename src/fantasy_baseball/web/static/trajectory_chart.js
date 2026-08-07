// One player's trajectory: career solid, projection dashed, p10-p90 as a filled band,
// comps thin and faint across the projected ages only.
//
// The band must stay visually dominant over the comps. They were selected ON the outcome
// -- the closest paths out of ~1,200 -- so drawn as equals they hug the projection and
// make the forecast look far more certain than it is.
(function () {
  const node = document.getElementById("trajectory-chart-data");
  const canvas = document.getElementById("trajectory-chart");
  if (!node || !canvas) return;
  const data = JSON.parse(node.textContent);

  const at = (pts, key) => pts.map((p) => ({ x: p.age, y: p[key] }));

  const datasets = [
    // Comps first so later datasets paint over them.
    ...data.comps.map((c) => ({
      label: `${c.name} (${c.season})`,
      data: c.path.map((p) => ({ x: p.age, y: p.value })),
      borderColor: "rgba(120,120,120,0.35)",
      borderWidth: 1,
      pointRadius: 0,
      order: 3,
    })),
    {
      label: "p10-p90",
      data: at(data.projection, "p90"),
      borderColor: "transparent",
      backgroundColor: "rgba(78,121,167,0.18)",
      fill: "+1",
      pointRadius: 0,
      order: 2,
    },
    { label: "_p10", data: at(data.projection, "p10"), borderColor: "transparent",
      pointRadius: 0, fill: false, order: 2 },
    {
      label: "projected",
      data: at(data.projection, "mean"),
      borderColor: "#4e79a7",
      borderDash: [6, 4],
      borderWidth: 2,
      pointRadius: 2,
      order: 1,
    },
    {
      label: data.name,
      data: data.history.map(([age, v]) => ({ x: age, y: v })),
      borderColor: "#4e79a7",
      borderWidth: 2.5,
      pointRadius: 2,
      order: 0,
    },
  ];

  if (typeof window.ensureChartJs !== "function") {
    // season_trends.js did not load (script failure, offline, etc). Leave the
    // table below the chart-wrapper as the page's real content -- degrade
    // rather than blank out.
    return;
  }

  window.ensureChartJs().then(() => {
    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets },
      options: {
        parsing: false,
        scales: {
          x: { type: "linear", title: { display: true, text: "age" } },
          y: { title: { display: true, text: data.scale.toUpperCase() } },
        },
        plugins: {
          legend: { labels: { filter: (item) => item.text !== "_p10" } },
        },
      },
    });
  }).catch(() => {
    // Chart.js failed to load from the CDN. The table already renders the
    // same numbers server-side -- leave it as the fallback rather than
    // throwing into the console and showing nothing.
  });
})();
