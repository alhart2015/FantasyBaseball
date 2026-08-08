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

  // Spec requirement 6: on the VAR scale every series is netted against THIS
  // player's own slot floor, not each comp's own -- the axis must say so, or a reader
  // comparing a comp line to a remembered SGP figure is off by the floor with no way
  // to find out why.
  //
  // SERVER-BUILT (`PlayerView.axis_label`), not rebuilt here from `floor`. This file
  // used to interpolate its own copy while the template's table header interpolated
  // another, so one rule about a subtraction was spelled twice in two languages and
  // the two were free to drift. The island now carries the finished string and no
  // longer carries `floor` or `scale` at all.

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
        // `.chart-wrapper` fixes the box at 360px (season.css); Chart.js's own
        // defaults are `responsive: true, maintainAspectRatio: true` at a 2:1
        // ratio, which ignores that box entirely and draws a canvas roughly
        // twice the box's height -- painting over the honesty paragraph below
        // it. `season_trends.js`'s `buildChart` sets this same pair for the
        // same reason; match it rather than inherit the default.
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        scales: {
          x: { type: "linear", title: { display: true, text: "age" } },
          y: { title: { display: true, text: data.axis_label } },
        },
        plugins: {
          legend: { labels: { filter: (item) => item.text !== "_p10" } },
          // The legend filter above hides the internal fill-target dataset from the
          // legend; Chart.js's default tooltip has no such filter, so hovering near
          // the lower band edge would otherwise show a series literally called
          // "_p10". Same rule, both surfaces.
          tooltip: { filter: (item) => item.dataset.label !== "_p10" },
        },
      },
    });
  }).catch(() => {
    // Chart.js failed to load from the CDN. The table already renders the
    // same numbers server-side -- leave it as the fallback rather than
    // throwing into the console and showing nothing.
  });
})();
