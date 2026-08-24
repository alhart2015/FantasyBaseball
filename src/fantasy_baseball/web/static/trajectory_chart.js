// One player's trajectory: career solid, projection dashed, p10-p90 as a filled band,
// comps drawn legibly across the projected ages.
//
// THE COMPS ARE MEANT TO BE READ (#358). They used to be near-invisible, and correctly
// so: they were chosen by how close their realized path landed to THIS forecast, so drawn
// as equals they hugged the projection and made it look far more certain than it was.
// They are now chosen on their REALIZED CAREER before the match age, with no predicted
// value entering, and what they draw is what actually happened to those players. That
// fan is real information about the forecast rather than a redrawing of it, so hiding it
// understates the uncertainty exactly as drawing the old ones boldly overstated it.
//
// DRAWING ORDER, and it inverted with the meaning. Chart.js paints HIGHER `order` first,
// so the band fill now sits at the bottom and the comps are drawn OVER it, with the
// projection and the career line on top of both. Comps beneath a 0.18-alpha fill were
// being washed out by the very band they exist to be compared against.
//
// THE Y-AXIS IS SCALED BY THE COMPS, and that is not a defect to correct. Chart.js
// autoscales over every dataset, so a comp who collapsed to 0 or held his peak widens
// the axis and the p10-p90 band occupies less of it. The old forward comps hugged the
// projection and could never do this. It is the honest picture: if the model's 80%
// interval is narrow next to what actually befell players who looked like this one, a
// reader should see that, and clamping the axis to flatter the band would be the one
// change here that manufactures confidence.
(function () {
  const node = document.getElementById("trajectory-chart-data");
  const canvas = document.getElementById("trajectory-chart");
  if (!node || !canvas) return;
  const data = JSON.parse(node.textContent);

  const at = (pts, key) => pts.map((p) => ({ x: p.age, y: p[key] }));
  // Its sibling for the `[[age, value]]` pair arrays -- history and comp careers ship
  // in that shape, `projection` ships as objects.
  const pairs = (rows) => rows.map(([age, v]) => ({ x: age, y: v }));

  // WHAT HAPPENED, as one series: realized seasons then the anchored base season.
  // Drawn as the main chart's career line AND as the head of the faint subject overlay
  // on every comp card, so it is built once -- the two used to spell the anchor point's
  // join independently, and a change to how it attaches would have made the big chart
  // and the cards disagree about the same player's own line.
  const career = [
    ...pairs(data.history),
    ...(data.anchor ? [{ x: data.anchor[0], y: data.anchor[1] }] : []),
  ];
  // ...and where it is going. This is what a comp card compares its arc against.
  const subject = [...career, ...at(data.projection, "mean")];
  // Per-point styling for the career line: everything plain except the anchored base
  // season, which gets an open marker -- it is part record and part rest-of-season
  // projection, not a finished season, and it must not read as one.
  const anchorPoint = (plain, open) =>
    career.map((_, i) => (data.anchor && i === career.length - 1 ? open : plain));

  // A dashed vertical rule at the age where the comp matched the subject. Ten lines of
  // canvas rather than the chartjs-plugin-annotation CDN bundle: a second external
  // script is a second thing that can fail to load, and `ensureChartJs` would have to
  // grow a dependency-ordering concept to serve it.
  const matchLine = {
    id: "matchLine",
    afterDatasetsDraw(chart, _args, opts) {
      if (typeof opts.age !== "number") return;
      const x = chart.scales.x.getPixelForValue(opts.age);
      const { top, bottom, left, right } = chart.chartArea;
      if (x < left || x > right) return;
      const ctx = chart.ctx;
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(120,120,120,0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

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
    // Array position decides nothing here -- `order` does, and the block at the top of
    // this file explains which way round it runs. Comps stay first in the array only
    // because `fill: "+1"` on the band below resolves by array INDEX, so the band and
    // its hidden p10 partner have to stay adjacent and in that order.
    ...data.comps.map((c) => ({
      label: `${c.name} (${c.season})`,
      data: c.path.map((p) => ({ x: p.age, y: p.value })),
      borderColor: "rgba(120,120,120,0.7)",
      borderWidth: 1.5,
      pointRadius: 0,
      // Above the band fill (order 4), below the projection and the career line.
      order: 3,
    })),
    {
      label: "p10-p90",
      data: at(data.projection, "p90"),
      borderColor: "transparent",
      backgroundColor: "rgba(78,121,167,0.18)",
      fill: "+1",
      pointRadius: 0,
      order: 4,
    },
    { label: "_p10", data: at(data.projection, "p10"), borderColor: "transparent",
      pointRadius: 0, fill: false, order: 4 },
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
      // ONE line, history running into the anchored base season. The anchor point is
      // styled per point rather than split into its own dataset: a second dataset
      // would be a second legend entry and a visible seam at the join, and the whole
      // point of this change is that there is no gap there.
      label: data.name,
      data: career,
      borderColor: "#4e79a7",
      borderWidth: 2.5,
      pointRadius: anchorPoint(2, 5),
      pointBackgroundColor: anchorPoint("#4e79a7", "transparent"),
      pointBorderColor: "#4e79a7",
      pointBorderWidth: anchorPoint(1, 2),
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
          tooltip: {
            filter: (item) => item.dataset.label !== "_p10",
            callbacks: {
              // The anchor point is the LAST point of the career line and looks like
              // any other on hover. Say which it is: a part-projected full-season line
              // is a different claim from a finished season.
              label: (item) => {
                const base = `${item.dataset.label}: ${item.formattedValue}`;
                const isAnchor =
                  data.anchor &&
                  item.dataset.label === data.name &&
                  item.dataIndex === career.length - 1;
                return isAnchor ? `${base} (${data.anchor_label})` : base;
              },
            },
          },
        },
      },
    });

    // ONE SMALL CHART PER COMP. Each draws the comp's WHOLE career, the subject faint
    // underneath, and a dashed rule at the age where the two matched.
    // The SAME list the table and the card headers iterate, so a card cannot title
    // itself from one array and draw itself from another.
    data.comps.forEach((comp, i) => {
      const el = document.getElementById(`comp-chart-${i}`);
      // Absent when this comp had no stored arc -- the template rendered a note in
      // place of the canvas. Skip it rather than treating it as a failure.
      if (!el || !comp.career || !comp.career.length) return;
      new Chart(el.getContext("2d"), {
        type: "line",
        data: {
          datasets: [
            {
              // The SUBJECT, faint and underneath. He is the reason this card is on
              // the page; without him the arc is just a stranger's career.
              label: data.name,
              data: subject,
              borderColor: "rgba(78,121,167,0.35)",
              borderWidth: 1,
              borderDash: [3, 3],
              pointRadius: 0,
              order: 1,
            },
            {
              label: comp.name,
              data: pairs(comp.career),
              borderColor: "#59a14f",
              borderWidth: 2,
              pointRadius: 1.5,
              order: 0,
            },
          ],
        },
        plugins: [matchLine],
        options: {
          responsive: true,
          maintainAspectRatio: false,
          parsing: false,
          // X-DOMAIN: whatever Chart.js autoscales over BOTH series -- the comp's
          // career and the subject's drawn ages. Clipping to the comp's career alone
          // would cut off the subject's projection whenever the comp retired young,
          // which is the half of the card being compared. And no domain is shared
          // across cards: forcing one would squeeze a 20-season career into a
          // 6-season card and shrink the region around the match line, which is the
          // part being read. The match line is the shared reference, not the axis.
          scales: {
            x: { type: "linear", title: { display: true, text: "age" } },
            // Title off: at 180px tall the axis label costs more width than it
            // explains, and the full-size chart directly above already carries it.
            y: { title: { display: false, text: data.axis_label } },
          },
          plugins: {
            legend: { display: false },
            // The subject's own age, identical on every card: `closest_careers` selects
            // on an EXACT age match, so the rule sits at the same x throughout, which
            // is what makes the grid comparable. One number, shipped once.
            matchLine: { age: data.age },
          },
        },
      });
    });
  }).catch(() => {
    // Chart.js failed to load from the CDN. The table already renders the
    // same numbers server-side -- leave it as the fallback rather than
    // throwing into the console and showing nothing.
  });
})();
