/* Suggest-as-you-type for the trajectory player search (#350).
 *
 * A PROGRESSIVE ENHANCEMENT over the GET form, never a replacement: with this file
 * unloaded, blocked or erroring, the form still submits and the server-side substring
 * fallback still turns a partial name into a candidate list. Everything below either
 * layers on top of that or does nothing.
 *
 * Mirrors the players page (templates/season/players.html): input listener, 150 ms
 * debounce, 2-character minimum, and the same stale-response guard.
 */
(function () {
  'use strict';

  var input = document.getElementById('traj-player');
  var form = input && input.closest('form');
  if (!input || !form) return;

  /* The list is BUILT HERE, not in the template. An empty <ul> in the markup is a
   * listbox that never fills for a reader without JS, and it would land ahead of the
   * candidate list the page renders below -- where being the first `</ul>` in the
   * document is load-bearing for how that block is read. */
  var list = document.createElement('ul');
  list.id = 'traj-suggest';
  list.className = 'traj-suggest';
  list.hidden = true;

  /* Appended to the FORM, and positioned from the INPUT's own offset.
   *
   * It briefly lived inside the <label> to fix the alignment, which was wrong three
   * ways: a <ul> is not valid label content, a label forwards clicks to its control
   * (so the truncation notice focused the search box), and it put up to 26 links
   * between the input and the Search button in tab order. It did not even fix the
   * alignment -- the label is the first flex item, so its left edge and the form's are
   * the same x, and `left: 0` still started the list under the word "Player".
   *
   * Measuring the input is what actually anchors it, and it keeps the list last in the
   * form so tab order stays input -> Search -> suggestions.
   *
   * NO combobox ARIA. An earlier version set role=combobox / listbox / option with no
   * arrow keys, no aria-activedescendant, and an <a> inside each option (interactive
   * content in an option is ignored by AT). Announcing a pattern that is not
   * implemented is worse for a screen-reader user than announcing nothing: this is a
   * list of links, so it ships as one. The players page it mirrors has no keyboard
   * navigation either, and #350 did not ask for it. */
  form.classList.add('traj-suggest-anchor');
  form.appendChild(list);

  function place() {
    list.style.left = input.offsetLeft + 'px';
    list.style.top = input.offsetTop + input.offsetHeight + 2 + 'px';
    list.style.minWidth = Math.max(input.offsetWidth, 320) + 'px';
  }

  /* Keeps focus on the input through a mousedown on the list, which is what makes the
   * click land. Browsers that do not focus a link on mousedown (Safari, iOS Safari)
   * deliver `focusout` with a null relatedTarget, and `dismiss()` emptied the list
   * between mousedown and click -- so tapping a suggestion did nothing at all and the
   * feature was unusable on those browsers. Introduced by the focusout fix; this is
   * what that fix needed to be safe. */
  list.addEventListener('mousedown', function (event) { event.preventDefault(); });

  var DEBOUNCE_MS = 150;
  var MIN_CHARS = 2;
  var timer = null;
  // Monotonic request id. A 1,169-row scan is fast enough that typing overlaps two
  // fetches easily, and without this an earlier reply landing later would render a
  // list for a query the box no longer holds.
  var loadSeq = 0;

  /* THE ONLY WAY THE LIST CLOSES. The old `hide()` cleared the DOM but left the
   * debounce timer scheduled and `loadSeq` untouched, so a list dismissed inside the
   * 150 ms window -- or while a fetch was in flight -- reopened over the page the
   * reader had just dismissed. Cancelling the timer and bumping the sequence together
   * is what makes dismissal mean dismissed; separating them is how it regressed. */
  function dismiss() {
    clearTimeout(timer);
    loadSeq++;
    list.hidden = true;
    list.innerHTML = '';
  }

  /* The query string for a suggestion, built from THE FORM'S OWN HIDDEN INPUTS.
   *
   * Not a second hand-written list of filter keys: the form already carries every key
   * in `filter_state` (view/scale/n/top/team/end/pool) as a hidden input, so reading
   * them back is the only spelling that cannot drift from it. A key added to
   * `filter_state` tomorrow is carried here with no change to this file.
   */
  function urlFor(hit) {
    var params = new URLSearchParams();
    var hidden = form.querySelectorAll('input[type="hidden"]');
    for (var i = 0; i < hidden.length; i++) {
      if (hidden[i].name) params.set(hidden[i].name, hidden[i].value);
    }
    params.set('player', hit.name);
    // pid AND ppool. The id separates same-pool namesakes (two Max Muncys); the pool
    // separates a two-way player, whose two rows share id and age. Without both, a
    // pick can land back on the candidate-disambiguation page -- which is the thing
    // this feature exists to stop happening.
    params.set('pid', hit.id);
    params.set('ppool', hit.pool);
    // Through URL, not a split on '?'. A form with no action reflects the document
    // URL including its fragment, and the fragment sits AFTER the query -- so
    // splitting on '?' leaves it attached and every parameter lands inside the hash.
    var base = new URL(form.getAttribute('action') || window.location.href,
                       window.location.href);
    base.search = params.toString();
    base.hash = '';
    return base.pathname + base.search;
  }

  function render(players, capped, total) {
    list.innerHTML = '';
    if (!players.length) {
      list.hidden = true;
      return;
    }
    players.forEach(function (hit) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = urlFor(hit);
      // textContent, not innerHTML: these names come from the board payload and one of
      // them is going into the DOM on every keystroke.
      a.textContent = hit.name;
      var meta = document.createElement('span');
      meta.className = 'traj-suggest-meta';
      // The DISCRIMINATORS. Two rows can share a name, and a two-way player's two rows
      // differ only by pool and slot -- a list showing the name alone offers two lines
      // a reader cannot choose between.
      meta.textContent = ' age ' + hit.age + ', ' + hit.slot + ' (' + hit.pool + ')';
      a.appendChild(meta);
      li.appendChild(a);
      list.appendChild(li);
    });
    if (capped) {
      // A silent cap makes 25-of-312 look like 25-of-25, and a reader whose player
      // fell past the cut concludes he is not on the board -- the conclusion this
      // feature exists to prevent.
      var note = document.createElement('li');
      note.className = 'traj-suggest-note';
      note.textContent = 'showing ' + players.length + ' of ' + total +
        ' - type more to narrow';
      list.appendChild(note);
    }
    place();
    list.hidden = false;
  }

  function search(q) {
    var reqId = ++loadSeq;
    fetch('/api/trajectory/find?q=' + encodeURIComponent(q))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (reqId !== loadSeq) return;
        // A cold board answers 503 with an `error`. Silently hiding is right: the form
        // below still works and the page it submits to reports the cold cache properly.
        if (data.error || !data.players) { list.hidden = true; return; }
        // `data.capped` is the server's answer, not a rule respelled here. Not
        // `data.total || ...`: a real 0 is falsy, and CLAUDE.md names that pattern.
        var total = data.total === undefined ? data.players.length : data.total;
        render(data.players, data.capped === undefined ? total > data.players.length
                                                       : data.capped, total);
      })
      .catch(function () {
        if (reqId !== loadSeq) return;
        list.hidden = true;
      });
  }

  input.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      var q = input.value.trim();
      if (q.length < MIN_CHARS) {
        dismiss();
        return;
      }
      search(q);
    }, DEBOUNCE_MS);
  });

  // Escape closes without submitting; a click elsewhere dismisses.
  input.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') dismiss();
  });
  document.addEventListener('click', function (event) {
    if (!form.contains(event.target)) dismiss();
  });
  // Tabbing away left the list open over the content below. `relatedTarget` inside the
  // list means the reader is tabbing INTO a suggestion, which must not close it.
  input.addEventListener('focusout', function (event) {
    if (!list.contains(event.relatedTarget)) dismiss();
  });
})();
