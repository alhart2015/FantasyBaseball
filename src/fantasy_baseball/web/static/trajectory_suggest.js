/* Suggest-as-you-type for the trajectory player search (#350).
 *
 * A PROGRESSIVE ENHANCEMENT over the GET form, never a replacement: with this file
 * unloaded, blocked or erroring, the form still submits and the server-side substring
 * fallback still turns a partial name into a candidate list. Everything below either
 * layers on top of that or does nothing.
 *
 * Mirrors the players page (templates/season/players.html): input listener, 150 ms
 * debounce, 2-character minimum, and the same stale-response guard.
 *
 * THE INVARIANT, enumerated once in a browser rather than patched a modality at a time:
 * every way a reader can reach this list -- mouse, touch, keyboard, assistive tech,
 * scrollbar -- must reach it, and nothing may dismiss it while the reader is still
 * using it. Three consecutive review passes each found one modality broken by the fix
 * for the previous one (tab order, then the focus guard, then the scrollbar), because
 * no gate in this repo could see DOM behaviour. tests/test_web/test_trajectory_suggest_browser.py
 * now drives all of it in Chromium and WebKit; run it after touching this file.
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
  list.setAttribute('role', 'listbox');

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
   * form so tab order stays input -> Search, with the suggestions reached by ArrowDown
   * rather than by Tab. */
  form.classList.add('traj-suggest-anchor');
  form.appendChild(list);

  /* THE COMBOBOX PATTERN, implemented rather than announced.
   *
   * An earlier version set role=combobox/listbox/option with no arrow keys and no
   * aria-activedescendant, so it announced a widget that did not exist; the fix for
   * that deleted the ARIA and left the list keyboard-unreachable in both engines --
   * measured, not assumed: Tab from the input lands on Search, and the focus guard
   * empties the list on the way past, so no number of Tabs ever reaches a suggestion.
   *
   * So: the options are NOT tab stops (`tabindex="-1"`), the input keeps focus, and
   * ArrowDown/ArrowUp/Enter drive the list through `aria-activedescendant`. That is
   * the one arrangement where the list is reachable by keyboard AND the Search button
   * is still one Tab away.
   *
   * role=option sits ON the anchor rather than on a wrapping <li>: an <a> inside an
   * option is interactive content that AT ignores, and the anchor is what carries the
   * href a mouse click and a middle-click both need. */
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-controls', list.id);
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-expanded', 'false');

  var active = -1;

  function options() {
    return list.querySelectorAll('a[role="option"]');
  }

  function setActive(index) {
    var opts = options();
    if (!opts.length) return;
    if (active >= 0 && opts[active]) opts[active].removeAttribute('aria-selected');
    active = (index + opts.length) % opts.length;
    var chosen = opts[active];
    chosen.setAttribute('aria-selected', 'true');
    input.setAttribute('aria-activedescendant', chosen.id);
    // `nearest`, so arrowing down one row scrolls one row instead of jumping the
    // active option to the middle of a 25-row list.
    chosen.scrollIntoView({ block: 'nearest' });
  }

  function clearActive() {
    var opts = options();
    if (active >= 0 && opts[active]) opts[active].removeAttribute('aria-selected');
    active = -1;
    input.removeAttribute('aria-activedescendant');
  }

  /* Sized and placed from the INPUT, then clamped to the FORM.
   *
   * The 320px floor is what keeps "Ronald Acuna Jr. - age 28, OF (hitter)" on one line,
   * but `left = input.offsetLeft` plus that floor ran 14px off the right edge of a
   * 375px phone in both engines. Below 320px of room the list gives up the floor and
   * slides left instead of overflowing: a suggestion the reader cannot see is not a
   * suggestion. Left and top are measured inside the form, which is the offset parent,
   * so both the input and the list move together when the page reflows.
   */
  function place() {
    var room = form.clientWidth;
    var width = Math.min(Math.max(input.offsetWidth, 320), room);
    list.style.width = width + 'px';
    list.style.left = Math.max(0, Math.min(input.offsetLeft, room - width)) + 'px';
    list.style.top = input.offsetTop + input.offsetHeight + 2 + 'px';
  }

  /* Keeps focus on the input through a mousedown on the list, which is what makes the
   * click land. Browsers that do not focus a link on mousedown (Safari, iOS Safari)
   * deliver `focusout` with a null relatedTarget, and `dismiss()` emptied the list
   * between mousedown and click -- so tapping a suggestion did nothing at all and the
   * feature was unusable on those browsers.
   *
   * UNCONDITIONAL, and a later pass should not narrow it to `event.target.closest('a')`
   * on the theory that it cancels a drag on the list's own scrollbar. Measured, both
   * engines: WebKit dispatches that mousedown (target: the UL) and scrolls anyway,
   * cancelled or not, and Chromium never routes a scrollbar band to the DOM at all --
   * its overlay scrollbar takes 2px of layout and the pointer at the right edge hits
   * the option. Narrowing buys nothing and costs a dismissal when a click lands on the
   * list's padding or on the truncation notice. */
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
    collapse();
    list.innerHTML = '';
  }

  /* Closed, without claiming the reader asked for it.
   *
   * Split out of `dismiss` because the three places that close the list on their own --
   * an empty result set, a cold-board `error`, a failed fetch -- must not bump
   * `loadSeq`: doing that from inside one response handler discards a DIFFERENT
   * request that is still in flight. They also must not leave `aria-expanded="true"`
   * on a hidden list, which is what a bare `list.hidden = true` did: sighted readers
   * see nothing while AT still announces an open listbox and a selected row. */
  function collapse() {
    clearActive();
    list.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-describedby');
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
    clearActive();
    if (!players.length) {
      collapse();
      return;
    }
    players.forEach(function (hit, index) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = urlFor(hit);
      a.id = 'traj-suggest-opt-' + index;
      a.setAttribute('role', 'option');
      // Not a tab stop. Tab is the way OUT of the widget (to Search); ArrowDown is the
      // way in. 25 tab stops between a search box and its button is the complaint that
      // moved this list once already.
      a.tabIndex = -1;
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
      note.id = 'traj-suggest-note';
      // Not an option: it is a statement about the list, and a listbox child that is
      // not an option confuses the count AT reports. Pointed at from the input instead,
      // so it is announced rather than silently dropped.
      note.setAttribute('role', 'presentation');
      note.textContent = 'showing ' + players.length + ' of ' + total +
        ' - type more to narrow';
      list.appendChild(note);
      input.setAttribute('aria-describedby', note.id);
    } else {
      input.removeAttribute('aria-describedby');
    }
    place();
    list.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }

  function search(q) {
    var reqId = ++loadSeq;
    fetch('/api/trajectory/find?q=' + encodeURIComponent(q))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (reqId !== loadSeq) return;
        // A cold board answers 503 with an `error`. Silently hiding is right: the form
        // below still works and the page it submits to reports the cold cache properly.
        if (data.error || !data.players) { collapse(); return; }
        // `data.capped` is the server's answer, not a rule respelled here. Not
        // `data.total || ...`: a real 0 is falsy, and CLAUDE.md names that pattern.
        var total = data.total === undefined ? data.players.length : data.total;
        render(data.players, data.capped === undefined ? total > data.players.length
                                                       : data.capped, total);
      })
      .catch(function () {
        if (reqId !== loadSeq) return;
        collapse();
      });
  }

  input.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      var q = input.value.trim();
      // Counted the way the SERVER counts. Counting raw characters here let a
      // decomposed glyph ("e" + U+0301: two code units, one letter) pass this gate,
      // fetch, and come back 400 -- which the catch below hides, so the reader got no
      // suggestions and no reason. Mirrors normalized_query: strip combining marks,
      // collapse runs of whitespace.
      var folded = q.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
                    .replace(/\s+/g, ' ').trim();
      if (folded.length < MIN_CHARS) {
        dismiss();
        return;
      }
      search(q);
    }, DEBOUNCE_MS);
  });

  input.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
      dismiss();
      return;
    }
    if (list.hidden || !options().length) return;
    if (event.key === 'ArrowDown') {
      // From nothing selected, ArrowDown picks the first row -- which is the board's
      // best match, so the common case is one key and Enter.
      setActive(active + 1);
      event.preventDefault();
    } else if (event.key === 'ArrowUp') {
      setActive(active <= 0 ? options().length - 1 : active - 1);
      event.preventDefault();
    } else if (event.key === 'Enter' && active >= 0) {
      // Only with a row selected. With none, Enter submits the form, which is the
      // JS-off behaviour and the server-side fallback the issue asked to keep.
      event.preventDefault();
      options()[active].click();
    }
  });

  document.addEventListener('click', function (event) {
    if (!form.contains(event.target)) dismiss();
  });
  // Tabbing away left the list open over the content below. `relatedTarget` inside the
  // list means the reader is tabbing INTO a suggestion, which must not close it.
  input.addEventListener('focusout', function (event) {
    if (!list.contains(event.relatedTarget)) dismiss();
  });
  // The width now depends on the form's, so a reflow that narrows the form has to
  // re-clamp or the list goes back to overhanging the viewport it was just fitted to.
  window.addEventListener('resize', function () {
    if (!list.hidden) place();
  });
})();
