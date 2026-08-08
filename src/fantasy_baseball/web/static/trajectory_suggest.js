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
   * document is load-bearing for how that block is read. The combobox ARIA goes on
   * with it, so the input never advertises a control that does not exist. */
  var list = document.createElement('ul');
  list.id = 'traj-suggest';
  list.className = 'traj-suggest';
  list.setAttribute('role', 'listbox');
  list.hidden = true;
  form.appendChild(list);

  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-controls', 'traj-suggest');
  input.setAttribute('aria-expanded', 'false');

  var DEBOUNCE_MS = 150;
  var MIN_CHARS = 2;
  var timer = null;
  // Monotonic request id. A 1,169-row scan is fast enough that typing overlaps two
  // fetches easily, and without this an earlier reply landing later would render a
  // list for a query the box no longer holds.
  var loadSeq = 0;

  function hide() {
    list.hidden = true;
    list.innerHTML = '';
    input.setAttribute('aria-expanded', 'false');
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
    return form.action.split('?')[0] + '?' + params.toString();
  }

  function render(players) {
    list.innerHTML = '';
    if (!players.length) {
      hide();
      return;
    }
    players.forEach(function (hit) {
      var li = document.createElement('li');
      li.setAttribute('role', 'option');
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
        if (data.error || !data.players) { hide(); return; }
        render(data.players);
      })
      .catch(function () {
        if (reqId !== loadSeq) return;
        hide();
      });
  }

  input.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      var q = input.value.trim();
      if (q.length < MIN_CHARS) {
        // Bump the sequence so a fetch already in flight cannot render after the box
        // has been cleared back below the minimum.
        loadSeq++;
        hide();
        return;
      }
      search(q);
    }, DEBOUNCE_MS);
  });

  // Escape closes without submitting; a click elsewhere dismisses.
  input.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') hide();
  });
  document.addEventListener('click', function (event) {
    if (!form.contains(event.target)) hide();
  });
})();
