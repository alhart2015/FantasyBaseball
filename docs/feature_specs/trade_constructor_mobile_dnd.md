# Trade constructor: mobile drag-and-drop

**Status:** Spec (2026-04-23)
**Scope:** `src/fantasy_baseball/web/templates/season/waivers_trades.html`, plus a new vendored asset under `src/fantasy_baseball/web/static/vendor/`.

## Problem

The trade constructor (Waivers & Trades page) uses native HTML5 drag-and-drop (`draggable="true"`, `dragstart`/`dragover`/`drop` event handlers in `waivers_trades.html:399,498,510-553`). Those events don't fire on touch devices — so the trade constructor is unusable on iOS Safari and Android Chrome. This is the only page in the app that uses DnD, so the fix is scoped to one template.

## Goal

Trade constructor works on phones and tablets with the same interactions the desktop version has: drag a chip from a slot/bench/drops zone (on either side) to another slot/bench/drops zone; drag a waiver suggestion onto the user's side. IL chips remain non-interactive.

## Approach

Replace the hand-rolled HTML5 DnD layer with [SortableJS](https://github.com/SortableJS/Sortable) (MIT-licensed, ~30 KB minified). SortableJS handles mouse + touch uniformly via Pointer Events and internal fallbacks. The state model (`state.placements`, `state.waiverMeta`, `placeAt`), legality logic, and payload construction are unchanged — only the drag UX layer swaps.

## Delivery

Vendored, not CDN. Added files:

- `src/fantasy_baseball/web/static/vendor/Sortable.min.js` — the minified library, from the official GitHub release (target version: latest stable 1.15.x or 1.16.x).
- `src/fantasy_baseball/web/static/vendor/LICENSE` — a copy of the upstream MIT license, as required by MIT.

Loaded via `<script src="{{ url_for('static', filename='vendor/Sortable.min.js') }}">` in `waivers_trades.html`.

Rationale: this project has no asset pipeline and no other vendored JS dependencies; vendoring keeps behavior identical online and offline and removes a third-party runtime dependency. Upgrade cost is manual file replacement (~once a year at most for this library).

## Sortable wiring

All drop zones — every `[data-bt-side][data-bt-zone]` container except IL — become Sortable instances in the `trade-builder` group. The waiver suggestion list is a one-way source (pull allowed, put forbidden).

```js
// Replaces attachDragHandlers() entirely.
function attachSortables() {
  // Drop zones on both sides: slots, BN, DROP. Exclude IL (read-only).
  for (const el of document.querySelectorAll('[data-bt-side][data-bt-zone]')) {
    if (el.dataset.btZone === 'IL') continue;
    new Sortable(el, {
      group: 'trade-builder',
      onStart: handleStart,
      onEnd: handleDrop,
    });
  }
  // Waiver suggestions: pull-only source.
  const waiver = document.getElementById('bt-waiver-suggestions');
  if (waiver) {
    new Sortable(waiver, {
      group: { name: 'trade-builder', put: false },
      sort: false,
      onStart: handleStart,
      onEnd: handleDrop,
    });
  }
}

function handleStart(evt) {
  const chip = evt.item;
  if (chip.dataset.btWaiverName) {
    state.waiverMeta.set(chip.dataset.btDragKey, {
      name: chip.dataset.btWaiverName,
      positions: chip.dataset.btWaiverPositions || '',
    });
  }
}

function handleDrop(evt) {
  if (!evt.to || evt.to === evt.from) return;
  const key = evt.item.dataset.btDragKey;
  const side = evt.to.dataset.btSide;
  const zone = evt.to.dataset.btZone;
  if (!key || !side || !zone) return;
  placeAt(key, side, zone);
}
```

## Re-render handling

`renderPanels()` rebuilds slot containers via `innerHTML` on the grid, which destroys Sortable instances bound to those containers. Call `attachSortables()` at the end of `renderPanels()` so bindings are fresh after every state change. Re-init cost is negligible (~10 containers, one call per placement).

A cleaner alternative — refactor `renderPanels()` to update chip contents without destroying slot containers — is out of scope for this fix. The re-init pattern is simple, local to the file, and easily removed later if someone wants to do that refactor.

## Removals

- `attachDragHandlers()` (lines 507–554) — replaced by `attachSortables()`.
- `draggable="true"` attribute in `renderChip()` (line 399) and in the waiver suggestion `<li>` render (line 498). SortableJS sets draggability via its internal mechanism; the attribute is no longer needed.
- The `data-bt-drag-key` / `data-bt-waiver-name` / `data-bt-waiver-positions` attributes stay — they carry the identity/metadata that the handlers read.

## IL zone

Currently IL chips are not draggable (`dragAttr = il ? "" : ...` at line 399). Under SortableJS, we simply skip the IL containers when binding Sortable instances. IL chips stay in the DOM but have no drag affordance, matching the current behavior.

## Non-goals

- No animations, ghost-styling, or other SortableJS polish options — accept defaults.
- No mobile-specific layout changes (tap-target sizing, viewport tweaks, responsive breakpoint adjustments).
- No JS extraction from the template; the inline `<script>` block stays where it is.
- No automated tests. DnD is a UI behavior that requires a browser; the existing test suite is Python-only and doesn't cover browser interaction.
- No changes to the waivers side of the page (autosuggest, search API) beyond removing `draggable="true"` from the suggestion `<li>`.
- No changes to legality / payload / submit flow.

## Verification

Manual, across three browsers:

- **Desktop Chrome/Firefox/Safari:** drag chips between slots, BN, DROP, on both my-side and opp-side; drag a waiver suggestion onto my side; confirm IL chips cannot be dragged; confirm legality indicators still update; submit a trade and verify the payload matches what desktop used to send before this change.
- **iOS Safari (phone):** same interactions. The drag should now work where it previously didn't.
- **Android Chrome:** same.

## Rollout

Single feature branch (`trade-swap-on-mobile`), single PR, no feature flag. Vendored asset is bundled with the template change in the same commit set — avoids a broken state where the `<script>` tag references a missing file.
