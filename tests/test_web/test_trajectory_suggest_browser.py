"""The trajectory suggestion overlay, driven in a real browser (#350).

WHY THIS EXISTS. Every other gate in this repo is blind to DOM behaviour, and three
consecutive review passes each found one input modality broken by the fix for the
previous one: the list moved into the <label> to fix its alignment and left 26 tab
stops before the Search button; moving it back out put it after that button, where the
focus guard emptied it before a keyboard reader could ever arrive; the focus guard
itself was added because Safari delivers `focusout` with a null relatedTarget and the
list vanished between mousedown and click. Each fix was correct about the defect it
named and blind to the one it created, because nothing could see the page.

So the states are enumerated here, together, and asserted in Chromium.

CHROMIUM ONLY, and that is a known gap (#364). These ran in WebKit too until the
engine was dropped for costing 300s timeouts under parallel test runs. Two of the
defects above were Safari-only -- `focusout` with a null `relatedTarget`, and a
scrollbar drag that the mousedown guard cancels differently in the two engines --
so nothing here can see them any more. If the overlay is changed again, check it
by hand in Safari; the suite will not.

Skipped unless Playwright and its browsers are installed -- `pip install playwright
&& playwright install chromium`. The suite must stay runnable without them.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from tests.test_web.test_season_routes import _trajectory_cache

#: Enough rows to overflow the 25-result cap on a common substring, so the truncation
#: notice and the list's own scrolling are both exercised rather than assumed.
POPULATION = 40
SHARED = "ar"


def _board_payload():
    """A board whose names all contain `SHARED`, built through the real sweep."""
    from fantasy_baseball.trajectory.board import BoardRow
    from fantasy_baseball.trajectory.sweep import sweep_pool, to_payload
    from tests._trajectory_panel import synthetic_panel

    rows = [
        BoardRow(
            mlbam_id=1000 + i,
            # Distinct names so a suggestion resolves to exactly one row, all sharing
            # the query substring so one keystroke pair fills the cap.
            name=f"Ar{i:02d} Marlow",
            pool="hitter",
            age=26,
            sgp=20.0 - i * 0.1,
            prior_sgp=19.0,
            slot="OF",
            floor=4.0,
        )
        for i in range(POPULATION)
    ]
    swept = sweep_pool(rows, synthetic_panel(), "hitter", (1, 2))
    return to_payload(
        swept,
        base_season=2026,
        max_horizon=2,
        min_sgp=2.0,
        generated_at="2026-08-08T09:00:00",
    )


@contextmanager
def _live_server():
    """The real app on a real port, with the board cache faked in-process.

    A threaded werkzeug server rather than the test client: the whole point is a
    browser, and the browser needs a socket. The patch is applied around the thread, so
    the server's own request handling sees it.
    """
    from werkzeug.serving import make_server

    from fantasy_baseball.web.season_app import create_app

    server = make_server("127.0.0.1", 0, create_app(), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def server():
    with _trajectory_cache(_board_payload()), _live_server() as url:
        yield url


@pytest.fixture(scope="module")
def playwright_instance():
    with sync_playwright() as pw:
        yield pw


def _engine(pw, name):
    try:
        return getattr(pw, name).launch()
    except PlaywrightError as exc:  # browsers not downloaded
        pytest.skip(f"{name} unavailable: {str(exc).splitlines()[0]}")


@pytest.fixture(params=["chromium"], scope="module")
def browser(request, playwright_instance):
    engine = _engine(playwright_instance, request.param)
    yield engine
    engine.close()


@pytest.fixture
def page(browser, server):
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    # `next` so the post-login redirect lands on a page this fixture's cache can
    # actually render -- the default is /standings, which the board payload is not.
    page.goto(server + "/login?next=/trajectory")
    page.fill('input[type="password"]', "dev")
    page.click('button[type="submit"]')
    page.wait_for_load_state("networkidle")
    yield page
    context.close()


def resolved(page):
    """The query the page ended on, parsed.

    Parsed rather than substring-matched: a plain form submit carries `pid=` too, as an
    EMPTY hidden input, so `"pid=" in url` cannot tell a resolved suggestion from an
    unresolved submit.
    """
    return parse_qs(urlparse(page.url).query, keep_blank_values=True)


def open_list(page, server, query=SHARED):
    """Type into the real input and wait for the real endpoint to answer.

    Waits for VISIBLE, not merely attached. `render` appends options one at a time to a
    list that is already in the document and unhides it only at the end, so an attached
    -state wait can return mid-loop -- and a test that then counts options is reading a
    half-built list. It passes on an idle machine and fails under a loaded one, which is
    the worst shape a gate can have.
    """
    page.goto(server + "/trajectory?view=player")
    page.wait_for_selector("#traj-player")
    page.click("#traj-player")
    page.fill("#traj-player", query)
    page.wait_for_selector('#traj-suggest a[role="option"]', state="visible", timeout=15_000)


def test_a_mouse_click_on_a_suggestion_resolves_the_player(page, server):
    """The baseline, and the one Safari broke: mousedown must not eat the click."""
    open_list(page, server)
    # `expect_navigation`, not `wait_for_load_state`: networkidle can resolve against
    # the page still on screen when the machine is loaded, and the assertion then reads
    # the pre-click URL. That is a flake, not a finding, and it cost one full-suite run.
    with page.expect_navigation():
        page.locator('#traj-suggest a[role="option"]').first.click()
    query = resolved(page)
    assert query.get("pid", [""])[0] and query.get("ppool", [""])[0], (
        "a picked suggestion must carry both discriminators so it never lands on the "
        f"candidate page; got {page.url}"
    )


def test_a_touch_tap_on_a_suggestion_resolves_the_player(browser, server):
    """iOS taps go through the emulated mouse events the focus guard cancels."""
    context = browser.new_context(
        viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True
    )
    page = context.new_page()
    try:
        page.goto(server + "/login?next=/trajectory")
        page.fill('input[type="password"]', "dev")
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")
        open_list(page, server)
        box = page.locator('#traj-suggest a[role="option"]').first.bounding_box()
        with page.expect_navigation():
            page.touchscreen.tap(box["x"] + box["width"] / 4, box["y"] + box["height"] / 2)
        assert resolved(page).get("pid", [""])[0], (
            f"a tapped suggestion must resolve, not just submit; got {page.url}"
        )
    finally:
        context.close()


def test_one_tab_reaches_the_search_button(page, server):
    """The suggestions are not tab stops -- 26 of them between an input and its
    button is what moved this list once already."""
    open_list(page, server)
    page.keyboard.press("Tab")
    page.wait_for_function(
        "() => document.activeElement !== document.getElementById('traj-player')",
        timeout=15_000,
    )
    focused = page.evaluate("() => document.activeElement.tagName")
    assert focused == "BUTTON", f"Tab should land on Search, landed on {focused}"


def test_arrow_keys_and_enter_choose_a_suggestion(page, server):
    """The other half of that trade: not tabbable means arrow-navigable, or the list
    is unreachable by keyboard entirely -- which is what shipped, in both engines."""
    open_list(page, server)
    page.keyboard.press("ArrowDown")
    page.wait_for_selector('#traj-suggest a[aria-selected="true"]', timeout=15_000)
    state = page.evaluate(
        """() => {
            const input = document.getElementById('traj-player');
            const sel = document.querySelectorAll('#traj-suggest a[aria-selected="true"]');
            return {
                active: input.getAttribute('aria-activedescendant'),
                expanded: input.getAttribute('aria-expanded'),
                selected: sel.length,
                matches: sel.length === 1 && sel[0].id === input.getAttribute('aria-activedescendant'),
            };
        }"""
    )
    assert state["expanded"] == "true", "an open list must report itself open"
    assert state["selected"] == 1, f"exactly one active option, got {state['selected']}"
    assert state["matches"], (
        "aria-activedescendant must name the selected option or AT reads a different "
        f"row than the one highlighted; active={state['active']}"
    )
    with page.expect_navigation():
        page.keyboard.press("Enter")
    assert resolved(page).get("pid", [""])[0], (
        f"Enter on the active option must resolve that player; got {page.url}"
    )


def test_enter_with_no_selection_still_submits_the_form(page, server):
    """The JS-off contract, preserved: Enter without an active row is a plain submit,
    which is what routes a partial name into the server-side candidate list."""
    open_list(page, server)
    with page.expect_navigation():
        page.keyboard.press("Enter")
    query = resolved(page)
    assert query.get("player") == [SHARED], f"the typed name must survive; got {page.url}"
    assert not any(query.get("pid", [""])), (
        f"a plain submit must not invent a resolved id; got {page.url}"
    )


def test_the_list_fits_a_phone(page, server):
    """The 320px floor plus the input's left offset ran 14px off a 375px viewport in
    both engines -- a suggestion the reader cannot see is not a suggestion."""
    page.set_viewport_size({"width": 375, "height": 812})
    open_list(page, server)
    box = page.evaluate(
        """() => {
            const r = document.getElementById('traj-suggest').getBoundingClientRect();
            return {right: Math.round(r.right), left: Math.round(r.left),
                    viewport: window.innerWidth,
                    doc: document.documentElement.scrollWidth};
        }"""
    )
    assert box["right"] <= box["viewport"], (
        f"list right edge {box['right']} overhangs a {box['viewport']}px viewport"
    )
    assert box["left"] >= 0, f"list starts off-screen at {box['left']}"
    assert box["doc"] <= box["viewport"], "the page must not scroll sideways"


def test_the_list_stays_anchored_to_the_input_across_a_reflow(page, server):
    """Position is measured once, at render. It survives a resize only because both
    the list and the input are placed inside the same offset parent -- assert that,
    so a later change to absolute/viewport coordinates cannot pass quietly."""
    open_list(page, server)
    for width in (900, 600, 420):
        page.set_viewport_size({"width": width, "height": 900})
        # Wait for layout to STOP moving rather than sleeping a guessed interval: the
        # reposition runs off the resize event, and a fixed wait is only ever tuned to
        # the machine that wrote it.
        page.wait_for_function(
            """() => {
                const l = document.getElementById('traj-suggest').getBoundingClientRect();
                const settled = window.__trajLast === Math.round(l.left);
                window.__trajLast = Math.round(l.left);
                return settled;
            }""",
            timeout=15_000,
        )
        geo = page.evaluate(
            """() => {
                const l = document.getElementById('traj-suggest').getBoundingClientRect();
                const i = document.getElementById('traj-player').getBoundingClientRect();
                return {drift: Math.round(Math.abs(l.left - i.left)),
                        gap: Math.round(l.top - i.bottom)};
            }"""
        )
        assert geo["drift"] <= 2, f"at {width}px the list drifted {geo['drift']}px from the input"
        assert 0 <= geo["gap"] <= 8, f"at {width}px the list sits {geo['gap']}px below the input"


def test_a_capped_list_scrolls_and_says_it_is_capped(page, server):
    """A silent cap makes 25-of-40 read as 25-of-25, which is the conclusion this
    feature exists to prevent."""
    open_list(page, server)
    state = page.evaluate(
        """() => {
            const l = document.getElementById('traj-suggest');
            const note = l.querySelector('.traj-suggest-note');
            const input = document.getElementById('traj-player');
            return {
                options: l.querySelectorAll('a[role="option"]').length,
                scrollable: l.scrollHeight > l.clientHeight,
                note: note ? note.textContent : null,
                described: input.getAttribute('aria-describedby'),
                noteRole: note ? note.getAttribute('role') : null,
            };
        }"""
    )
    assert state["options"] == 25, f"the cap is 25, rendered {state['options']}"
    assert state["scrollable"], "a 25-row list must scroll rather than run off the page"
    assert state["note"] and str(POPULATION) in state["note"], (
        f"the notice must name the true total; got {state['note']!r}"
    )
    assert state["noteRole"] == "presentation", (
        "a non-option child of a listbox must not be counted as an option"
    )
    assert state["described"], "the notice must be announced, not only drawn"


def test_escape_and_an_outside_click_dismiss(page, server):
    """Both used to leave the debounce timer scheduled, so the list reopened over the
    page the reader had just dismissed."""
    open_list(page, server)
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.getElementById('traj-suggest').hidden", timeout=15_000)
    assert not page.evaluate(
        "() => document.getElementById('traj-player').getAttribute('aria-activedescendant')"
    ), "a dismissed list must not leave AT pointing at a row that is gone"

    open_list(page, server)
    page.mouse.click(5, 5)
    page.wait_for_function("() => document.getElementById('traj-suggest').hidden", timeout=15_000)


def _aria(page):
    return page.evaluate(
        """() => {
            const input = document.getElementById('traj-player');
            return {
                hidden: document.getElementById('traj-suggest').hidden,
                expanded: input.getAttribute('aria-expanded'),
                active: input.getAttribute('aria-activedescendant'),
                described: input.getAttribute('aria-describedby'),
            };
        }"""
    )


def test_a_query_that_matches_nothing_closes_the_listbox(page, server):
    """Hiding the list is not the same as closing it. A bare `hidden = true` left
    aria-expanded="true" behind, so AT announced an open listbox -- and, after an
    ArrowDown, a selected row -- over a list the sighted reader can no longer see."""
    open_list(page, server)
    page.keyboard.press("ArrowDown")
    page.fill("#traj-player", "zzqq")
    page.wait_for_function("() => document.getElementById('traj-suggest').hidden", timeout=15_000)
    state = _aria(page)
    assert state["hidden"], "no matches means no list"
    assert state["expanded"] == "false", "a hidden listbox must not report itself open"
    assert state["active"] is None, "and must not still point at a row"


def test_a_cold_board_closes_the_listbox_rather_than_freezing_it(page, server):
    """The endpoint answers 503 with an `error` when the board cache is cold. Hiding
    silently is right -- the form below still works -- but the widget's state has to
    follow, and the close path must not bump the request sequence and discard a fetch
    that is still in flight."""
    open_list(page, server)
    page.keyboard.press("ArrowDown")
    page.route(
        "**/api/trajectory/find*",
        lambda route: route.fulfill(
            status=503, content_type="application/json", body='{"error": "cold board"}'
        ),
    )
    page.fill("#traj-player", "arl")
    page.wait_for_function("() => document.getElementById('traj-suggest').hidden", timeout=15_000)
    state = _aria(page)
    assert state["hidden"] and state["expanded"] == "false", state
    assert state["active"] is None and state["described"] is None, state

    # And it recovers: the next successful query reopens, which a loadSeq bump inside
    # the error path would have made intermittent.
    page.unroute("**/api/trajectory/find*")
    page.fill("#traj-player", "arlo")
    page.wait_for_selector('#traj-suggest a[role="option"]', timeout=10_000)
    assert _aria(page)["expanded"] == "true"
