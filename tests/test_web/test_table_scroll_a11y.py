"""Every .table-scroll wrapper must be keyboard-operable.

`.table-scroll` sets `overflow-x: auto`, which makes the element a scroll
container. A scroll container is only reachable by keyboard if it can take
focus, and most of the tables it wraps (ERoto, Monte Carlo, category risk,
draft grade) contain no focusable cells of their own -- so without `tabindex`
the right-hand columns are unreachable without a mouse. That is the axe/WCAG
2.1.1 "scrollable region must have keyboard access" failure.

The attributes are hand-written at every wrap site rather than produced by a
shared macro (a `{% call %}` wrapper was judged less readable than the two
lines it would replace), so this pins the invariant at the source instead:
copying just `class="table-scroll"` without the rest now fails here.

Scope note: this reads template *source*, so it cannot see which wrappers a
given request actually renders. Wrappers inside a macro carry a parameterised
label (`aria-label="{{ label }}"`), which is one literal here but many strings
at render time; those are resolved to their macro CALL SITES so they can be
compared alongside the literal ones -- see
test_region_labels_are_distinct_within_a_template.
"""

import re
from pathlib import Path

import pytest

import fantasy_baseball.web as web_pkg

TEMPLATE_DIR = Path(web_pkg.__file__).parent / "templates" / "season"

# Any element, either quote style: a wrapper written as <section class='...'>
# must fail the attribute assertions rather than slip past this regex.
WRAPPER_RE = re.compile(
    r"""<\w+\b[^>]*\bclass=["'][^"']*(?<![\w-])table-scroll(?![\w-])[^"']*["'][^>]*>"""
)
LABEL_RE = re.compile(r'aria-label="([^"]*)"')
# Jinja {# #} and HTML <!-- --> comments: commented-out markup must not prop up
# the anti-vacuity floor or be asserted against.
COMMENT_RE = re.compile(r"{#.*?#}|<!--.*?-->", re.S)


TEMPLATE_TEXT = {
    p.name: COMMENT_RE.sub("", p.read_text(encoding="utf-8"))
    for p in sorted(TEMPLATE_DIR.glob("*.html"))
}


def _wrappers():
    """(template name, index within that template, opening tag) per .table-scroll."""
    for name, text in TEMPLATE_TEXT.items():
        for i, match in enumerate(WRAPPER_RE.finditer(text)):
            yield name, i, match.group(0)


ALL_WRAPPERS = list(_wrappers())


def _is_parameterised(tag):
    """True when a wrapper's label is supplied by its enclosing macro rather than
    written literally."""
    match = LABEL_RE.search(tag)
    return bool(match and "{{" in match.group(1))


def test_table_scroll_wrappers_are_found():
    """Guard the guard: if the regex stops matching, every assertion below would
    pass vacuously."""
    assert len(ALL_WRAPPERS) >= 12, (
        f"expected the known .table-scroll wrappers, found {len(ALL_WRAPPERS)} -- "
        "if wrappers were intentionally removed, lower this floor deliberately"
    )


@pytest.mark.parametrize(
    "tag",
    [w[2] for w in ALL_WRAPPERS],
    ids=[f"{name}#{idx}" for name, idx, _ in ALL_WRAPPERS],
)
def test_table_scroll_is_keyboard_reachable(tag):
    assert 'tabindex="0"' in tag, f".table-scroll without tabindex -- {tag}"
    assert 'role="region"' in tag, f".table-scroll without role=region -- {tag}"
    label = LABEL_RE.search(tag)
    assert label and label.group(1).strip(), (
        f".table-scroll without a non-empty aria-label -- {tag}"
    )


# A macro whose wrapper label is parameterised, e.g.
#   {% macro eroto_table(data, cell_class, label) %} ... aria-label="{{ label }}"
MACRO_DEF_RE = re.compile(r"{%-?\s*macro\s+(\w+)\s*\(([^)]*)\)\s*-?%}(.*?){%-?\s*endmacro", re.S)


def _split_args(text):
    """Split an argument list on top-level commas, so a positional index into it
    lines up with the macro's parameter list (a literals-only scan would silently
    skip non-literal arguments and shift every index)."""
    args, depth, quote, cur = [], 0, None, ""
    for ch in text:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote, cur = ch, cur + ch
        elif ch in "([{":
            depth, cur = depth + 1, cur + ch
        elif ch in ")]}":
            depth, cur = depth - 1, cur + ch
        elif ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def _unquote(arg):
    """The string a Jinja literal denotes, or None if it isn't one. Either quote
    style -- the templates use both."""
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
        return arg[1:-1]
    return None


def _label_param_index(text):
    """{macro name: position of its label parameter} for every macro in `text`
    that wraps a table in a .table-scroll with a parameterised aria-label."""
    out = {}
    for name, params, body in MACRO_DEF_RE.findall(text):
        for tag in WRAPPER_RE.finditer(body):
            label = LABEL_RE.search(tag.group(0))
            if label and "{{" in label.group(1):
                var = label.group(1).strip(" {}")
                # split on top-level commas, then drop any `=default`, so a
                # parameter written `label=""` still maps to its position
                names = [a.split("=")[0].strip() for a in _split_args(params)]
                if var in names:
                    out[name] = names.index(var)
    return out


def _build_wrapper_macros():
    """{macro name: label parameter position} across every template. Two macros
    sharing a name would silently resolve call sites at the wrong argument, so
    that is refused rather than last-one-wins."""
    macros = {}
    for name, text in TEMPLATE_TEXT.items():
        for macro, idx in _label_param_index(text).items():
            assert macros.get(macro, idx) == idx, (
                f"{name}: two wrapper macros named {macro} with different label "
                "positions -- call sites cannot be resolved unambiguously"
            )
            macros[macro] = idx
    return macros


WRAPPER_MACROS = _build_wrapper_macros()


def _labels_rendered_by(name):
    """Every aria-label a template contributes to its own page: the literal ones
    on its wrappers, plus the ones it passes into a wrapper macro. Macros may be
    defined in another template (the macros.html + import pattern), so calls are
    resolved against the macro table built from all of them."""
    text = TEMPLATE_TEXT[name]
    labels = []
    for tag in WRAPPER_RE.finditer(text):
        match = LABEL_RE.search(tag.group(0))
        if match and "{{" not in match.group(1):
            labels.append(match.group(1))
    for macro, idx in WRAPPER_MACROS.items():
        for call in re.finditer(r"{{-?\s*" + macro + r"\s*\((.*?)\)\s*-?}}", text, re.S):
            args = _split_args(call.group(1))
            assert len(args) > idx, f"{name}: {macro}(...) passes no label -- {call.group(0)}"
            literal = _unquote(args[idx])
            assert literal is not None, (
                f"{name}: {macro} label is not a string literal ({args[idx]}) -- "
                "this guard can only compare literals"
            )
            labels.append(literal)
    return labels


def test_every_parameterised_wrapper_maps_to_a_macro():
    """Anti-vacuity, and stronger than a count floor: a wrapper whose label comes
    from its macro is only checked if that macro was recognised. `_label_param_index`
    only recognises a label that is exactly `{{ param }}`, so a composed one like
    `aria-label="Team {{ label }}"` would drop its whole macro out of the scan --
    and a plain `len(...) >= 3` floor cannot see that, because the three existing
    macros still satisfy it. Compare the two populations instead."""
    found = sum(1 for _, _, tag in ALL_WRAPPERS if _is_parameterised(tag))
    mapped = sum(
        1
        for text in TEMPLATE_TEXT.values()
        for macro, _params, body in MACRO_DEF_RE.findall(text)
        if macro in WRAPPER_MACROS
        for tag in WRAPPER_RE.finditer(body)
        if _is_parameterised(tag.group(0))
    )
    assert found >= 3, f"expected the standings wrapper macros, found {found}"
    assert found == mapped, (
        f"{found - mapped} parameterised .table-scroll wrapper(s) belong to no "
        f"recognised macro, so their call-site labels are never checked "
        f"(recognised: {sorted(WRAPPER_MACROS)})"
    )


@pytest.mark.parametrize("template", sorted(TEMPLATE_TEXT))
def test_region_labels_are_distinct_within_a_template(template):
    """Two identically-named regions on one page are indistinguishable in a
    screen reader's landmark list. Literal and macro-supplied labels are checked
    TOGETHER -- a macro call colliding with a literal wrapper is the easy mistake
    to make, and checking the two sets separately would miss exactly that."""
    labels = _labels_rendered_by(template)
    assert all(x.strip() for x in labels), f"{template}: empty aria-label among {labels}"
    assert len(labels) == len(set(labels)), f"{template}: duplicate region labels {labels}"
