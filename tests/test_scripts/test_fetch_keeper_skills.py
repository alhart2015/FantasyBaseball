"""Unit tests for the keeper-skills ingest script's presentation helpers.

The derivation itself lives in `fantasy_baseball.keepers.skills` and is tested in
`tests/test_keepers/test_skills.py`; what is script-local is attaching names and
repairing Baseball Reference's double-encoded accents.
"""

import pandas as pd

from scripts.fetch_keeper_skills import unescape_name, with_names

# Spelled out rather than written as a literal so this file stays plain ASCII:
# the 17-character string BBRef actually returns for "Acuna" with a tilde.
ESCAPED_ACUNA = "Luisangel Acu" + chr(92) + "xc3" + chr(92) + "xb1a"
N_TILDE = chr(0x00F1)


def test_unescape_repairs_double_encoded_accent():
    assert unescape_name(ESCAPED_ACUNA) == "Luisangel Acu" + N_TILDE + "a"


def test_unescape_leaves_plain_ascii_untouched():
    assert unescape_name("Andrew Abbott") == "Andrew Abbott"


def test_unescape_leaves_an_already_correct_accent_untouched():
    name = "Jos" + chr(0x00E9) + " Alvarado"
    assert unescape_name(name) == name


def test_unescape_returns_input_when_it_does_not_round_trip():
    """A lone backslash escape that is not valid UTF-8 must not raise or mangle."""
    assert unescape_name("Weird" + chr(92) + "xff") == "Weird" + chr(92) + "xff"


def test_with_names_prepends_repaired_name_keyed_by_mlbam():
    skills = pd.DataFrame({"pa": [600, 500]}, index=pd.Index([11, 22], name="mlbam_id"))
    source = pd.DataFrame({"mlbID": [22, 11], "Name": [ESCAPED_ACUNA, "Andrew Abbott"]})
    out = with_names(skills, source)
    assert list(out.columns) == ["name", "pa"]
    assert out.loc[11, "name"] == "Andrew Abbott"
    assert out.loc[22, "name"] == "Luisangel Acu" + N_TILDE + "a"


def test_with_names_blanks_an_unmatched_id_rather_than_dropping_it():
    """A player missing from the name source must keep his skill row."""
    skills = pd.DataFrame({"pa": [600]}, index=pd.Index([999], name="mlbam_id"))
    source = pd.DataFrame({"mlbID": [11], "Name": ["Andrew Abbott"]})
    out = with_names(skills, source)
    assert len(out) == 1
    assert out.loc[999, "name"] == ""
    assert out.loc[999, "pa"] == 600
