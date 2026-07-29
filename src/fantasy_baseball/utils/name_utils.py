import codecs
import unicodedata


def repair_double_encoded(name: str) -> str:
    """Undo a name whose UTF-8 bytes arrived spelled out as literal escapes.

    Baseball Reference returns roughly 7% of names this way -- "Acu\\xc3\\xb1a"
    as 17 characters rather than "Acuna" with a tilde. Left alone it survives
    :func:`normalize_name` as a distinct key (accent-stripping has nothing to
    strip) and silently matches nothing on the board, so repair must happen
    before any normalized join. Anything that does not round-trip, including a
    non-string, is returned unchanged.
    """
    if not isinstance(name, str):
        return name
    try:
        return codecs.decode(name, "unicode_escape").encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def normalize_name(name: str) -> str:
    """Normalize a player name for comparison.

    Strips Unicode accents, lowercases, and removes extra whitespace.
    'José Ramírez' -> 'jose ramirez'
    'Julio Rodríguez' -> 'julio rodriguez'
    """
    # Bad/blank source rows arrive as NaN (float) or None — pandas yields a
    # float for an empty CSV name cell. Treat any non-string as no name rather
    # than letting unicodedata.normalize raise "argument 2 must be str".
    if not isinstance(name, str):
        return ""
    # Decompose Unicode characters, strip combining marks (accents)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.lower().strip()
