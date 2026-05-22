import unicodedata


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
