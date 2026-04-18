"""Rate stat utility functions.

Centralizes AVG, ERA, and WHIP computation to eliminate inline duplication.
"""


def calculate_avg(h: float, ab: float, default: float = 0.0) -> float:
    """Batting average: H / AB."""
    return h / ab if ab > 0 else default


def calculate_era(er: float, ip: float, default: float = 99.0) -> float:
    """Earned run average: ER * 9 / IP."""
    return er * 9 / ip if ip > 0 else default


def calculate_whip(bb: float, h_allowed: float, ip: float, default: float = 99.0) -> float:
    """Walks plus hits per inning pitched: (BB + H) / IP."""
    return (bb + h_allowed) / ip if ip > 0 else default


def format_ip(ip: float | None) -> str:
    """Format innings pitched in baseball notation (6.1 = 6⅓, 6.2 = 6⅔).

    IP stored as a decimal third (e.g. 6.3333…) must display as the
    canonical baseball string "6.1" — naive ``%.1f`` rounding would
    render it as "6.3" and misread as six and three-tenths.
    """
    if ip is None:
        return "\u2014"
    outs = round(float(ip) * 3)
    whole, frac = divmod(outs, 3)
    return f"{whole}.{frac}"
