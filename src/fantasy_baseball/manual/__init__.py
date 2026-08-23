"""Yahoo-free adapter layer for the manual (hand-transcribed) pipeline.

When the Yahoo API is unavailable, the two things the season pipeline
cannot compute for itself -- the league standings and the ten team
rosters -- are hand-transcribed off the Yahoo web UI into YAML under
``data/manual/``, and the free-agent pool is synthesized from the ROS
projections minus everything on those rosters.

INVARIANT: this package acquires and shapes data only. It contains no
scoring, roto, optimizer or audit math. Every number the manual run
produces comes from the same ``scoring`` / ``lineup`` code the Yahoo run
uses; the modules here only put Yahoo-shaped inputs where that code
already looks for them. If you find yourself writing a category-points
calculation, a delta-roto step, or a replacement-level lookup in here,
you are in the wrong module.
"""
