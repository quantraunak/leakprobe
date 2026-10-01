"""leakprobe -- find features that read data they were never supposed to see.

    import leakprobe as lp

    report = lp.check(
        compute=build_features,
        sources={"events": events, "profiles": profiles},
        timestamps={"events": "occurred_at", "profiles": "updated_at"},
        declared={"rolling_7d_spend": ["events"]},
    )
    report.raise_for_leaks()
"""

from .core import check, discover
from .perturb import advance, delay, shuffle, truncate, use_column
from .report import Finding, Report

__all__ = [
    "Finding",
    "Report",
    "advance",
    "check",
    "delay",
    "discover",
    "shuffle",
    "truncate",
    "use_column",
]
__version__ = "0.2.3"
