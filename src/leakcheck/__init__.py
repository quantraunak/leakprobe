"""leakcheck -- find features that read data they were never supposed to see.

    import leakcheck as lc

    report = lc.check(
        compute=build_features,
        sources={"events": events, "profiles": profiles},
        timestamps={"events": "occurred_at", "profiles": "updated_at"},
        declared={"rolling_7d_spend": ["events"]},
    )
    report.raise_for_leaks()
"""

from .core import check
from .perturb import advance, delay, use_column
from .report import Finding, Report

__all__ = ["check", "delay", "advance", "use_column", "Report", "Finding"]
__version__ = "0.1.0"
