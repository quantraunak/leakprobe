"""A two-second demonstration that needs nothing but the package.

    python -m leakprobe.demo

Builds a small event table, computes three features, plants one leak, and runs
the check. The leaky feature counts rows without looking at the clock, which is
the commonest shape of temporal leakage and the one that never raises.
"""
from __future__ import annotations

import pandas as pd

from . import check


def main() -> None:
    cutoff = pd.Timestamp("2024-03-01")
    events = pd.DataFrame({
        "customer": [1, 1, 1, 2, 2, 3],
        "occurred_at": pd.to_datetime([
            "2024-01-10", "2024-02-05", "2024-04-20",   # customer 1: one event AFTER the cutoff
            "2024-01-15", "2024-05-01",                 # customer 2: one event AFTER the cutoff
            "2024-02-20",
        ]),
        "amount": [10.0, 20.0, 999.0, 5.0, 999.0, 7.0],
    })

    def build_features(sources: dict) -> pd.DataFrame:
        e = sources["events"]
        visible = e[e["occurred_at"] <= cutoff]
        return pd.DataFrame({
            "spend_to_date":  visible.groupby("customer").amount.sum(),      # correct
            "events_to_date": visible.groupby("customer").size().astype(float),  # correct
            "event_count":    e.groupby("customer").size().astype(float),   # LEAK: no cutoff
        }).fillna(0.0)

    report = check(
        compute=build_features,
        sources={"events": events},
        timestamps={"events": "occurred_at"},
        declared={"spend_to_date": ["events"], "events_to_date": ["events"], "event_count": ["events"]},
        cutoff=cutoff,
    )
    print(report)
    print("\nevent_count ignores the cutoff, so it counts events that had not happened yet.")
    print("Your pipeline never raised; the number just looked fine. leakprobe found it")
    print("by deleting the rows after the cutoff and watching which feature changed.")


if __name__ == "__main__":
    main()
