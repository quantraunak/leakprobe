"""A churn model that cheats, and gets caught.

Run:  python examples/churn_features.py

The bug is one line, it raises nothing, and it improves your metrics. That is
what makes temporal leakage the most expensive class of quiet bug in applied ML.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import leakcheck as lc

rng = np.random.default_rng(7)
CUTOFF = pd.Timestamp("2024-06-01")

customers = pd.DataFrame({"customer_id": range(200)})

events = pd.DataFrame(
    {
        "customer_id": rng.integers(0, 200, 4000),
        "occurred_at": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 250, 4000), "D"),
        "amount": rng.gamma(2.0, 30.0, 4000).round(2),
    }
)

# A support ticket table. `resolved_at` is when the row reached the warehouse.
tickets = pd.DataFrame(
    {
        "customer_id": rng.integers(0, 200, 600),
        "opened_at": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 250, 600), "D"),
        "severity": rng.integers(1, 4, 600),
    }
)
tickets["resolved_at"] = tickets["opened_at"] + pd.to_timedelta(rng.integers(1, 40, 600), "D")


def build_features(src):
    """Everything a customer looked like as of CUTOFF."""
    ev, tk, cust = src["events"], src["tickets"], src["customers"]

    visible_events = ev[ev["occurred_at"] <= CUTOFF]
    spend = visible_events.groupby("customer_id")["amount"].sum()
    frequency = visible_events.groupby("customer_id").size()

    # Correct: a ticket is knowable once resolved.
    visible_tickets = tk[tk["resolved_at"] <= CUTOFF]
    tickets_resolved = visible_tickets.groupby("customer_id").size()

    # The bug. Filters on when the ticket was *opened*, so tickets still
    # unresolved at CUTOFF -- the ones that predict churn -- leak in.
    leaked_tickets = tk[tk["opened_at"] <= CUTOFF]
    open_severity = leaked_tickets.groupby("customer_id")["severity"].mean()

    index = cust["customer_id"]
    return pd.DataFrame(
        {
            "total_spend": spend.reindex(index).fillna(0.0).to_numpy(),
            "event_count": frequency.reindex(index).fillna(0).to_numpy(),
            "tickets_resolved": tickets_resolved.reindex(index).fillna(0).to_numpy(),
            "avg_severity": open_severity.reindex(index).fillna(0.0).to_numpy(),
        }
    )


report = lc.check(
    compute=build_features,
    sources={"events": events, "tickets": tickets, "customers": customers},
    timestamps={
        "events": "occurred_at",
        "tickets": "resolved_at",
        "customers": None,
    },
    declared={
        "total_spend": ["events"],
        "event_count": ["events"],
        "tickets_resolved": ["tickets"],
        "avg_severity": ["tickets"],
    },
)

print(report)
