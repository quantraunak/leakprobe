"""Ways to move when a source becomes knowable.

A perturbation changes *availability*, never values. That distinction is what
makes the test sound: if a feature's number changes after a perturbation, the
feature read something whose timing moved, and timing is the only thing that
moved.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["delay", "advance", "use_column"]


def _check(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise KeyError(
            f"No availability column {column!r}. Columns present: {list(frame.columns)}"
        )
    if not pd.api.types.is_datetime64_any_dtype(frame[column]):
        raise TypeError(
            f"Availability column {column!r} is {frame[column].dtype}, not a datetime. "
            "Parse it with pd.to_datetime first."
        )


def delay(frame: pd.DataFrame, column: str, by: pd.Timedelta) -> pd.DataFrame:
    """The same rows, knowable `by` later.

    The safe default. Delaying a source can only remove information, so a
    feature that legitimately reads it will change and one that does not cannot.
    """
    _check(frame, column)
    out = frame.copy()
    out[column] = out[column] + by
    return out


def advance(frame: pd.DataFrame, column: str, by: pd.Timedelta) -> pd.DataFrame:
    """The same rows, knowable `by` earlier -- look-ahead, injected on purpose.

    Use when you want to measure what a leak would be *worth* rather than
    whether one exists.
    """
    _check(frame, column)
    out = frame.copy()
    out[column] = out[column] - by
    return out


def use_column(frame: pd.DataFrame, column: str, source: str) -> pd.DataFrame:
    """Availability taken from another column.

    Models the specific mistake of treating a record as knowable when the period
    it describes ended, rather than when it was published: `use_column(facts,
    "available_at", "period_end")`.
    """
    _check(frame, column)
    if source not in frame.columns:
        raise KeyError(f"No column {source!r} to take availability from.")
    out = frame.copy()
    out[column] = out[source]
    return out
