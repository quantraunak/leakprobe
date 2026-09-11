"""Ways to move when a source becomes knowable.

A perturbation changes *availability*, never values. That distinction is what
makes the test sound: if a feature's number changes after a perturbation, the
feature read something whose timing moved, and timing is the only thing that
moved.
"""

from __future__ import annotations

import numpy as np
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


def truncate(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows after `cutoff` removed entirely, rather than made late.

    Stronger than `delay` for the commonest bug. A feature built as of a cutoff
    cannot notice the deletion of rows it was never allowed to see, so any
    movement is proof it saw them. `delay` misses this whenever the offending
    code ignores the clock rather than mis-filtering on it -- a global mean or a
    z-score denominator computed over all of time responds to `delay` exactly as
    a correct feature does, because the visible slice moved either way.

    Changes the row count, so a feature frame indexed by entities present only
    after the cutoff will change shape. That is reported rather than compared.
    """
    _check(frame, column)
    return frame[frame[column] <= cutoff].reset_index(drop=True)


def shuffle(frame: pd.DataFrame, column: str, seed: int = 0) -> pd.DataFrame:
    """Timestamps kept, every other column independently permuted.

    For testing a source a feature claims *not* to read. Such a feature must be
    invariant to arbitrary changes in that source's contents, which makes this
    the strongest admissible perturbation: it breaks every row-to-row
    association while leaving the clock and the row count intact.

    `delay` cannot do this job. A feature that reads an undeclared source's
    payload while ignoring its clock -- joining a disposition table for an
    outcome flag, say -- does not move when that table's timestamps shift, and
    the dependency stays invisible. Only apply this to undeclared pairs: a
    feature that legitimately reads a source will move, and should.
    """
    _check(frame, column)
    generator = np.random.default_rng(seed)
    out = frame.copy()
    for name in out.columns:
        if name != column:
            out[name] = generator.permutation(out[name].to_numpy())
    return out
