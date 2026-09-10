"""Find features that read a source they never declared.

The test needs no ground truth. It needs one change the answer must be
invariant to: move when a source became knowable, recompute, and read the list
of features that flinched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .perturb import delay
from .report import BYPASS, LEAK, OK, Finding, Report

__all__ = ["check"]

Sources = Mapping[str, pd.DataFrame]
Compute = Callable[[Sources], pd.DataFrame]


def _as_frame(result: object, label: str) -> pd.DataFrame:
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, pd.DataFrame):
        return result
    raise TypeError(f"{label} must return a DataFrame or Series, got {type(result).__name__}")


def _changed(left: pd.Series, right: pd.Series, tolerance: float) -> tuple[bool, float]:
    """Did this column move? NaN in the same place counts as unchanged."""
    both_null = left.isna() & right.isna()
    if (left.isna() != right.isna()).any():
        return True, float("inf")
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        gap = (left - right).abs()
        gap = gap[~both_null]
        worst = float(gap.max()) if len(gap) else 0.0
        return worst > tolerance, worst
    unequal = (left != right) & ~both_null
    return bool(unequal.any()), float(unequal.sum())


def check(
    compute: Compute,
    sources: Sources,
    timestamps: Mapping[str, str],
    declared: Mapping[str, Sequence[str]],
    *,
    by: pd.Timedelta = pd.Timedelta(days=30),
    tolerance: float = 0.0,
    perturb: Callable[[pd.DataFrame, str, pd.Timedelta], pd.DataFrame] = delay,
) -> Report:
    """Perturb each source's availability in turn; report features that moved.

    Args:
        compute: builds your features. Takes the sources dict, returns a frame
            whose columns are features. Must be deterministic -- this is checked.
        sources: name -> raw table. Each carries an availability timestamp.
        timestamps: source name -> the column holding that timestamp, or None for
            a static table with no clock, which is skipped.
        declared: feature name -> the sources it is supposed to read. A feature
            absent from this mapping is treated as declaring nothing, which is
            the strictest reading and usually what you want for a new pipeline.
        by: how far to move availability. Large enough to bite, small enough to
            leave rows in the window.
        tolerance: treat changes at or below this as no change. Leave at 0.0
            unless your pipeline has genuine floating-point nondeterminism, and
            prefer to fix that instead.
        perturb: the perturbation. `delay` removes information and is the safe
            default; see `leakcheck.perturb` for others.

    Returns:
        A Report. `report.leaks` is the list of undeclared dependencies, and
        `report.raise_for_leaks()` turns them into a test failure.
    """
    absent = set(sources) - set(timestamps)
    if absent:
        raise KeyError(
            f"No availability column given for source(s): {sorted(absent)}. "
            "Pass None for a static table that has no clock."
        )

    unknown = {s for deps in declared.values() for s in deps} - set(sources)
    if unknown:
        raise KeyError(f"declared refers to unknown source(s): {sorted(unknown)}")

    baseline = _as_frame(compute(sources), "compute")
    control = _as_frame(compute(sources), "compute")

    notes: list[str] = []
    if list(baseline.columns) != list(control.columns):
        raise RuntimeError(
            "compute() returned different columns on two identical calls. The test "
            "cannot separate a leak from nondeterminism until that is fixed."
        )
    unstable = [c for c in baseline.columns if _changed(baseline[c], control[c], tolerance)[0]]
    if unstable:
        raise RuntimeError(
            "compute() is not deterministic: "
            + ", ".join(unstable[:5])
            + (" ..." if len(unstable) > 5 else "")
            + ". Seed it, or every result below is noise."
        )

    features = [str(c) for c in baseline.columns]
    findings: list[Finding] = []

    for source in sources:
        if timestamps[source] is None:
            notes.append(
                f"note: {source!r} has no availability column, so nothing about it was "
                f"perturbed. Dependencies on it are untested."
            )
            continue
        moved_frame = _as_frame(
            compute({**sources, source: perturb(sources[source], timestamps[source], by)}),
            "compute",
        )
        if list(moved_frame.columns) != list(baseline.columns):
            notes.append(
                f"note: perturbing {source!r} changed the feature set itself, which is "
                f"a stronger dependency than this test is designed to describe."
            )
            continue

        for column, feature in zip(baseline.columns, features):
            moved, worst = _changed(baseline[column], moved_frame[column], tolerance)
            is_declared = source in declared.get(feature, ())
            if moved and not is_declared:
                kind = LEAK
            elif is_declared and not moved:
                kind = BYPASS
            else:
                kind = OK
            findings.append(
                Finding(
                    feature=feature,
                    source=source,
                    kind=kind,
                    declared=is_declared,
                    moved=moved,
                    max_abs_change=worst,
                )
            )

    return Report(
        findings=findings, features=features, sources=list(sources), notes=notes
    )
