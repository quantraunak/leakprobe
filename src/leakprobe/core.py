"""Find features that read a source they never declared.

The test needs no ground truth. It needs one change the answer must be
invariant to: move when a source became knowable, recompute, and read the list
of features that flinched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .perturb import delay, shuffle, truncate
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
    probe_undeclared: bool = True,
    cutoff: pd.Timestamp | None = None,
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
            default; see `leakprobe.perturb` for others.
        probe_undeclared: also test each source a feature claims not to read with
            `shuffle`, which permutes that source's payload while leaving its
            clock intact. `delay` alone cannot see a feature that reads an
            undeclared source's *contents* while ignoring its timestamps -- an
            outcome flag joined from a disposition table does not move when that
            table's clock shifts. A feature that genuinely does not read a source
            is invariant to arbitrary changes in it, so this costs no false
            positives; it is only ever applied to undeclared pairs. Set False to
            restore the clock-only behaviour.
        cutoff: the as-of date your features are built for. When given, each
            source is additionally truncated to rows at or before it, and any
            feature that moves is reading rows it should never have seen. This
            is the only probe that catches a statistic computed over all of time
            -- a global mean, a z-score denominator -- because such a feature
            responds to a delayed clock exactly as a correct one does. Features
            whose index changes under truncation are compared on the rows
            common to both.

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

    if cutoff is not None:
        _probe_cutoff(compute, sources, timestamps, baseline, features, tolerance,
                      cutoff, findings, notes)

    if probe_undeclared:
        _probe_undeclared(
            compute, sources, timestamps, declared, baseline, features, tolerance,
            findings, notes,
        )

    return Report(
        findings=findings, features=features, sources=list(sources), notes=notes
    )


def _probe_undeclared(compute, sources, timestamps, declared, baseline, features,
                      tolerance, findings, notes) -> None:
    """Catch reads of an undeclared source that ignore that source's clock.

    Shifting a timestamp only removes rows from code that filters on it. Code
    that joins a table and takes a column off it never consults the clock at
    all, so `delay` leaves it untouched and the dependency stays invisible.
    Permuting the payload breaks it, and a feature that does not read the source
    cannot notice.
    """
    already = {(f.feature, f.source) for f in findings if f.kind == LEAK}

    for source in sources:
        if timestamps[source] is None:
            continue
        untested = [f for f in features if source not in declared.get(f, ())
                    and (f, source) not in already]
        if not untested:
            continue
        moved_frame = _as_frame(
            compute({**sources, source: shuffle(sources[source], timestamps[source])}),
            "compute",
        )
        if list(moved_frame.columns) != list(baseline.columns):
            notes.append(
                f"note: shuffling {source!r} changed the feature set itself; its "
                f"undeclared dependencies were not probed."
            )
            continue
        for column, feature in zip(baseline.columns, features):
            if feature not in untested:
                continue
            moved, worst = _changed(baseline[column], moved_frame[column], tolerance)
            if moved:
                findings.append(
                    Finding(feature=feature, source=source, kind=LEAK, declared=False,
                            moved=True, max_abs_change=worst)
                )


def _probe_cutoff(compute, sources, timestamps, baseline, features, tolerance,
                  cutoff, findings, notes) -> None:
    """Catch features built from rows that postdate the cutoff.

    A feature computed as of `cutoff` cannot notice the deletion of rows after
    it. Anything that moves was reading them. This is what `delay` cannot see:
    shifting timestamps moves the visible slice for correct and incorrect code
    alike, so a denominator averaged over all of time responds identically to a
    properly filtered one.
    """
    for source in sources:
        column = timestamps[source]
        if column is None:
            continue
        cut = truncate(sources[source], column, cutoff)
        if len(cut) == len(sources[source]):
            continue
        moved_frame = _as_frame(compute({**sources, source: cut}), "compute")
        if list(moved_frame.columns) != list(baseline.columns):
            notes.append(
                f"note: truncating {source!r} at the cutoff changed the feature set "
                f"itself, so future-row dependence on it was not tested."
            )
            continue
        shared = baseline.index.intersection(moved_frame.index)
        if not len(shared):
            continue
        for column_name, feature in zip(baseline.columns, features):
            moved, worst = _changed(
                baseline.loc[shared, column_name], moved_frame.loc[shared, column_name], tolerance
            )
            if moved:
                findings.append(
                    Finding(feature=feature, source=source, kind=LEAK, declared=False,
                            moved=True, max_abs_change=worst)
                )
