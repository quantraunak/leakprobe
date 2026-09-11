"""The claims the tool rests on, each broken on purpose."""

import numpy as np
import pandas as pd
import pytest

import leakprobe as lc


@pytest.fixture
def sources():
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "available_at": days,
            "price": 100 + rng.normal(0, 1, len(days)).cumsum(),
            "volume": rng.integers(1_000, 5_000, len(days)),
        }
    )
    quarters = pd.to_datetime(["2023-12-31", "2024-03-31"])
    filings = pd.DataFrame(
        {
            "period_end": quarters,
            "available_at": quarters + pd.Timedelta(days=34),
            "earnings": [4.0, 5.0],
            "shares": [1_000.0, 1_100.0],
        }
    )
    return {"prices": prices, "filings": filings}


def _as_of(filings, dates, column):
    """Latest value knowable on each date. The whole point is that this respects
    `available_at`, so moving it moves the answer."""
    ordered = filings.sort_values("available_at")
    idx = np.searchsorted(ordered["available_at"].to_numpy(), dates.to_numpy(), side="right") - 1
    out = np.where(idx >= 0, ordered[column].to_numpy()[np.clip(idx, 0, None)], np.nan)
    return pd.Series(out, index=range(len(dates)))


def build(sources):
    prices, filings = sources["prices"], sources["filings"]
    dates = prices["available_at"]
    return pd.DataFrame(
        {
            "momentum": prices["price"].pct_change(20),
            "earnings_yield": _as_of(filings, dates, "earnings") / prices["price"],
            # Looks like a price feature. Divides by a filed number.
            "turnover": prices["volume"] / _as_of(filings, dates, "shares"),
        }
    )


def test_price_only_feature_is_exactly_unchanged(sources):
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices", "filings"]},
    )
    hit = next(f for f in report.findings if f.feature == "momentum" and f.source == "filings")
    assert not hit.moved
    assert hit.max_abs_change == 0.0, "not approximately zero -- exactly zero"
    assert report.clean


def test_undeclared_filing_dependency_is_caught(sources):
    """turnover divides volume by shares outstanding, so it reads a filing.
    Declared as price-only, which is the mistake this tool exists to catch."""
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices"]},
    )
    assert [f.feature for f in report.leaks] == ["turnover"]
    assert report.leaks[0].source == "filings"
    assert not report.clean
    with pytest.raises(AssertionError, match="turnover"):
        report.raise_for_leaks()


def test_declaring_nothing_flags_every_timing_dependency(sources):
    """Everything that consults a timestamp is caught. `momentum` is not, and
    that is the method's boundary rather than a miss: it reads price *values*
    and never asks when they arrived, so shifting availability uniformly cannot
    move it. See test_values_read_without_consulting_time_are_invisible."""
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={},
    )
    assert {f.feature for f in report.leaks} == {"earnings_yield", "turnover"}


def test_declared_but_clock_insensitive_is_reported_separately(sources):
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices", "filings"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices", "filings"]},
    )
    assert set((f.feature, f.source) for f in report.bypassed) == {
        ("momentum", "filings"),
        ("momentum", "prices"),
    }
    assert report.clean, "a stale declaration is not a leak"


def test_nondeterministic_compute_is_refused(sources):
    def unstable(src):
        return pd.DataFrame({"x": np.random.default_rng().normal(size=len(src["prices"]))})

    with pytest.raises(RuntimeError, match="not deterministic"):
        lc.check(unstable, sources,
                 timestamps={"prices": "available_at", "filings": "available_at"},
                 declared={})


def test_unknown_source_in_declared_is_refused(sources):
    with pytest.raises(KeyError, match="unknown source"):
        lc.check(build, sources,
                 timestamps={"prices": "available_at", "filings": "available_at"},
                 declared={"momentum": ["typo_source"]})


def test_missing_timestamp_column_is_refused(sources):
    with pytest.raises(KeyError, match="availability column"):
        lc.check(build, sources, timestamps={"prices": "available_at"}, declared={})


def test_nan_in_the_same_place_is_not_a_change(sources):
    """A feature that is NaN for its first 20 rows in both runs has not moved."""
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices", "filings"]},
    )
    assert build(sources)["momentum"].isna().sum() > 0
    assert report.clean


def test_values_read_without_consulting_time_are_invisible(sources):
    """The limitation, asserted so it cannot quietly stop being true.

    The perturbation moves availability, so it can only reveal dependencies that
    flow through a timestamp -- an as-of join, a merge on a date, a window
    anchored to one. A feature that reads a source's values directly, with no
    reference to when they arrived, is invariant to it and will not be flagged.

    That boundary is not much of a loss in practice: look-ahead is by definition
    a dependency on timing, so the leaks worth catching are the detectable ones.
    """
    def timeless(src):
        # Reads filings values, never their availability. Undetectable here.
        return pd.DataFrame({"mean_earnings": [src["filings"]["earnings"].mean()]})

    report = lc.check(
        timeless, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={},
    )
    assert report.clean
    assert report.leaks == []


def test_source_without_a_clock_is_skipped_and_said_so(sources):
    static = pd.DataFrame({"region": ["us", "eu"]})
    report = lc.check(
        lambda src: pd.DataFrame({"x": [len(src["static"])]}),
        {**sources, "static": static},
        timestamps={"prices": "available_at", "filings": "available_at", "static": None},
        declared={},
    )
    assert any("no availability column" in n for n in report.notes)
    assert all(f.source != "static" for f in report.findings)
