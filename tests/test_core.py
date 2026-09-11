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


def test_declaring_nothing_flags_every_dependency(sources):
    """Declaring nothing flags everything that reads anything.

    `momentum` used to be absent here, and was documented as the method's
    boundary: it reads price *values* and never asks when they arrived, so
    shifting availability uniformly cannot move it. The undeclared probe closes
    that gap by permuting an undeclared source's payload instead of its clock,
    which breaks the row associations `momentum` depends on. What survives of
    the boundary is narrower and is asserted in
    test_marginal_only_reads_are_invisible."""
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={},
    )
    assert {f.feature for f in report.leaks} == {"momentum", "earnings_yield", "turnover"}


def test_undeclared_probe_can_be_switched_off(sources):
    """The clock-only behaviour, for anyone who wants exactly the old semantics."""
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={}, probe_undeclared=False,
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


def test_marginal_only_reads_are_invisible(sources):
    """The remaining limitation, asserted so it cannot quietly stop being true.

    Two perturbations run: the clock moves, and an undeclared source's payload is
    permuted. Permutation is column-wise, so a column's *marginal* survives it --
    a mean, a sum, a max, a quantile is identical before and after. A feature
    that reads only such an aggregate of a source, never joining on a key and
    never consulting a timestamp, is invariant to both and will not be flagged.

    This is much narrower than the original boundary, which missed any read that
    ignored the clock. Look-ahead is a dependency on timing or on row identity,
    and both are now covered; a global constant taken off an undeclared table is
    what is left.
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


def test_undeclared_probe_catches_a_clock_ignoring_join():
    """The shape `delay` is blind to: an outcome joined off an undeclared table.

    Found on Chicago crime data, where an arrest flag was joined from a
    disposition table. Shifting that table's clock moves nothing -- the flag is
    the same value on the same row -- so the dependency was invisible until the
    payload itself was permuted.
    """
    events = pd.DataFrame({
        "id": range(60),
        "at": pd.date_range("2024-01-01", periods=60, freq="D"),
        "region": [f"r{i % 4}" for i in range(60)],
    })
    outcomes = pd.DataFrame({
        "id": range(60),
        "resolved_at": pd.date_range("2024-03-01", periods=60, freq="D"),
        "was_upheld": [i % 3 == 0 for i in range(60)],
    })

    def build_it(src):
        joined = src["events"].merge(src["outcomes"][["id", "was_upheld"]], on="id", how="left")
        # Grouped, not a plain mean. A column-wise permutation preserves the
        # global rate but destroys which region each outcome belongs to, and it
        # is that association the feature depends on.
        by_region = joined.groupby("region").was_upheld.mean()
        return pd.DataFrame({
            "event_count": [float(len(src["events"]))] * len(by_region),
            "upheld_rate": by_region.to_numpy(dtype=float),
        }, index=by_region.index)

    shared = dict(
        sources={"events": events, "outcomes": outcomes},
        timestamps={"events": "at", "outcomes": "resolved_at"},
        declared={"event_count": ["events"], "upheld_rate": ["events"]},
    )
    blind = lc.check(build_it, probe_undeclared=False, **shared)
    assert blind.leaks == [], "delay alone cannot see this, which is why the probe exists"

    report = lc.check(build_it, **shared)
    assert [(f.feature, f.source) for f in report.leaks] == [("upheld_rate", "outcomes")]


def test_cutoff_probe_catches_a_statistic_over_all_of_time():
    """A z-score denominator computed over every row, including the future.

    Invisible to both other probes: it reads only its declared source, and it
    does move when that source's clock moves, exactly as a correct feature does.
    Only deleting the post-cutoff rows separates them.
    """
    cutoff = pd.Timestamp("2024-02-01")
    events = pd.DataFrame({
        "occurred_at": pd.date_range("2024-01-01", periods=90, freq="D"),
        "amount": [float(i) for i in range(90)],
    })

    def correct(src):
        visible = src["events"][src["events"]["occurred_at"] <= cutoff]
        return pd.DataFrame({"mean_amount": [visible.amount.mean()]})

    def leaky(src):
        visible = src["events"][src["events"]["occurred_at"] <= cutoff]
        everything = src["events"].amount
        return pd.DataFrame({"mean_amount": [(visible.amount.mean() - everything.mean())
                                             / everything.std()]})

    shared = dict(sources={"events": events}, timestamps={"events": "occurred_at"},
                  declared={"mean_amount": ["events"]})
    assert lc.check(correct, cutoff=cutoff, **shared).clean
    assert lc.check(leaky, **shared).clean, "without a cutoff this is genuinely undetectable"
    assert [f.feature for f in lc.check(leaky, cutoff=cutoff, **shared).leaks] == ["mean_amount"]
