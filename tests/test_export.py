"""Structured export, the delay guard, and submodule completeness."""

import json

import numpy as np
import pandas as pd
import pytest

import leakprobe as lc
from leakprobe import Finding
from leakprobe.perturb import __all__ as perturb_all


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
            "turnover": prices["volume"] / _as_of(filings, dates, "shares"),
        }
    )


def test_finding_to_dict_round_trips_every_field():
    f = Finding(feature="x", source="s", kind="leak", declared=False,
                moved=True, max_abs_change=7.5)
    d = f.to_dict()
    assert d == {
        "feature": "x", "source": "s", "kind": "leak", "declared": False,
        "moved": True, "max_abs_change": 7.5,
    }
    assert set(d) == {"feature", "source", "kind", "declared", "moved", "max_abs_change"}


def test_report_to_dict_is_json_serialisable(sources):
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices"]},
    )
    d = report.to_dict()
    json.dumps(d)  # must not raise
    assert d["features"] == ["momentum", "earnings_yield", "turnover"]
    assert d["sources"] == ["prices", "filings"]
    assert d["clean"] is False
    # the leak shows up in both the full list and the leaks bucket
    assert [f["feature"] for f in d["leaks"]] == ["turnover"]
    assert any(f["feature"] == "turnover" and f["source"] == "filings"
               for f in d["findings"])


def test_report_to_json_parses_back(sources):
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices"]},
    )
    parsed = json.loads(report.to_json())
    assert parsed == report.to_dict()
    assert len(json.loads(report.to_json(indent=None))["findings"]) == 6


def test_as_frame_is_a_tidy_dataframe(sources):
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices"]},
    )
    df = report.as_frame()
    assert list(df.columns) == ["feature", "source", "kind", "declared", "moved",
                                 "max_abs_change"]
    assert len(df) == 6  # 3 features x 2 sources
    leak = df[(df["feature"] == "turnover") & (df["source"] == "filings")].iloc[0]
    assert leak["kind"] == "leak"
    assert not leak["declared"]
    assert leak["moved"]
    pd.testing.assert_frame_equal(df, report.as_frame())  # stable order


def test_clean_report_exports_empties(sources):
    report = lc.check(
        build, sources,
        timestamps={"prices": "available_at", "filings": "available_at"},
        declared={"momentum": ["prices"], "earnings_yield": ["filings", "prices"],
                  "turnover": ["prices", "filings"]},
    )
    d = report.to_dict()
    assert d["clean"] is True
    assert d["leaks"] == []
    assert d["future"] == []
    # a clean report is not empty -- it still describes every pair it checked;
    # it is clean because none of those findings is a leak or a future read.
    kinds = set(report.as_frame()["kind"])
    assert kinds <= {"ok", "bypass"}


def test_delay_rejects_non_positive_by():
    frame = pd.DataFrame({"available_at": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    for bad in (pd.Timedelta(0), pd.Timedelta(-1), pd.Timedelta(days=-5)):
        with pytest.raises(ValueError, match="positive"):
            lc.delay(frame, "available_at", bad)


def test_delay_accepts_positive_by():
    frame = pd.DataFrame({"available_at": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    out = lc.delay(frame, "available_at", pd.Timedelta(days=1))
    assert (out["available_at"] - frame["available_at"] == pd.Timedelta(days=1)).all()


def test_submodule_all_lists_every_public_perturbation():
    has_all = set(dir(lc))
    for name in ("delay", "advance", "use_column", "truncate", "shuffle"):
        assert name in perturb_all, f"{name} missing from perturb.__all__"
        assert name in has_all, f"{name} not re-exported from package root"