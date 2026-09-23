"""Does leakprobe catch real leaks, and does it stay quiet on correct code?

    python3 benchmarks/leak_zoo.py

Five public datasets, five shapes of temporal leakage, and a correct pipeline
for every dataset. The correct pipelines matter more than the bugged ones: a
detector that flags clean code is worse than no detector, because it trains you
to ignore it.

Shapes under test:

  S1  missing cutoff filter     aggregate computed over every row, not just the visible ones
  S2  undeclared source read    feature quietly reads a second table it never declared
  S3  outcome leakage           feature reads a field only knowable after the event,
                                where that field lives in a table with its own later clock
  S4  same-clock outcome        the outcome field sits in the same table under the same
                                timestamp as the event -- expected MISS, see the report
  S5  global normalisation      z-score against statistics computed over all of time

Datasets: UCI Online Retail, NYC yellow taxi, Chicago crimes, NYC 311, UCI bike
sharing. Downloaded separately; see DATA below.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import leakprobe as lp

DATA = Path(__file__).resolve().parent / "data"   # populated by fetch_data.py
CUTOFF = {}


@dataclass
class Case:
    dataset: str
    name: str
    shape: str | None          # None = correct pipeline, must produce no findings
    build: Callable
    sources: dict
    timestamps: dict
    declared: dict
    expect: str | None         # feature expected to be flagged


# ----------------------------------------------------------------- retail ---
def retail_sources():
    frame = pd.read_csv(DATA / "online_retail.csv", parse_dates=["InvoiceDate"])
    frame = frame[frame.CustomerID.notna()]
    return {"orders": frame[frame.Quantity > 0].reset_index(drop=True),
            "returns": frame[frame.Quantity < 0].reset_index(drop=True)}


def _retail_base(sources, cutoff):
    orders, returns = sources["orders"], sources["returns"]
    vo = orders[orders.InvoiceDate <= cutoff]
    vr = returns[returns.InvoiceDate <= cutoff]
    index = pd.Index(sorted(orders.CustomerID.unique()), name="CustomerID")
    spend = vo.assign(v=vo.Quantity * vo.UnitPrice).groupby("CustomerID").v.sum()
    return orders, returns, vo, vr, index, spend


def retail_correct(sources):
    cutoff = CUTOFF["retail"]
    orders, returns, vo, vr, index, spend = _retail_base(sources, cutoff)
    return pd.DataFrame({
        "total_spend": spend.reindex(index).fillna(0.0),
        "order_count": vo.groupby("CustomerID").InvoiceNo.nunique().reindex(index).fillna(0).astype(float),
        "avg_unit_price": vo.groupby("CustomerID").UnitPrice.mean().reindex(index).fillna(0.0),
        "return_count": vr.groupby("CustomerID").InvoiceNo.nunique().reindex(index).fillna(0).astype(float),
    }, index=index)


def retail_s1(sources):
    cutoff = CUTOFF["retail"]
    orders, returns, vo, vr, index, spend = _retail_base(sources, cutoff)
    out = retail_correct(sources)
    out["avg_unit_price"] = orders.groupby("CustomerID").UnitPrice.mean().reindex(index).fillna(0.0)
    return out


def retail_s2(sources):
    cutoff = CUTOFF["retail"]
    orders, returns, vo, vr, index, spend = _retail_base(sources, cutoff)
    out = retail_correct(sources)
    refunds = vr.assign(v=vr.Quantity * vr.UnitPrice).groupby("CustomerID").v.sum()
    out["total_spend"] = spend.add(refunds, fill_value=0).reindex(index).fillna(0.0)
    return out


def retail_s5(sources):
    cutoff = CUTOFF["retail"]
    orders, returns, vo, vr, index, spend = _retail_base(sources, cutoff)
    out = retail_correct(sources)
    allspend = orders.assign(v=orders.Quantity * orders.UnitPrice).groupby("CustomerID").v.sum()
    out["total_spend"] = ((spend.reindex(index).fillna(0.0) - allspend.mean()) / allspend.std())
    return out


RETAIL_DECL = {"total_spend": ["orders"], "order_count": ["orders"],
               "avg_unit_price": ["orders"], "return_count": ["returns"]}


# ------------------------------------------------------------------ taxi ---
def taxi_sources():
    frame = pd.read_parquet(DATA / "taxi.parquet", columns=[
        "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
        "fare_amount", "tip_amount", "trip_distance", "payment_type"])
    frame = frame[(frame.tpep_pickup_datetime >= "2024-01-01") &
                  (frame.tpep_pickup_datetime < "2024-02-01")]
    return {"trips": frame[frame.payment_type != 1].reset_index(drop=True),
            "card_trips": frame[frame.payment_type == 1].reset_index(drop=True)}


def _taxi_base(sources, cutoff):
    trips, card = sources["trips"], sources["card_trips"]
    vt = trips[trips.tpep_pickup_datetime <= cutoff]
    vc = card[card.tpep_pickup_datetime <= cutoff]
    index = pd.Index(sorted(trips.PULocationID.unique()), name="PULocationID")
    return trips, card, vt, vc, index


def taxi_correct(sources):
    cutoff = CUTOFF["taxi"]
    trips, card, vt, vc, index = _taxi_base(sources, cutoff)
    return pd.DataFrame({
        "trip_count": vt.groupby("PULocationID").size().reindex(index).fillna(0).astype(float),
        "avg_fare": vt.groupby("PULocationID").fare_amount.mean().reindex(index).fillna(0.0),
        "avg_distance": vt.groupby("PULocationID").trip_distance.mean().reindex(index).fillna(0.0),
        "card_share": (vc.groupby("PULocationID").size().reindex(index).fillna(0) /
                       vt.groupby("PULocationID").size().reindex(index).fillna(0).clip(lower=1)),
    }, index=index)


def taxi_s1(sources):
    cutoff = CUTOFF["taxi"]
    trips, card, vt, vc, index = _taxi_base(sources, cutoff)
    out = taxi_correct(sources)
    out["avg_fare"] = trips.groupby("PULocationID").fare_amount.mean().reindex(index).fillna(0.0)
    return out


def taxi_s2(sources):
    cutoff = CUTOFF["taxi"]
    trips, card, vt, vc, index = _taxi_base(sources, cutoff)
    out = taxi_correct(sources)
    out["avg_distance"] = pd.concat([vt, vc]).groupby("PULocationID").trip_distance.mean(
        ).reindex(index).fillna(0.0)
    return out


TAXI_DECL = {"trip_count": ["trips"], "avg_fare": ["trips"],
             "avg_distance": ["trips"], "card_share": ["trips", "card_trips"]}


# ---------------------------------------------------------------- crimes ---
def crime_sources():
    frame = pd.read_csv(DATA / "crimes.csv", usecols=[
        "id", "date", "primary_type", "district", "arrest", "updated_on"],
        parse_dates=["date", "updated_on"])
    frame = frame[frame.district.notna()]
    reports = frame[["id", "date", "primary_type", "district"]].reset_index(drop=True)
    # The disposition is knowable only later, and carries its own clock. This is
    # the honest shape of outcome leakage: a separate table with a later stamp.
    dispositions = frame[["id", "arrest", "updated_on"]].reset_index(drop=True)
    return {"reports": reports, "dispositions": dispositions}


def _crime_base(sources, cutoff):
    reports, disp = sources["reports"], sources["dispositions"]
    vr = reports[reports.date <= cutoff]
    vd = disp[disp.updated_on <= cutoff]
    index = pd.Index(sorted(reports.district.unique()), name="district")
    return reports, disp, vr, vd, index


def crime_correct(sources):
    cutoff = CUTOFF["crimes"]
    reports, disp, vr, vd, index = _crime_base(sources, cutoff)
    return pd.DataFrame({
        "report_count": vr.groupby("district").size().reindex(index).fillna(0).astype(float),
        "theft_share": (vr[vr.primary_type == "THEFT"].groupby("district").size()
                        .reindex(index).fillna(0) /
                        vr.groupby("district").size().reindex(index).fillna(0).clip(lower=1)),
    }, index=index)


def crime_s3(sources):
    """Outcome leakage: arrest rate built from dispositions, declared as reports only."""
    cutoff = CUTOFF["crimes"]
    reports, disp, vr, vd, index = _crime_base(sources, cutoff)
    out = crime_correct(sources)
    joined = vr.merge(disp, on="id", how="left")
    out["arrest_rate"] = joined.groupby("district").arrest.mean().reindex(index).fillna(0.0)
    return out


def crime_s4(sources):
    """Same-clock outcome: arrest read through the reports table's own timestamp.

    Expected MISS. Perturbing the reports clock moves this feature exactly as it
    moves a legitimate one, so no invariant is violated and nothing distinguishes
    it. Recorded because the boundary is the point.
    """
    cutoff = CUTOFF["crimes"]
    reports, disp, vr, vd, index = _crime_base(sources, cutoff)
    out = crime_correct(sources)
    joined = vr.merge(disp[["id", "arrest"]], on="id", how="left")
    out["arrest_rate_sameclock"] = joined.groupby("district").arrest.mean().reindex(index).fillna(0.0)
    return out


CRIME_DECL = {"report_count": ["reports"], "theft_share": ["reports"],
              "arrest_rate": ["reports"], "arrest_rate_sameclock": ["reports"]}


# ------------------------------------------------------------------- 311 ---
def nyc311_sources():
    frame = pd.read_csv(DATA / "nyc311.csv", usecols=[
        "unique_key", "created_date", "closed_date", "complaint_type", "agency"],
        parse_dates=["created_date", "closed_date"])
    frame = frame[frame.complaint_type.notna()]
    requests = frame[["unique_key", "created_date", "complaint_type", "agency"]].reset_index(drop=True)
    closures = frame[frame.closed_date.notna()][["unique_key", "closed_date"]].reset_index(drop=True)
    return {"requests": requests, "closures": closures}


def _n311_base(sources, cutoff):
    req, clo = sources["requests"], sources["closures"]
    vq = req[req.created_date <= cutoff]
    vc = clo[clo.closed_date <= cutoff]
    index = pd.Index(sorted(req.complaint_type.unique()), name="complaint_type")
    return req, clo, vq, vc, index


def n311_correct(sources):
    cutoff = CUTOFF["nyc311"]
    req, clo, vq, vc, index = _n311_base(sources, cutoff)
    return pd.DataFrame({
        "request_count": vq.groupby("complaint_type").size().reindex(index).fillna(0).astype(float),
        "agency_count": vq.groupby("complaint_type").agency.nunique().reindex(index).fillna(0).astype(float),
    }, index=index)


def n311_s3(sources):
    """Resolution time needs the closure table, but declares only requests."""
    cutoff = CUTOFF["nyc311"]
    req, clo, vq, vc, index = _n311_base(sources, cutoff)
    out = n311_correct(sources)
    joined = vq.merge(clo, on="unique_key", how="inner")
    hours = (joined.closed_date - joined.created_date).dt.total_seconds() / 3600
    out["avg_resolution_hours"] = joined.assign(h=hours).groupby(
        "complaint_type").h.mean().reindex(index).fillna(0.0)
    return out


N311_DECL = {"request_count": ["requests"], "agency_count": ["requests"],
             "avg_resolution_hours": ["requests"]}


# ------------------------------------------------------------------ bike ---
def bike_sources():
    frame = pd.read_csv(DATA / "hour.csv", parse_dates=["dteday"])
    frame["ts"] = frame.dteday + pd.to_timedelta(frame.hr, unit="h")
    return {"hourly": frame[["ts", "cnt", "temp", "hum", "weathersit"]].reset_index(drop=True)}


def bike_correct(sources):
    cutoff = CUTOFF["bike"]
    hourly = sources["hourly"]
    visible = hourly[hourly.ts <= cutoff].sort_values("ts")
    index = pd.Index(range(24), name="hour_of_day")
    by_hour = visible.assign(h=visible.ts.dt.hour).groupby("h")
    return pd.DataFrame({
        "mean_count": by_hour.cnt.mean().reindex(index).fillna(0.0),
        "mean_temp": by_hour.temp.mean().reindex(index).fillna(0.0),
    }, index=index)


def bike_s1(sources):
    cutoff = CUTOFF["bike"]
    hourly = sources["hourly"]
    out = bike_correct(sources)
    all_by_hour = hourly.assign(h=hourly.ts.dt.hour).groupby("h")
    out["mean_temp"] = all_by_hour.temp.mean().reindex(pd.Index(range(24))).fillna(0.0)
    return out


BIKE_DECL = {"mean_count": ["hourly"], "mean_temp": ["hourly"]}


# ----------------------------------------------------------------- runner ---
def cases() -> list[Case]:
    r, t, c, n, b = (retail_sources(), taxi_sources(), crime_sources(),
                     nyc311_sources(), bike_sources())
    CUTOFF.update({"retail": pd.Timestamp("2011-09-01"), "taxi": pd.Timestamp("2024-01-20"),
                   "crimes": pd.Timestamp("2023-09-01"), "nyc311": pd.Timestamp("2023-06-20"),
                   "bike": pd.Timestamp("2012-09-01")})
    rt = {"orders": "InvoiceDate", "returns": "InvoiceDate"}
    tt = {"trips": "tpep_pickup_datetime", "card_trips": "tpep_pickup_datetime"}
    ct = {"reports": "date", "dispositions": "updated_on"}
    nt = {"requests": "created_date", "closures": "closed_date"}
    bt = {"hourly": "ts"}
    return [
        Case("retail", "correct", None, retail_correct, r, rt, RETAIL_DECL, None),
        Case("retail", "S1 missing cutoff", "S1", retail_s1, r, rt, RETAIL_DECL, "avg_unit_price"),
        Case("retail", "S2 undeclared read", "S2", retail_s2, r, rt, RETAIL_DECL, "total_spend"),
        Case("retail", "S5 global normalisation", "S5", retail_s5, r, rt, RETAIL_DECL, "total_spend"),
        Case("taxi", "correct", None, taxi_correct, t, tt, TAXI_DECL, None),
        Case("taxi", "S1 missing cutoff", "S1", taxi_s1, t, tt, TAXI_DECL, "avg_fare"),
        Case("taxi", "S2 undeclared read", "S2", taxi_s2, t, tt, TAXI_DECL, "avg_distance"),
        Case("crimes", "correct", None, crime_correct, c, ct, CRIME_DECL, None),
        Case("crimes", "S3 outcome leakage", "S3", crime_s3, c, ct, CRIME_DECL, "arrest_rate"),
        Case("crimes", "S4 same-clock outcome", "S4", crime_s4, c, ct, CRIME_DECL,
             "arrest_rate_sameclock"),
        Case("nyc311", "correct", None, n311_correct, n, nt, N311_DECL, None),
        Case("nyc311", "S3 outcome leakage", "S3", n311_s3, n, nt, N311_DECL, "avg_resolution_hours"),
        Case("bike", "correct", None, bike_correct, b, bt, BIKE_DECL, None),
        Case("bike", "S1 missing cutoff", "S1", bike_s1, b, bt, BIKE_DECL, "mean_temp"),
    ]


def main() -> None:
    rows = []
    for case in cases():
        try:
            # One declaration map per dataset; each pipeline returns a subset of
            # its features. leakprobe refuses a declaration for a feature the
            # pipeline never produced (a misspelling would otherwise pass as
            # clean), so restrict the map to what this build actually returns.
            produced = set(case.build(case.sources).columns)
            declared = {k: v for k, v in case.declared.items() if k in produced}
            report = lp.check(compute=case.build, sources=case.sources,
                              timestamps=case.timestamps, declared=declared,
                              by=pd.Timedelta(days=21), cutoff=CUTOFF[case.dataset])
        except Exception as exc:
            rows.append({"dataset": case.dataset, "case": case.name, "shape": case.shape or "-",
                         "flagged": f"ERROR {type(exc).__name__}", "verdict": "ERROR"})
            continue

        leaked = {f.feature for f in report.leaks}
        future = {f.feature for f in report.future}   # declared, read past the cutoff
        bypass = {f.feature for f in report.bypassed}
        flagged = leaked | future | bypass
        if case.shape is None:
            verdict = "clean" if not flagged else "FALSE POSITIVE"
        else:
            verdict = "caught" if case.expect in flagged else "MISSED"
        rows.append({"dataset": case.dataset, "case": case.name, "shape": case.shape or "-",
                     "flagged": ", ".join(sorted(flagged)) or "(nothing)", "verdict": verdict})

    frame = pd.DataFrame(rows)
    width = max(len(r["case"]) for r in rows)
    print(f"\n{'dataset':<9} {'case':<{width}} {'verdict':<15} flagged")
    for r in rows:
        print(f"{r['dataset']:<9} {r['case']:<{width}} {r['verdict']:<15} {r['flagged']}")

    correct = frame[frame["shape"] == "-"]
    bugged = frame[frame["shape"] != "-"]
    print(f"\ncorrect pipelines : {(correct.verdict == 'clean').sum()}/{len(correct)} clean "
          f"({(correct.verdict == 'FALSE POSITIVE').sum()} false positives)")
    print(f"planted leaks     : {(bugged.verdict == 'caught').sum()}/{len(bugged)} caught")
    for shape, g in bugged.groupby("shape"):
        print(f"  {shape}: {(g.verdict=='caught').sum()}/{len(g)}")


if __name__ == "__main__":
    main()
