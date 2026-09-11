"""Two silent leaks in a customer feature set, on a real public dataset.

    pip install leakprobe openpyxl
    python examples/online_retail.py

Data: the UCI Online Retail set -- 541,909 real transactions from a UK gift
retailer, December 2010 to December 2011, 4,372 customers, 10,624 of the rows
returns rather than sales. Downloaded once and cached beside this file.

The task is the ordinary one: build per-customer features as of a cutoff, to
predict something after it. Two of the six features are wrong. Neither raises,
neither produces an implausible number, and both make the feature set look
richer than it is. They are the two shapes temporal leakage actually takes:

  avg_unit_price   computes over every row it can see, ignoring the cutoff
  net_spend        quietly reads a second source it never declared

leakprobe finds both without being told which is correct, by perturbing each
source's clock and checking that every feature responds the way its declaration
says it must.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

import leakprobe as lp

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
CACHE = Path(__file__).resolve().parent / "online_retail.csv"
CUTOFF = pd.Timestamp("2011-09-01")


def load() -> pd.DataFrame:
    if not CACHE.exists():
        print(f"downloading {URL} (~23 MB, once) ...", flush=True)
        with urllib.request.urlopen(URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        frame = pd.read_excel(io.BytesIO(archive.read(archive.namelist()[0])))
        frame.to_csv(CACHE, index=False)
    frame = pd.read_csv(CACHE, parse_dates=["InvoiceDate"])
    return frame[frame.CustomerID.notna()]


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders, returns = sources["orders"], sources["returns"]
    visible_orders = orders[orders.InvoiceDate <= CUTOFF]
    visible_returns = returns[returns.InvoiceDate <= CUTOFF]
    index = pd.Index(sorted(orders.CustomerID.unique()), name="CustomerID")

    spend = visible_orders.assign(v=visible_orders.Quantity * visible_orders.UnitPrice)
    total_spend = spend.groupby("CustomerID").v.sum()
    order_count = visible_orders.groupby("CustomerID").InvoiceNo.nunique()
    last_order = visible_orders.groupby("CustomerID").InvoiceDate.max()
    return_count = visible_returns.groupby("CustomerID").InvoiceNo.nunique()

    # BUG 1. The cutoff filter is missing. This averages every row in the
    # source, including invoices that had not happened yet at CUTOFF. It is one
    # dropped subscript, it raises nothing, and the column looks entirely normal.
    avg_unit_price = orders.groupby("CustomerID").UnitPrice.mean()

    # BUG 2. Declared against `orders` only -- see `declared` below -- but it
    # reaches into `returns` as well. The dependency is real and undeclared, so
    # nothing downstream knows this column inherits the returns table's latency.
    refunds = visible_returns.assign(v=visible_returns.Quantity * visible_returns.UnitPrice)
    net_spend = total_spend.add(refunds.groupby("CustomerID").v.sum(), fill_value=0)

    return pd.DataFrame({
        "total_spend": total_spend.reindex(index).fillna(0.0),
        "order_count": order_count.reindex(index).fillna(0).astype(float),
        "recency_days": (CUTOFF - last_order.reindex(index)).dt.days.fillna(-1).astype(float),
        "return_count": return_count.reindex(index).fillna(0).astype(float),
        "avg_unit_price": avg_unit_price.reindex(index).fillna(0.0),
        "net_spend": net_spend.reindex(index).fillna(0.0),
    }, index=index)


def main() -> None:
    frame = load()
    orders = frame[frame.Quantity > 0].reset_index(drop=True)
    returns = frame[frame.Quantity < 0].reset_index(drop=True)
    print(f"{len(orders):,} orders and {len(returns):,} returns "
          f"for {frame.CustomerID.nunique():,} customers, cutoff {CUTOFF.date()}\n")

    report = lp.check(
        compute=build_features,
        sources={"orders": orders, "returns": returns},
        timestamps={"orders": "InvoiceDate", "returns": "InvoiceDate"},
        declared={
            "total_spend":    ["orders"],
            "order_count":    ["orders"],
            "recency_days":   ["orders"],
            "return_count":   ["returns"],
            "avg_unit_price": ["orders"],
            "net_spend":      ["orders"],
        },
    )
    print(report)
    print("\nexpected: avg_unit_price ignores the clock, net_spend reads returns undeclared")


if __name__ == "__main__":
    main()
