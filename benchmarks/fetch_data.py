"""Download the five public datasets the leak zoo runs against.

    python benchmarks/fetch_data.py

Every source is public and needs no account. About 300 MB total, cached under
benchmarks/data/ and skipped on re-run. The retail set arrives as an Excel file
and is converted once; the others are used as downloaded.
"""
from __future__ import annotations

import io
import sys
import ssl
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

SOURCES = {
    "online_retail.csv": ("https://archive.ics.uci.edu/static/public/352/online+retail.zip", "retail"),
    "taxi.parquet":      ("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet", "raw"),
    "crimes.csv":        ("https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit=300000&$where=year=2023", "raw"),
    "nyc311.csv":        ("https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=200000"
                          "&$where=created_date%20between%20%272023-01-01%27%20and%20%272023-06-30%27", "raw"),
    "hour.csv":          ("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", "bike"),
}


def _ssl_context():
    # python.org builds on macOS ship without a certificate bundle, so a stock
    # install fails every https download. certifi (pulled in by pip and most
    # environments) has one. Fall back to the system default if it is missing.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "leakprobe-benchmark"})
    with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as r:
        return r.read()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (url, kind) in SOURCES.items():
        out = DATA / name
        if out.exists():
            print(f"  have  {name}")
            continue
        print(f"  fetch {name} ...", flush=True)
        raw = fetch(url)
        if kind == "retail":
            import pandas as pd
            z = zipfile.ZipFile(io.BytesIO(raw))
            pd.read_excel(io.BytesIO(z.read(z.namelist()[0]))).to_csv(out, index=False)
        elif kind == "bike":
            z = zipfile.ZipFile(io.BytesIO(raw))
            out.write_bytes(z.read("hour.csv"))
        else:
            out.write_bytes(raw)
        print(f"        {out.stat().st_size / 1e6:.0f} MB")
    print(f"\nready: {DATA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # a stranger should get one line, not a stack
        sys.exit(f"download failed: {exc}\nRe-run to resume; completed files are kept.")
