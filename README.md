# leakprobe

[![CI](https://github.com/quantraunak/leakprobe/actions/workflows/ci.yml/badge.svg)](https://github.com/quantraunak/leakprobe/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/leakprobe.svg)](https://pypi.org/project/leakprobe/) [![Python](https://img.shields.io/pypi/pyversions/leakprobe.svg)](https://pypi.org/project/leakprobe/)


**A detector for temporal leakage that needs no ground truth. On five public datasets with
five planted leak shapes it catches 9 of 9 and flags 0 of 5 correct pipelines. The shipped
version caught 6 of 9; the three misses had one cause and produced two new probes.**

## Objective

Temporal leakage has no failure mode. A feature reads something that was not knowable yet,
nothing raises, no number looks implausible, and your metrics improve. You cannot unit-test
it, because the correct value is the thing in dispute — if you knew what the feature should
have been, you would not have the bug.

## Hypothesis

You do not need the right answer. You need an invariant: **changing when a data source
became available must change the features that read it, and must leave the features that do
not read it bit-identical.** That is a metamorphic relation, the standard move for programs
where no oracle exists ([Chen's oracle problem](https://dl.acm.org/doi/10.1145/3143561)),
and leakage fits it exactly because look-ahead is by construction a dependency on time.

## Result

Five public datasets, five leak shapes, and a correct pipeline for each.

| shape | caught |
|---|---|
| S1 missing cutoff filter | 3/3 |
| S2 undeclared source read | 2/2 |
| S3 outcome leakage from a later-clocked table | 2/2 |
| S4 outcome read under the event's own clock | 1/1 |
| S5 statistic computed over all of time | 1/1 |
| **correct pipelines flagged** | **0/5** |

The last row is the one that matters. A detector that flags clean code teaches you to
ignore it.

**The version on PyPI at 0.1.0 scored 6 of 9.** All three misses shared a cause: `delay`
only removes rows from code that *filters* on a timestamp. Code that joins a table and
takes a column off it never consults the clock, so shifting timestamps leaves it untouched.
Chicago's arrest flag was invisible for exactly that reason — and the test suite already
documented the gap as "the method's boundary rather than a miss."

## Framework proposed

**A perturbation detects a dependency only if it changes what the feature actually reads.**
One perturbation is never enough, so three run:

| probe | what it changes | what it catches |
|---|---|---|
| `delay` | timestamps move later | code that filters on a clock |
| `shuffle` | an undeclared source's payload permuted, clock intact | code that joins a table and reads a column, never consulting its clock |
| `truncate` | rows after your cutoff deleted | a global mean or z-score denominator taken over all of time |

`shuffle` runs only on pairs the user declared independent, so it turns that declaration
into something tested rather than assumed, and costs no false positives. `truncate` is the
only probe that catches an all-of-time statistic, because those respond to a delayed clock
exactly as correct code does.

## Data

| dataset | size | role |
|---|---|---|
| UCI Online Retail | 406,829 transactions with a customer | per-customer spend, returns |
| NYC yellow taxi | 2.9M January 2024 trips | per-zone fares |
| Chicago crime reports | 263,837 from 2023 | per-district arrest rates |
| NYC 311 | 200,000 service requests | resolution times |
| UCI bike sharing | 17,379 hours | hour-of-day aggregates |

All public, no authentication. `benchmarks/leak_zoo.py` runs the whole thing.

## Limitations

Column-wise permutation preserves a column's marginal distribution — a mean is a mean after
shuffling, and so is a max or a quantile. A feature that reads only such an aggregate of an
undeclared table, never joining on a key and never consulting a timestamp, survives every
probe. That boundary is asserted in the test suite so it cannot change unnoticed. `truncate`
requires you to pass your cutoff; without it the all-of-time statistic is genuinely
undetectable.

## Reproduce

```bash
pip install leakprobe
python -m leakprobe.demo           # two seconds, no downloads: one planted leak, caught
```

Then, from a clone of this repository:

```bash
python -m pytest                   # 19 tests
python examples/online_retail.py   # real data, two planted bugs, both caught
python benchmarks/fetch_data.py    # ~300 MB of public data, cached
python benchmarks/leak_zoo.py      # the full 9/9 table above
```

## Value

Point it at a feature pipeline, declare what each feature is supposed to read, and it tells
you what moved that shouldn't have. It found `turnover_1m` in a 22-factor study — a
price-and-volume factor that divides by shares outstanding, so it silently inherited the
filing calendar, in a design whose whole point was that the price block could not. That bug
was worth 59% of mean IC and four spurious t-statistics.

[Full write-up](https://raunaksood.vercel.app/writing/testing-for-leakage).

## Detail

`leakprobe` finds it without needing to know the right answer. It needs one
thing you already know: **a change to your data that your features must be
invariant to.**

```bash
pip install leakprobe
```

```python
import leakprobe as lp

report = lp.check(
    compute=build_features,                  # (sources) -> DataFrame of features
    sources={"events": events, "tickets": tickets},
    timestamps={"events": "occurred_at", "tickets": "resolved_at"},
    declared={
        "total_spend":      ["events"],
        "event_count":      ["events"],
        "tickets_resolved": ["tickets"],
        "avg_severity":     ["tickets"],
    },
)
report.raise_for_leaks()      # fails your test suite if anything leaked
```

```
4 features x 2 sources

feature                 events       tickets
total_spend           reads it     exactly 0
event_count           reads it     exactly 0
tickets_resolved     exactly 0      reads it
avg_severity         exactly 0       bypass?

No undeclared dependencies.

Declared, but did not respond to the source's clock:
  - avg_severity declares tickets but did not move when its clock did -- it
    reads the source without consulting availability, or the declaration is stale
```

That last line is a real bug. `avg_severity` filters tickets on `opened_at`
instead of `resolved_at`, so tickets that were still open at scoring time leak
in — exactly the ones that predict churn. Nothing about the code looks wrong.

## Try it on real data

```bash
pip install leakprobe openpyxl
python examples/online_retail.py
```

The UCI Online Retail set: 397,924 real orders and 8,905 returns across 4,372
customers of a UK gift retailer, 2010-2011. Six ordinary per-customer features
built as of a cutoff, two of them wrong in the two ways temporal leakage
actually happens. Neither raises. Both are caught:

```
feature               orders       returns
total_spend         reads it     exactly 0
order_count         reads it     exactly 0
recency_days        reads it     exactly 0
return_count       exactly 0      reads it
avg_unit_price       bypass?     exactly 0
net_spend           reads it          LEAK

1 undeclared dependencies:
  - net_spend moved when returns was perturbed, and does not declare it (max change 7.46e+03)

Declared, but did not respond to the source's clock:
  - avg_unit_price declares orders but did not move when its clock did
```

`avg_unit_price` is missing one cutoff filter, so it averages invoices that had
not happened yet. `net_spend` reaches into the returns table without declaring
it, silently inheriting that table's latency. One dropped subscript and one
undeclared read -- the two shapes this bug takes in production.

## What it catches, measured

Five public datasets, five shapes of leakage, and a correct pipeline for each.
`benchmarks/leak_zoo.py` runs it.

| shape | caught |
|---|---|
| S1 missing cutoff filter — aggregate over every row | 3/3 |
| S2 undeclared source read | 2/2 |
| S3 outcome leakage from a later-clocked table | 2/2 |
| S4 outcome read under the event's own clock | 1/1 |
| S5 statistic computed over all of time | 1/1 |
| **correct pipelines flagged** | **0/5** |

The last row matters most. A detector that flags clean code teaches you to
ignore it.

Three probes run, because no single perturbation sees everything:

- **the clock moves** (`delay`) — catches code that filters on a timestamp
- **an undeclared source's payload is permuted** (`shuffle`) — catches code that
  joins a table and takes a column off it without ever consulting its clock.
  Shifting that table's timestamps moves nothing, so `delay` is blind here.
  Applied only to sources a feature says it does not read, so it costs no false
  positives.
- **rows after the cutoff are deleted** (`truncate`, when you pass `cutoff=`) —
  catches a global mean or a z-score denominator taken over all of time. Those
  respond to a delayed clock exactly as correct code does; only removing the
  rows separates them.

```python
report = lp.check(..., cutoff=pd.Timestamp("2024-01-01"))
```

What is left: a feature that reads only a column *marginal* of an undeclared
table — a mean, a max, a quantile — and never joins on a key or consults a
timestamp. Permutation preserves marginals, so nothing moves. That boundary is
asserted in the test suite so it cannot change unnoticed.

## How it works

Four steps, and no ground truth anywhere in them.

1. **Move when a source became knowable.** Not its values — only its
   availability timestamp. `delay` pushes it later, which can only ever remove
   information.
2. **Recompute every feature.**
3. **Compare, exactly.** A feature that genuinely cannot read that source gets
   the identical input arrays through the identical code and returns bit-identical
   floats. Its difference is `0.0`, not `1e-15`. So any movement at all is proof
   of a dependency, not a number you have to squint at.
4. **Check what moved against what you declared.** Anything in one list and not
   the other is the finding.

Three kinds of finding:

- **`leak`** — moved, but doesn't declare the source. It has a dependency you
  didn't know about. In a temporal pipeline, an unknown dependency on *when*
  data arrived is look-ahead.
- **`future`** — declares the source, and changed when rows after your cutoff
  were deleted. It reads what it should not have seen. Only reported when you
  pass `cutoff=`. `raise_for_leaks()` fails on these too.
- **`bypass?`** — declares the source but didn't move when that source's clock
  did. It's reaching the data by a path that ignores availability, which is how
  look-ahead usually gets in. Not a failure on its own. Worth reading.

Before any of that, `check` runs `compute` twice on untouched inputs and refuses
to continue if the two runs disagree. A nondeterministic pipeline makes every
result below it noise, so that's a hard error rather than a warning.

## Declaring nothing

You don't have to write the `declared` map. Leave it out and every real timing
dependency is reported:

```python
report = lp.check(compute, sources, timestamps, declared={})
for f in report.leaks:
    print(f.feature, "reads", f.source)
```

That's the fastest way to answer "what does this pipeline actually depend on"
for code you inherited, which is usually a shorter list than the author believed.

## In CI

```python
def test_no_temporal_leakage():
    lp.check(build_features, SOURCES, TIMESTAMPS, DECLARED).raise_for_leaks()
```

The declaration map becomes the thing code review argues about, which is where
that argument belongs.

## Perturbations

| | |
|---|---|
| `delay(frame, col, by)` | knowable later. The safe default: removes information only. |
| `advance(frame, col, by)` | knowable earlier. Injects look-ahead on purpose, to measure what a leak is worth. |
| `use_column(frame, col, other)` | availability taken from another column. Models "treated as knowable when the period ended, not when it was published." |

```python
report = lp.check(..., perturb=lp.advance, by=pd.Timedelta(days=90))
```

## What it will not catch

**Dependencies that don't flow through a timestamp.** The perturbation moves
availability, so it reveals as-of joins, merges on a date, and windows anchored
to one. A feature that reads a source's values with no reference to when they
arrived is invariant to it and will not be flagged. In practice this costs less
than it sounds: look-ahead *is* a dependency on timing, so the leaks worth
catching are the detectable ones. The boundary is asserted in the test suite so
it can't quietly stop being true.

**Sources with no clock.** Pass `None` and static tables are skipped, with a
note saying dependencies on them went untested.

**Leakage across rows rather than time** — target encoding fit on the full
dataset, a scaler fit before the split. Different bug, different tool.

## Where this came from

The technique is metamorphic testing, which software testing has used for
decades and research code almost never does. This is an extraction of a check
built for a quantitative finance pipeline, where the sources are stock prices
and regulatory filings and the question is whether a factor read an earnings
figure before it was published.

It caught a real one. A factor called `turnover_1m` was classified as
price-and-volume — it's built from volume, it lives in `price.py`, and every
human who looked at it filed it under prices. It divides by shares outstanding,
which comes off a filing. It was the only member of its group that moved when
the filing calendar shifted, while eleven genuine price factors held at exactly
zero. That measurement is written up in
[bias-fingerprints](https://github.com/quantraunak/bias-fingerprints).

## Install

```bash
pip install leakprobe          # pandas is the only dependency
```

Python 3.10+.

## Licence

MIT.
