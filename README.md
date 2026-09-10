# leakcheck

**Find features that read data they were never supposed to see.**

Temporal leakage is the most expensive quiet bug in applied ML. A feature reads
something that wasn't knowable yet, nothing throws, no number looks implausible,
and your model gets better. You find out in production, if you find out at all.

`leakcheck` finds it without needing to know the right answer. It needs one
thing you already know: **a change to your data that your features must be
invariant to.**

```bash
pip install leakcheck
```

```python
import leakcheck as lc

report = lc.check(
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

Two kinds of finding:

- **`leak`** — moved, but doesn't declare the source. It has a dependency you
  didn't know about. In a temporal pipeline, an unknown dependency on *when*
  data arrived is look-ahead.
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
report = lc.check(compute, sources, timestamps, declared={})
for f in report.leaks:
    print(f.feature, "reads", f.source)
```

That's the fastest way to answer "what does this pipeline actually depend on"
for code you inherited, which is usually a shorter list than the author believed.

## In CI

```python
def test_no_temporal_leakage():
    lc.check(build_features, SOURCES, TIMESTAMPS, DECLARED).raise_for_leaks()
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
report = lc.check(..., perturb=lc.advance, by=pd.Timedelta(days=90))
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
pip install leakcheck          # pandas is the only dependency
```

Python 3.10+.

## Licence

MIT.
