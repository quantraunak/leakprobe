# Launch copy

Post in this order, same morning, 8-10am Pacific, Tue-Thu. HN first. First
comment within five minutes of the submission. Stay near the laptop for three
hours after.

Swap the site link for raunaksood.com if the domain is live by then.

## Hacker News

Title (80 char limit; this is 78):

    Show HN: Leakprobe – test ML features for temporal leakage without ground truth

URL: https://github.com/quantraunak/leakprobe

First comment, post it yourself right after submitting:

    Author here. A factor in a backtest of mine read shares outstanding from a
    10-Q through its denominator, so a "price-only" feature was inheriting the
    filing calendar. Nothing errored. Results were just a bit too good. Took
    months to notice.

    You can't unit-test this the normal way because the correct value is the
    thing in dispute. What you do know: if you delay when a source became
    available, features that read it must change, and features that don't must
    come back bit-identical. That's a metamorphic relation (Chen, 1998), and it
    needs no oracle and no threshold.

    leakprobe runs three of those perturbations against your feature function:
    shift a source's clock, permute a source you claim not to read, delete rows
    after the cutoff. Anything that moves when it shouldn't, or holds still when
    it should move, is reported.

    Benchmark: 5 public datasets (UCI retail, NYC taxi, Chicago crimes, NYC
    311, UCI bike), 9 planted leaks in 5 shapes. 9/9 caught, 0 false positives
    on the 5 correct pipelines. The first version caught 6/9; the misses all had
    one cause, and the write-up covers what the tool still cannot see.

    pip install leakprobe. Write-up:
    https://raunaksood.vercel.app/writing/testing-for-leakage

## r/MachineLearning

Title:

    [P] leakprobe: temporal leakage detection for feature pipelines by metamorphic testing (no ground truth needed)

Body:

    Temporal leakage has no failure mode. A feature reads a value that was not
    knowable yet, metrics improve, and nobody investigates results that are
    slightly better than expected. I lost months to one of these in a backtest.

    You cannot write the expected output for a feature whose correct value is
    exactly what is in dispute. But you can state a property that links two
    runs: delay when a source became available and every feature that reads it
    must change, while every feature that does not must come back with a
    difference of exactly zero. Metamorphic testing, applied to look-ahead.

    leakprobe takes your feature function, your source tables with their
    timestamp columns, and a map of which features read which sources. It runs
    three perturbations per source (clock shift, payload permutation for
    undeclared sources, truncation after the cutoff) and reports three kinds of
    finding: an undeclared dependency, a declared source read past the cutoff,
    and a declared source the feature never actually consulted the clock of.

    Benchmark on 5 public datasets with 9 planted leaks in 5 shapes: 9/9 caught,
    0 false positives on the 5 correct pipelines. The first version caught
    6/9; the three misses shared one cause (features that responded to a
    delayed clock exactly as correct code does), which is what the second and
    third probes were built for. The write-up also states what it still cannot
    see: a feature that reads only a marginal of an undeclared table.

    pip install leakprobe
    https://github.com/quantraunak/leakprobe
    https://raunaksood.vercel.app/writing/testing-for-leakage

    Interested in leak shapes I have not planted. If you have one from a real
    pipeline, I will add it to the zoo.

## X thread

1/
    Temporal leakage has no failure mode. A feature reads something it
    couldn't have known yet, your metrics go up, nobody investigates. I lost
    months to one. So I built a test for it that needs no ground truth.

    pip install leakprobe

2/
    The bug: a "price-only" factor, turnover = volume / shares outstanding.
    Shares outstanding comes from a 10-Q. The factor was inheriting the filing
    calendar through its denominator. Nothing errored. Results were just a bit
    too good.

3/
    You can't unit-test this. The correct value is the thing in dispute.

    What you can test: delay when a source became available. Features that
    read it must change. Features that don't must come back bit-identical.
    Exactly zero, not "small."

4/
    That's a metamorphic relation (Chen, 1998). No oracle, no threshold.
    leakprobe runs three of them: shift a source's clock, permute a source you
    claim not to read, delete rows after the cutoff.

5/
    Benchmark: 5 public datasets, 9 planted leaks, 5 shapes.
    v0.1 caught 6/9. The misses had one cause.
    v0.2 catches 9/9, 0 false positives on the 5 correct pipelines.

6/
    What it still can't see, and why, is in the write-up:
    https://raunaksood.vercel.app/writing/testing-for-leakage

    Code: https://github.com/quantraunak/leakprobe

## Replies you will get, and the answer

"Why not just do a proper time split?"
    A time split catches leakage in the label. It does nothing for a feature
    that reads a value from after the as-of date, because the split is applied
    to rows and the leak is inside the row. leakprobe tests the feature
    function itself.

"Isn't this just a unit test?"
    A unit test needs the expected output. Here the expected output is the
    thing in dispute. The test is a property linking two runs, not a run to a
    known value.

"What about leaks it can't catch?"
    A feature that reads only a column marginal (mean, max, quantile) of a
    table it does not declare and never joins on. Permutation preserves
    marginals. That boundary is asserted in the test suite so it cannot change
    unnoticed. It is in the README and the post.

"Does it work with [feature store / polars / spark]?"
    It takes a function from a dict of DataFrames to a DataFrame. Anything
    you can wrap in that signature works. Polars: convert at the boundary for
    now; native support is a reasonable next step if someone needs it.

"How slow is it?"
    3 probes x number of sources runs of your feature function, plus one
    determinism check. Cost is your pipeline's cost times that. The five-
    dataset zoo (14 pipelines, ~600 MB of data) runs in 13 seconds on a laptop.
