# Testing for leakage without knowing the right answer

A factor called `turnover_1m` broke a backtest of mine, and it took months to
notice because nothing about it looked wrong.

Turnover is a price-and-volume factor. Monthly volume over shares outstanding.
Everyone treats it as price-only, since volume comes off the tape and the tape
is available the instant it prints, so there is nothing to date carefully. I had
twenty-two factors split into two blocks, eleven that read a filed figure and
eleven that did not, and the split was load-bearing. The whole design depended on
the second block being unable to respond to the filing calendar.

Shares outstanding comes from a 10-Q.

So `turnover_1m` sat in the price-only block, inheriting the filing calendar
through its denominator. Nothing raised. No number looked implausible. The factor
behaved, the backtest behaved, and the only symptom was that results came out
slightly better than they should have, which is not a symptom anyone
investigates.

That is the whole problem with temporal leakage. It has no failure mode. A
feature reads something that was not knowable yet, your metrics improve, and you
find out in production or you find out when someone asks a question you cannot
answer. Or you never find out.

## You cannot test it the normal way

The normal way to test a computation is to know what it should produce, write the
expected output, and compare. That does not work here. If you knew what
`turnover_1m` should have been you would not have had the bug, because the
correct value is the thing in dispute.

There is something you do know without the answer. Changing when a data source
became available must change the features that read it, and must leave the
features that do not read it alone.

That is a metamorphic relation, a property linking two runs of the same program
rather than a program's output to a known value. The technique is Chen's, from
1998, and it exists precisely for programs where you cannot write down the
expected answer, which the testing literature calls
[the oracle problem](https://dl.acm.org/doi/10.1145/3143561). Machine learning
pipelines are the textbook case, and leakage is a good fit, because the property
is crisp. Delay your filings by
three weeks and recompute. Every factor that reads a filing should move. Every
factor that does not should come back bit-identical, with a difference of exactly
zero rather than an acceptably small one. That sharpness is what makes it usable.
There is no threshold to tune and no judgment call about whether a small
difference matters.

Run it on my twenty-two factors and eleven move, eleven do not, and one of the
eleven that moves is `turnover_1m`, which had declared itself price-only.

That bug was worth having. Joining fundamentals on period-end rather than filing
date inflated mean IC by 59%, and pushed 4 of 11 factors across t = 2 that did
not belong there.

I packaged the check as [leakprobe](https://pypi.org/project/leakprobe/). You
describe what each feature is supposed to read, it perturbs each source in turn,
and it reports anything that moved when it should not have or held still when it
should have moved.

## Then I tried to break it

A tool that finds the one bug it was built for is not evidence of anything. So I
built a zoo: five public datasets, five shapes of leakage, and a correct pipeline
for each dataset.

The datasets are UCI Online Retail (406,829 transactions with a customer
attached), NYC yellow taxi (2.9 million January 2024 trips), Chicago crime
reports (263,837 from 2023), NYC 311 (200,000 service requests), and UCI bike
sharing. Real data, ordinary features: per-customer spend, per-zone fare
averages, per-district report counts.

The five shapes of leakage:

- S1, a cutoff filter is missing, so an aggregate runs over every row
- S2, a feature reads a second table it never declared
- S3, outcome leakage, where the outcome lives in a table with its own later clock
- S4, the same, but the outcome is read under the event's own timestamp
- S5, a z-score whose mean and standard deviation come from all of time

The correct pipelines mattered more to me than the bugged ones. A detector that
flags clean code is worse than no detector, because it teaches you to ignore it.

The shipped version got zero false positives on the 5 correct pipelines, and
caught 6 of 9 planted leaks.

Six of nine is not a good number for a tool whose pitch is finding what you
cannot see.

## The misses had one cause

Chicago was the clearest. I had split the crime data into reports, carrying an
incident date, and dispositions, carrying an `updated_on` stamp, because whether
an arrest happened is knowable only later. Then I wrote the classic bug: join the
disposition table, take the arrest flag, compute an arrest rate per district, and
declare the feature as reading reports only.

leakprobe saw nothing.

The reason is almost obvious once you say it out loud. The perturbation moves
timestamps, and moving a timestamp only removes rows from code that filters on
that timestamp. My buggy code never looked at `updated_on` at all. It joined on
an ID and took a boolean. Shift the disposition clock by three weeks and the
arrest flag on every row is the same flag it always was, so nothing moves and
nothing is reported.

The same explanation covers S5. A z-score denominator computed over all of time
responds to a delayed clock the same way a correct one does, because the visible
slice shifts either way. Both versions move. Movement was the signal, and here it
carries no information.

I then found this limitation already documented in my own test suite, in a test
asserting it so it "cannot quietly stop being true," with a docstring calling it
"the method's boundary rather than a miss." I had written the gap down and then
stopped thinking about it, which is a comfortable way to stay wrong about
something.

## The principle

A perturbation detects a dependency only if it changes what the feature actually
reads.

Delay changes which rows are visible, which is the right instrument for code that
consults a clock and the wrong instrument for everything else. So I added two
more.

The first permutes the payload. For a source a feature claims not to read,
shuffle every column except the timestamp. Row count unchanged, clock unchanged,
every row-to-row association destroyed. A feature that genuinely does not read
that table cannot notice, while one that joins it and takes a column off it moves
immediately. This costs no false positives because it only ever runs on pairs the
user declared independent, which turns that declaration into something tested
rather than assumed.

The second deletes the future. Given the cutoff your features are built for, drop
every row after it. A feature built as of a cutoff cannot notice the removal of
rows it was never allowed to see, so anything that moves was reading them. This
is the only probe that catches the all-of-time z-score, and it catches it
cleanly.

Three probes now run, asking different questions. Does this feature filter on
this clock, does it touch this table's contents, and does it reach past the
cutoff.

9 of 9 caught. Still 0 false positives on the 5 correct pipelines.

## What it still cannot see

Column-wise permutation preserves a column's marginal distribution. A mean is a
mean after shuffling, and so is a max or a sum or a quantile. A feature that
reads only an aggregate of an undeclared table, never joining on a key and never
consulting a timestamp, survives every probe.

In practice this is narrow. Look-ahead is a dependency on timing or on row
identity, and both are covered now. What remains is a global constant lifted off
an undeclared table, and it is asserted in the test suite so that it cannot
quietly stop being true. That phrasing is load-bearing this time.

## What I would take from this

The useful idea is not the tool. It is that you can test a computation you cannot
verify, as long as you can name something it must be invariant to. Temporal
availability is one such thing, and a good one, because leakage is by
construction a dependency on time. There are others. A feature set should be
invariant to row order, to a rename of an ID column, to shuffling entities that
are supposed to be independent.

The second idea is less comfortable. I documented this tool's limitation in a
test, then treated having documented it as having dealt with it. It took building
an adversarial benchmark on data I did not choose to turn that note back into a
problem. Writing down what your method cannot do is where the work starts.

```bash
pip install leakprobe
```

The zoo is in `benchmarks/leak_zoo.py` if you want to run it against your own
pipeline, or add a shape I did not think of.
