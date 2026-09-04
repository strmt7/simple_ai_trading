# CPI cross-market mapping: definitions before probabilities

The exact August annual/monthly event pair does not yet support an exact
cross-market hedge. This is a source finding, not measured negative expected
value. No market prices, fees or books were screened. The broader conditional
component-bound idea remains research, not a qualified profitable edge.

Two frozen public Gamma requests retained all 12 annual and 9 monthly
definitions for events 838710 and 838709. The annual contract uses unadjusted
12-month inflation; the monthly contract uses seasonally adjusted one-month
CPI-U. Both use the first relevant official release to one decimal place and
can fall back to a previous available month after a delayed-release deadline.
Matching dates therefore do not prove a common scalar or a fixed outcome map.
See the [complete retained definitions](inflation-definition-pair/definitions.json)
and [source plan](inflation-definition-pair/plan.json). Search probabilities
were visible before that plan and remain inadmissible discovery values.

## What the official methods change

BLS distinguishes direct seasonal adjustment from aggregation of adjusted
components. Its current usage page identifies 81 all-items components, 45
seasonally adjusted in 2026, and warns that seasonal data are revised.
The older methodological page describes aggregating adjusted detail for
higher-level indexes. Neither retained page supplies a current, exact
single-factor replication contract for this pair. Sources:
[BLS usage and revision guidance](https://www.bls.gov/cpi/seasonal-adjustment/using-seasonally-adjusted-data.htm)
(modified February 13, 2026) and
[BLS aggregation methodology](https://www.bls.gov/cpi/seasonal-adjustment/estimation-seasonal-effects.htm)
(modified February 11, 2019, citing 2015 methods).

The BLS evidence is retained **web-tool extraction**, not original HTTP bytes;
the Gamma bodies and request journals are exact retained bytes. The
[structured review](inflation-definition-pair/review.json) binds both types
without presenting them as equivalent or historical implementation proof.

## Conditional counterexample and useful remaining lead

In a toy two-component additive index, unadjusted contributions `(90,110)`
and `(110,90)` both total 200. With fixed component factors `(0.9,1.1)`, their
adjusted totals are 200 and approximately 204.0404. With fixed denominators
of 200, annual inflation is zero in both examples while monthly inflation is
zero versus approximately 2.0202 percent. This is not actual CPI data or a
BLS replication; it defeats inferring a unique adjusted outcome from an
unadjusted total without the required component constraints.

A useful conditional bound survives. If the same positive component
contributions `v_i` aggregate as `NSA=sum(v_i)` and `SA=sum(v_i/f_i)`, then
`SA/NSA` is a weighted average of reciprocal factors and lies between their
minimum and maximum. A future source-qualified factor/weight contract might
therefore prove a bounded joint support even without a unique mapping.
Do not assume these simplified aggregation equations implement actual BLS.

Before any economic screen, bind the correct first-release aggregation and
factors, denominator vintages, rounding intervals and fallback-month cases.
Prove exhaustive admissible annual/monthly outcome pairs, then price only
packages whose floor survives every pair. Revised historical data cannot
stand in for information available at entry. No factor-history download or
model training is justified merely by correlation between the headlines.

## Continuation boundary

Both exact event GETs and both BLS page opens are consumed; no alias, refresh,
sibling substitution or price rescue. A distinct complete component-mapping
source can advance this question; otherwise switch lanes. Only a separately
frozen, source-qualified retained-price screen could later use these raw
metadata bodies. That would be exploratory, not untouched validation.
Rank 31 gains a source-review artifact; accepted and terminal economic counts
do not change. Old experiments are untouched. This is not the whole-repo
review or evidence of institutional-grade profitability.
