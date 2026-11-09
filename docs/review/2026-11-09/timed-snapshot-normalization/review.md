# Timed native replay initialization

The local exporter previously loaded a REST snapshot through HftBacktest's
static `initial_snapshot` API. A synthetic probe of the actual 2.4.4 engine
showed finite prices before clock initialization and before the recorded
snapshot receipt. That is a look-ahead path, regardless of strategy quality.
The repaired exporter puts snapshot clear/seed events into the timed stream,
leaves the standalone initial-state artifact empty, and never loads it as a
static initial book.

## Causality and exact inputs

The snapshot becomes locally available at its recorded receipt plus the declared
clock-offset estimate. Buffered updates become available no earlier than that
boundary; stable source order releases the seed before buffered updates at a
shared timestamp. Later messages retain their corrected receipt. The raw source
is unchanged. Logical release times are explicitly distinguished from physical
receipts in the normalization report.

The adapter rejects missing or lossy native identities, missing snapshot receipt,
nonfinite or boolean economics, sequence gaps, regressing depth/receipt clocks,
crossed initial/BBO prices and negative corrected feed latency. It does not let
the converter's automatic timestamp repair conceal those failures. Integer
offset rounding uses exact decimal arithmetic, conservatively upward to a
nanosecond. IDs above `2**53` remain integers. Normalization outputs and its log
must be new; temporary conversion input is scoped to the call.

The smoke check verifies an empty standalone initial state, waits beyond the
snapshot boundary and requires both a successful native clock step and finite,
uncrossed quotes. End-of-file can apply the last event without advancing the
engine clock, so end-of-file is a failure, not proof of a valid observation.
The smoke test's fixed order-latency/fee settings are technical fixtures, not
calibrated execution economics. No order is submitted.

## Verification and limits

The first focused run had 115 passes and one existing readiness-fixture failure:
its asserted-valid payload still named capture v3 while production requires v4.
The fixture now explicitly checks current v4, rejection of v3, and rejection of
smoke-only evidence without strategy replay. All 20 readiness checks pass.
After the final receipt-regression guard, all 64 normalization/capture checks
pass. Native integration shows no book before receipt, then the buffered delta
quantity rather than the older seed quantity. Source hashes remain unchanged.
Authored Python files pass Ruff. These overlapping lanes are not a green full
repository suite; the earlier broad dependency audit had 86 unresolved failures.

The one-use official REST documentation attempt returned HTTP 302 to a canonical
catalog route, final HTTP 202, and one whitespace character of Markdown. It is
rejected despite native `success=true`. Both discovery attempts returned the
wrong product or a change log and qualify no field semantics. Native responses,
provider version/digests, journals and failures are retained privately; the
controller is consumed. No alias retry or alternate reader was used.

The adapter's declared millisecond units and clock estimate are conditional
inputs, not independent Binance protocol/clock qualification. This repair does
not qualify the third-party historical adapter, continuous paired data, passive
fills, a strategy replay, financing, margin, capacity or solvency. Previously
saved artifacts are unchanged and gain no new validation from this work.

Financial counts remain **203 observations / 65 hypotheses / 37 scoped mechanisms
/ 0 qualified stable edges**. This round used synthetic fixtures and one public
technical document only: no market/account API, credentials, fit, orders,
economic observation or protected partial-data access. Next prioritize native
paired cash/cost and source qualification under an eligible distinct question;
do not train on unqualified data or repeat consumed captures to seek a favorable
result. The broader goal remains active, with no persistent worker installed.

See [source-bound receipts](evidence.json) and the frozen discovery/crawl intents.
