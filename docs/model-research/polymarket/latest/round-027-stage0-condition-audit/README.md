# Round 27 Stage 0 replay eligibility

> Target-free data-integrity evidence only. No P&L, edge, profitability, paper-trading, or live-trading claim.

![Condition replay eligibility](condition-replay-eligibility.svg)

| Evidence | Result |
|---|---:|
| Exact UTC capture | 2026-08-14 22:59:11 to 2026-08-15 03:59:14 |
| Recorded conditions | 59 |
| Condition-isolated eligible | 53 |
| Excluded | 6 |
| Recorded stream gaps | 9 |
| Minimum executable interval | 250 ms |

Eligibility means that a condition had a fresh two-outcome baseline and a
checksum-valid replay interval bounded by the connection's recorded lifetime.
It does not make the full capture model-eligible. The JSON and CSV files are
the numeric sources of truth; the SVG is derived from them.
