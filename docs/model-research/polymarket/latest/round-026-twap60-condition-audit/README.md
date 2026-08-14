# Round 26 replay eligibility

> Target-free data-integrity evidence only. No P&L, edge, profitability, paper-trading, or live-trading claim.

![Condition replay eligibility](condition-replay-eligibility.svg)

| Evidence | Result |
|---|---:|
| Exact UTC capture | 2026-08-14 19:50:00 to 2026-08-14 20:50:03 |
| Recorded conditions | 13 |
| Condition-isolated eligible | 11 |
| Excluded | 2 |
| Recorded stream gaps | 2 |
| Minimum executable interval | 250 ms |

Eligibility means that a condition had a fresh two-outcome baseline and a
checksum-valid replay interval bounded by the connection's recorded lifetime.
It does not make the full capture model-eligible. The JSON and CSV files are
the numeric sources of truth; the SVG is derived from them.
