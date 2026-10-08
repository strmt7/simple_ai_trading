# September funding population: exact archive/API corroboration

The [frozen contract](contract.json) and [terminal result](result.json) qualify
the September 5-30 intersection of the previously retained native marked
BTCUSDT, ETHUSDT and SOLUSDT histories. Each published monthly archive contains
90 September events; its full 78-event intersection exactly matches the native
timestamps and decimal rates. All 234 paired events match, including native
millisecond offsets. No float conversion, clock rounding or missing-row fill
was used. Original native marks and old results remain unchanged.

Six public unauthenticated GETs returned 3,066 bytes: one ZIP and one published
SHA256 checksum per asset. The [durable journal](journal.jsonl) retains all six
intent/completion pairs and the study's terminal pair. Raw ZIP/checksum bytes
remain beside it. Checksums, named bounded CSV members and whole-intersection
equality were verified before advancing to the next asset. Contract, result,
implementation and raw-response bindings were audited separately.

Official archive addresses use the existing production URL builder:
[BTC](https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-09.zip),
[ETH](https://data.binance.vision/data/futures/um/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-09.zip),
[SOL](https://data.binance.vision/data/futures/um/monthly/fundingRate/SOLUSDT/SOLUSDT-fundingRate-2026-09.zip).
These are primary Binance channels accessed October 8, 2026, not independent
publishers. Published checksums establish retained artifact consistency, not
independent truth, private receipts or exchange solvency.

The new bounded binary qualifier preserves the text-only source tools and all
consumed implementations. Its locked-runtime preflight checked the exact
retained schemas, native source certificate, hashes and unused output paths
before network access. 115 affected checks pass, including 24 new cases, and
Ruff passes. They cover exact values/clocks, malformed rows, checksums, member
identity, first-failure stop, restart refusal, bounded raw/HTTP receipts, tamper
and CLI preflight. An initially omitted parser checksum argument was corrected
before access; no consumed plan or captured implementation was repaired.
No full suite, hosted CI, benchmark, GPU workload or model fit was run.

This is a narrower population certificate, not complete September 5-October 6
coverage or admitted cash labels. October's 18 native events per asset remain
uncorroborated by this study. Owned settlement entitlement, executable paired
net-base quantities, fee assets, basis, financing, continuous margin/ADL/custody
and untouched causal roles remain unqualified. The
[retained hedge frontier](../../2026-10-07/native-funding-hedge-frontier/review.md)
is not rerun or promoted; thin conditional cost budgets remain conditional.
Use this exact subset binding rather than asserting whole-history completeness.

No economic return, rate ranking, orientation selection or training admission
was computed. No credentials, accounts, orders, protected captures, campaign
logic or automation were used. Financial counts remain 202 observations / 65
hypotheses / 37 scoped mechanisms / zero qualified stable profitable edges.
