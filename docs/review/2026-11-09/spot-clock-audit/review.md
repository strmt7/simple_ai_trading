# Retained Spot clock diagnostic

The fixed October 1 minute was processed once using the vendor-declared
microsecond mode selected before access. All 1,755 depth updates and three
generated snapshots passed the exact clock-only scope. No market download,
economic field inspection, book reconstruction or model fit occurred.

| Symbol | Depth events | Minimum receipt minus E, ms | Maximum receipt minus E, ms | Negative differences |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 600 | 2.2044 | 17.5574 | 0 |
| ETHUSDT | 600 | 1.6642 | 16.5885 | 0 |
| SOLUSDT | 555 | 1.8113 | 11.5847 | 0 |

These are observed differences between two clocks. They are **not physical
latency, a native freshness certificate or an absolute offset estimate**. No
relative offset was assumed zero or fitted from these signs. The unit was not
chosen by magnitude or by a more favorable comparison. The authoritative
[result](result.json) stores exact integer nanoseconds; the table divides those
integers by one million.

The observed difference equals elapsed event-to-receipt time plus collector
clock offset minus exchange clock offset. Its positive sign cannot isolate
those terms. A collector-ordered feature path can retain the snapshot's actual
availability and a nullable exchange timestamp; native-clock execution replay
and age-sensitive admission need their additional evidence.

No event-clock regression appeared within a symbol, and collector receipts did
not regress in source order. Every snapshot lacks both native `E` and `T` fields.
Its collector receipt must remain availability evidence, not an invented exchange
timestamp. The original source still requires buffering around late snapshots;
the [earlier identity audit](../depth-source/review.md) and its results are
unchanged. This minute does not qualify continuous coverage or cross-collector
clock accuracy.

The [intent](intent.json) fixes the input digest, declared conversion, field scope
and terminal consequence. The [evidence](evidence.json) binds the exact controller,
preflight, journal and private results. The complete retained loader was checked
before freezing. Its first scope preflight omitted the documented `@100ms` stream
suffix; the allowlist was corrected before any `E`-to-receipt comparison. That
local preflight error remains separate from the terminal audit. Raw bytes and old
numerical studies were not changed.

This resolves the declared-unit arithmetic question for the supplied source
while preserving uncertainty. A causal Spot adapter must keep snapshot
availability separate from absent snapshot exchange time, bind every source's
unit explicitly, and reject unsupported initial joins or gaps. The futures
`T`/`pu` normalizer cannot supply those Spot semantics. No economic, passive-fill,
training or strategy-replay admission follows. Financial counts remain
**203 observations / 65 hypotheses / 37 scoped mechanisms / 0 qualified stable
edges**. The audit is terminal and no persistent worker was installed.
