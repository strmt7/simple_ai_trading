# Native depth evidence and ingestion repair

The fixed October 1 first minute contains actual generated snapshots for all
three Binance spot and USD-M perpetual books. Unlike spot BBO messages, spot
incremental depth carries a native `E` field. This advances the historical paired
data route beyond metadata; it does not qualify a trading strategy or model.
The [result receipt](source-result.json) binds the original contracts, raw bytes,
journals, offline audit, implementation and focused verification.

## What the source actually contains

| Instrument | Spot increments | Perpetual increments | Snapshots per venue |
| --- | ---: | ---: | ---: |
| BTCUSDT | 600 | 2,263 | 1 |
| ETHUSDT | 600 | 2,238 | 1 |
| SOLUSDT | 555 | 2,073 | 1 |

Two sequential anonymous raw-feed requests retained 1,719,812 compressed bytes
and 8,541,668 decoded bytes. The same preselected minute and all three symbols
were fixed before access. Both depth and generated-snapshot channels were
documented in retained vendor metadata; they are distinct from the consumed BBO
request. No replacement minute, price inspection or economic ranking occurred.

The [zero-network identity audit](offline-intent.json) examines only envelope,
update-ID and clock fields. All native identity/clock fields examined are exact
positive integers. No decreasing native `E` value or observed per-symbol link
break appears inside the minute. Spot interval adjacency and perpetual
`pu`-to-previous-`u` equality are source diagnostics, not a new official protocol
qualification, full-day coverage certificate or authenticated-fill evidence.

Every generated snapshot arrives **after** at least one incremental update.
An adapter must buffer and bridge updates and retain when the snapshot became
available. A static snapshot ID alone cannot establish causal availability;
backdating later snapshot knowledge into earlier decisions would leak information.
The existing synchronizer takes a static snapshot ID and is not, by itself,
proof that this new source can feed causal training. That integration remains
open. No book was reconstructed and no price/size value was inspected here.

## Official source failure

The separately frozen [documentation attempt](documentation-intent.json)
returned original HTTP 302, effective HTTP 202, native success true and one
Markdown character for the Spot page. Admission failed. The USD-M second page
was forbidden and never requested. The failure, redirect URL and native bytes
remain preserved; no alias, alternative reader or challenge bypass was used.
Current official bridge rules and clock units were not qualified by this request.
The independent ingestion repair below relies on exact data types and finite
values, not on inferred protocol changes from an empty page.

## Demonstrated production defects repaired

The existing futures ingestion/synchronization path used `int()` coercion for
native update identities. Fractional values and numeric strings could therefore
become apparently valid sequence IDs. Its observer also accepted some NaN or
infinite trade/BBO values, let malformed clock fields escape its invalid-event
handler, and advanced sequence/timing statistics before validating a message.

The repair integrates exact positive-integer validation into snapshot fetching,
depth observation and synchronization, preserving integers above `2**53`.
Snapshot levels are validated before normalization. Trade/BBO prices and sizes
must be finite, and invalid events cannot advance the repaired sequence/timing
statistics. Existing native source/result bytes and old numeric studies are
unchanged. This is a scoped repair; remaining trade-ID coercions, snapshot
availability integration and other ingestion paths still need review.

Twenty-four focused regression cases failed against the previous implementation.
After repair and the added snapshot/precision checks, **44 pass and one skips**.
The skip is the pre-existing optional HftBacktest snapshot integration because
that package is unavailable; it is not an integration pass. Ruff lint and format
checks pass. AST comparison proves behavioral edits are limited to snapshot
fetching, observation and synchronization plus two validation helpers; the other
definitions have formatting changes only. No broad suite, training, benchmark,
GPU claim or dependency installation was necessary. An initial test invocation
failed during setup because its new temporary parent was absent; that local
setup error was corrected before the actual red/green comparison.

## Next decision and limits

Reuse these retained bytes. Qualify current protocol and native clock semantics
through a genuinely eligible distinct primary source before admitting a source
adapter. Preserve buffered-update order, snapshot availability, disconnects and
native/collector clocks separately; do not simply load the generated snapshot
at time zero. Then qualify paired execution costs/cash, financing, equal initial
wealth and continuous margin before fitting or assessing profitability.

Counts remain **203 observations / 65 hypotheses / 37 scoped mechanisms /
0 qualified stable edges**. This source diagnostic adds no independent economic
observation. No protected partial payload, account, credential, order or funds
was accessed. Controllers were retained as inert text after completion; no
persistent installation or GitHub push was made. The full goal remains active.
