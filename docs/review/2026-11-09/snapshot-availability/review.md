# Snapshot availability in shadow promotion

The preceding native-depth review found that each generated snapshot followed
some incremental updates. Sequence continuity alone could not prove when that
snapshot was known. This checkpoint carries an explicit availability boundary
through the existing local public-feed collector, inference coordinator, shadow
replay and saved-model validator. It does not admit the third-party historical
feed, normalize new books, fit market models or establish profitability.

## Enforced behavior

New local captures use `binance-usdm-l2-v4`. After a snapshot response has returned
and passed validation, the collector records `snapshot_received_at_ns` using the
same local wall-clock domain as its received stream messages. The timestamp is
part of the capture manifest checked against the supplied evidence. It is a
local observation, not an exchange timestamp or authenticated venue receipt.

The shared `SnapshotAvailability` mapping retains the integer nanoseconds and
uses exact arithmetic with the declared clock-offset estimate. The first
eligible decision tick is the whole millisecond strictly after that estimate;
an exact millisecond tie does not round backward. Raw receive-time ties are also
ineligible. This adds no claim about clock synchronization accuracy or native
book age; the existing offset is still an estimate.

The streaming coordinator continues draining bounded buffers and building
features while inference is disabled. It rejects feature decisions before the
availability boundary before calling the scorer. Shadow replay also requires
the actual raw receipt time to be strictly later than the snapshot receipt
before inference or virtual entry processing. Skipping all processing while
unavailable would overflow the bounded aggregator and destroy warmup; that is
why the two operations are explicit.

Shadow promotion checks that the snapshot receipt lies inside the capture,
that replay applied the expected boundary, and that no virtual trade predates
it. Eligible replay duration excludes the unavailable prefix. New saved shadow
evidence uses `microstructure-public-feed-shadow-v2`; the runtime recomputes
the boundary and rejects missing, lossy or contradictory timing. Old v1 evidence
remains readable without changing its bytes, but cannot pass current runtime
acceptance. No automatic promotion or retraining occurs during loading.

## Verification and scope

The [verification receipt](verification.json) binds the changed implementation,
test sources and private JUnit output. **107 focused checks pass; one optional
HftBacktest integration skips** because that package is unavailable. The checks
include an actual streaming/shadow synthetic replay with a late snapshot,
preserved warmup, no early inference or virtual entries, duration exclusion,
sub-millisecond precision, positive/negative offsets, unavailable end-of-capture,
capture-boundary rejection and saved-model tampering/legacy rejection.

Ruff lint passes for all changed Python files. Eight files pass whole-file
format checks; changed parser/validator blocks in the larger model module pass
range formatting checks, preserving unrelated existing layout. Formatting is
not semantic review. No broad suite, new dependency, benchmark, network call,
account operation, GPU performance claim or GitHub push was required.

This is the local collector/shadow path. The normalized HftBacktest initial-state
timeline, a causal adapter for the retained third-party source, cross-collector
clock comparability, current source protocol/clock units, full paired costs,
financing and continuous solvency remain open. The previous official-page failure
and its forbidden second navigation remain terminal. Do not infer that the
historical paired dataset or all model routes are repaired by this checkpoint.

No old source/result bytes or protected partial payloads were changed or read.
Financial counts remain **203 observations / 65 hypotheses / 37 scoped mechanisms
/ 0 qualified stable edges**. The full research, app revamp, capital-protection,
security and exhaustive-review goal remains active. No capture, training, test
or index worker remains from this round, and no persistence was installed.
Ten separate Python MCP adapter processes were observed; their client ownership
is unresolved. A missing parent PID does not prove a dead client or ownership.
They were left untouched to avoid disrupting the user's concurrent tasks.
