# Payoff execution compatibility belongs at entry admission

Baseline `690f260fb44531c97f2aab1809bda3f3a5a84045` corrected fixed-base
[exit sizing](../model-utility/review.md) and versioned its modeled cost law.
The ordinary autonomous decision builder still consumed a payoff expert tagged
with the old law. The retained baseline fixture reached numerical inference
with that stale expert. This was a component reproduction, not a venue order,
proof of financial promotion or an account operation.

Historical model loading remains unchanged. Blocking the generic loader would
also impede archival research and could introduce another dependency into
owned-position recovery. The new check is an explicit entry-only compatibility
boundary, separate from reading an artifact or closing existing exposure.

## Current behavior

The shared metadata helper checks every `signed_payoff_` expert against the
runtime's exact `EXECUTION_MODEL_VERSION`. It admits no coercion, whitespace
repair or automatic metadata upgrade. Missing, malformed and incompatible
dependent metadata yields a fixed reason code; untrusted metadata values are
not included in the diagnostic. Empty or unrelated expert sets retain their
existing gates. A matching tag establishes only this compatibility condition.

The real autonomous builder checks the loaded model before candle/feature work,
then checks again before prediction after that work or a paper fallback fit.
An incompatible model produces a FLAT decision from the independent public
price reader, with explicit `model_entry_ready=False` and a reason. It performs
no numerical prediction, external signal collection or AI review on that path.
The entry gate requires Boolean true for readiness and exposes the denial even
when the side is FLAT. Existing model overrides and promotion checks were not
relaxed. Retagging an old artifact is not requalification.

The running heartbeat gives the deterministic model denial priority over an
empty FLAT message or AI status. Its writer and schema otherwise retain their
existing behavior. The AI shadow wrapper remains unchanged and copies decisions
with `dataclasses.replace`, preserving the added fields.

## Verification

The [intent](intent.json) froze the stale-model question before baseline access.
[Sealed evidence](evidence.json) binds the actual source snapshots and logs.
The baseline failed at the intended assertion: the stale payoff model reached
prediction. The first repaired phase passed 40 focused cases. After adding the
operator-message coverage, the complete affected lane passed **316 cases**;
the earlier count is included in that domain, not additive coverage.

Cases cover all three current payoff expert families, exact current/old/missing/
malformed tags, mixed experts and immutable collections, explicit Boolean entry
readiness, post-feature revocation and the first paper-fallback fit. A real
serialize/load roundtrip preserves the original v2 artifact bytes while entry
compatibility rejects it. Two existing CLI fixtures now explicitly declare the
empty expert list supplied by the real model type; their prior behavior remains.

The owned paper fixture stages inventory through the actual shared broker and
journal. With the stale-model path active, automatic threshold closing and
operator Stop each close that owned position, create no new opening and retain
the model-denial heartbeat. These are local synthetic lifecycle controls, not
authenticated testnet or live execution acceptance. Existing autonomous, Spot
cash/inventory, AI-assist and new-command modules pass, together with the exact
autonomous-builder cases from the large CLI coverage module. The whole latter
module and whole repository were not rerun. Targeted Ruff checks passed.

## Remaining boundaries

This guard covers the ordinary autonomous builder and its entry gate. Injected
custom decisions, other execution callers and other model families are not a
universal version-admission boundary. The existing startup loader can still
reject other unreadable, unpromoted or incompatible models before the loop;
this change does not complete model-independent recovery for every failure.
Independent supervisor/gateway integration, native crash/reboot recovery and
complete caller fencing remain required.

A current tag does not prove the training law, native costs, data continuity,
causal targets, statistical significance, untouched utility or profitable fills.
Do not rewrite historical evidence or retrain on consumed populations to make
an artifact pass. New source-complete economic qualification remains necessary.

No production fit, source/market/account request, venue order, real credential,
protected-data read, GPU benchmark, dependency installation, recurring task,
service or release was part of this work. Tests use synthetic/local fixtures.
Financial counts remain **206 observations / 65 hypotheses / 37 historical
scopes / 0 qualified stable edges**. The complete goal remains active.
