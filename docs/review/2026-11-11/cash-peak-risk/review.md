# Settled-cash peak capital admission

The Binance autonomous entry gate previously equated drawdown with loss from the
starting reference. A retained +100 quote-unit gain followed by an 80-unit loss
left positive lifetime PnL and therefore reported zero drawdown. With a 1,000-unit
reference and a 5% configured drawdown limit, both direct capital admission and
the real entry gate incorrectly admitted more risk. The same result survived
restart. Three source-bound original-code regressions reproduce that behavior.

The source baseline is `abf36376761fc198cd89fc7ef4f3208be5b0225a`.
The [evidence](evidence.json) binds exact original/final source and test phases.
This is a partial `SAFE-02` capital repair, not the full durable policy service.

## Accounting and integration

For the retained closed rows in one already-qualified quote unit, group realized
PnL by close timestamp and order the groups chronologically. Let `H` be the
maximum of zero and every grouped cumulative realized-PnL prefix, `R` the total
retained realized PnL, `U` the current qualified unrealized PnL and `K` the existing
positive risk-capital reference. The new metric is `max(0, H - R - U) / K`.

The denominator is the configured capital reference, not peak account equity.
Fractions preserve the exact declared decimal values; they do not recover
precision or native provenance lost before those values reached the ledger.
Equal-time groups avoid inventing an intermediate peak from file order. Invalid
or future chronology rejects. Unknown current marks/quote/fees retain the existing
admission blocks. Current unrealized PnL enters once; entry fees are not charged
again by the peak calculation.

The shared autonomous capital guard now enforces this configured drawdown limit
and provides the same metric to the entry gate. A breach requests owned reduction
through the existing loop's risk-close paths, preserving their Stop, ownership,
receipt and UNKNOWN rules. There is no new account operation or write transport.
Explicit owned closing remains separate from opening admission.

Capital snapshots now use strict parsing. Malformed/incomplete retained rows
cannot silently become an empty loss record. Invalid effective loss-limit values
(nonfinite, negative or boolean) and noninteger loss-streak counts block admission
instead of silently disabling it. Existing constructor normalization is separate
and unchanged; this is not a strict versioned configuration/activation contract.

The peak is reconstructed from existing durable history on each guard evaluation.
Restart with the same history/reference does not reset it. No second database,
automatic state provisioning, migration or rearm was introduced.

## Verification and lessons

The final affected suite passes **264 cases** across autonomous execution, capital
valuation, native Spot cash/inventory and the new cash-peak cases. These include
profit retracement, restart and later UTC days, current mark losses, file-order
independence, equal-time grouping, exact units, malformed history and invalid
effective policy fields. Existing native receipt/owned execution cases remain in
the affected suite. No actual market/account, credentials, order, training or
benchmark was used.

The original three cases fail against unchanged production source. The first
changed-source domain phase retained 245 passes and 19 test-fixture failures:
constructor normalization prevented invalid effective-value injection; direct
rewriting of an enrolled ledger correctly triggered its integrity guard; and an
old snapshot wrapper did not accept the new strict argument. The fixtures now
exercise effective values explicitly, test ordering without corrupting enrolled
storage, and verify one strict coherent snapshot. All original phases and their
complete source/log bindings remain retained; they were not relabeled as passes.

## Remaining capital requirements

This settled-PnL metric is not complete marked-NAV drawdown. Intermediate marks,
omitted/replaced partial-fill cash prefixes, withdrawals/deposits, account-wide
assets/liabilities, FX, native funding/margin and independent ledger rollback
protection remain separate requirements. Neither a small metric nor the test
result proves complete financial health. The capital reference and loss settings
are not yet bound to an independently activated persistent policy revision.

Full `SAFE-02` still needs that versioned policy and transactional state,
chronological native cash/mark checkpoints, UTC-day baselines, external-flow
handling, reservations, model/generation identity, quarantine and controlled
rearm. Independent supervision and universal writer coverage remain open. This
phase changes no Polymarket gate, historical research result, financial census,
edge qualification or trading/release authority.
