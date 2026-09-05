# CPI within-event partitions: no retained taker edge

All 42 evaluated rows across both retained events fail even before fees.
The source snapshot supplies both sides for every market, so missing quotes
do not explain this rejection. No new network, book, fee or account requests
were made. This is not evidence of a family-wide negative expected value.

| Retained partition | Markets / frontier rows | Complete YES cost per share, payout 1 | Best gross floor at 5 shares | Best fee + one-tick floor at 5 shares |
| --- | ---: | ---: | ---: | ---: |
| Annual, event 838710 | 12 / 24 | 1.217 pUSD | -0.020 pUSD | -0.03199 pUSD |
| Monthly, event 838709 | 9 / 18 | 1.102 pUSD | -0.015 pUSD | -0.02675 pUSD |

The best rows in both columns are same-market YES/NO pairs on the lower-tail
bins, not completed trades. Gamma acquisition fields are a rejection proxy,
not synchronized executable depth or own fills. Fees and ticks are the frozen
configured sensitivity; the negative gross result does not depend on them.

## Why this does not repair the cross-event failure

The [earlier mapping review](inflation-mapping.md) still has no proved
annual-to-monthly conversion. This separate [frozen plan](cpi-partitions/plan.json)
uses only each event's own common scalar. Its one-decimal interior bins and
inclusive tails exhaust that scalar, including the common previous-month
fallback. No annual/monthly packages or component-factor assumptions enter.
The optimistic common-valid-resolution floor does not establish on-chain
question-count, dispute, conversion or account qualification.

The already retained raw bodies were hash-checked and all rule/count gates
were validated before any side-specific price calculation. The existing
long-only frontier and exact fee implementation was reused unchanged. At five
shares per leg it evaluates all YES, every binary YES/NO pair, and every
optimal NO cardinality under raw cost and fee/tick cost. A k-NO subset has
floor k-1 only under the stated mutually exclusive resolution model. These
42 related rows are not 42 independent statistical observations.

The [result](cpi-partitions/result.json) retains every row, not only the best.
The [journal](cpi-partitions/journal.jsonl) binds the one-use offline run.
This is exploratory reused evidence; pre-plan search probabilities were
already visible, and no fresh out-of-sample claim is made. Original source
captures, BLS extracts, mapping review and old failures remain unchanged.

## Decision

Close this exact within-event screen without refresh, reprice, missing-side
repair, sibling substitution or book escalation. The separate component-bound
research can proceed only with the previously identified primary evidence;
it cannot turn this result into a cross-event acceptance. Seek genuinely new
mechanism/eligibility evidence or a satisfied existing retry trigger instead.
The canonical registry adds one terminal screen: 37 scoped acceptances,
65 hypotheses, 189 terminal observations, zero current account-qualified
after-all-cost edges. No training or GPU benchmark was useful for this check.
