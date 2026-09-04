# Clearing source: useful mechanics, unresolved sign conflict

The distinct [Binance RCH Clearing Procedures PDF](inverse-clearing-source.pdf)
was captured once: HTTP 200, 716,997 bytes, SHA-256
`53197b612332da02c20b5b7d19b81ff53ee5f4938c6330c72a30a1ca4f91049f`.
The fixed request used an eight-MiB streamed ceiling, 30-second cancellation,
no redirects/cookies and durable start/completion records. No market/account
request followed. The previous FAQ failures and conditional scenario results
remain unchanged. This source is not proof of current account applicability;
its effective date has not been established.

The [structured review](inverse-clearing-source-review.json) separates facts,
inferences, page coverage and unresolved gates. PnL and settlement pages were
visually inspected as well as text-extracted; the conflict is visible in the
PDF, not just a parser ordering problem.

## Decision-changing findings

1. Section 80 (printed page 26) uses the entry-minus-mark reciprocal difference
   times direction, with long positive and short negative. Section 86.2
   (page 28) instead prints mark-minus-entry with the same direction labels.
   For normalized notional 100, entry 100 and mark 200, a short loses 0.5 coin
   under the former and gains 0.5 under the latter. With one initial collateral
   coin and spot 200, these produce quote wealth 100 versus 300. We cannot
   silently select the convenient sign or claim the PDF unambiguously validates
   the venue adapter. This is a **source inconsistency**, not evidence Binance
   actually pays the wrong sign or an exploitable exchange bug.
2. The maintenance definition includes a maintenance-amount adjustment, not
   only a proportional rate; current tiers live in a separate table. The prior
   model remains an illustrative proportional scenario, not liquidation proof.
3. Funding cash enters the settlement asset and debits may reduce margin.
   Combined with reciprocal COIN-M position valuation, this supports tracking
   each cash flow in coin and its later quote conversion rather than summing
   rates as if all settlements had identical quote value.
4. Dated settlement references an every-second index average over the final
   30 minutes. An instantaneous spot exit need not match that average. Delays,
   fees, entry restrictions near expiry, default close-out and possible
   settlement reversals must remain in the risk/operational specification.

An offline metadata projection also reused the exact August 31 exchange-info
bytes: all nine BTC/ETH/SOL perpetual/current-quarter/next-quarter rows.
Retained BTC contract unit is 100 USD, ETH and SOL units 10 USD, collateral is
the respective coin. Those are **historical metadata**, not a current listing,
price, fee or availability claim. USD quote units cannot silently become USDT
or USDC realized cash. No funding outcomes were reopened.

## Next action

The source gate did not pass unambiguously, so this does not justify new
inverse-market sampling. First require independent official correction or
authoritative reconciliation of the signs and effective-version applicability;
do not refetch the same PDF or FAQ locale aliases. Reuse the retained document
for conditional accounting and settlement-risk work. Preserve the original
rank-14 fee/account trigger and terminal linear/inverse funding result.

Continue other eligible economic lanes when they offer better information
value. This issue does not impose a global research pause. It narrows the next
inverse-source question and rules out treating a simpler collateral formula
as an already-qualified profitable hedge.

Verification: three offline checks passed for the retained PDF/runner/journal
hashes, the numerical sign counterexample and all nine historical instrument
units. Ruff passed; no broad CI or GPU timing was run. A read-only GitHub check
at this checkpoint returned zero open PRs and zero open Dependabot alerts.
This does not close the separately documented SDK migration or whole-repo
semantic review. A narrow extracted-text scan found no private-key blocks,
assigned API-key examples or bearer-token patterns in this public document;
that scan is not an exhaustive security audit.
