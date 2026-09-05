# Invalid account quantities are not flat-account evidence

Fixed a reproduced false reconciliation pass before implementing recovery/rearm.
The old account gate checked only that `positions` or `balances` was a list;
downstream numeric defaults converted malformed quantities into zero. Four
offline reproductions passed invalid accounts as reconciled: futures NaN, Spot
NaN, negative free offset by positive locked, and duplicate futures rows.

## Shared repair

The new small quantity validator is used by both reconciliation and the public
exposure extractor. It rejects missing/nonfinite/boolean/overflowing quantities,
negative Spot components, nonfinite Spot totals, malformed row identities,
duplicate symbol/side or asset rows, and contradictory futures side/mode rows.
Valid one-way rows and distinct LONG/SHORT hedge rows remain supported for
reconciliation; that does not qualify the order adapter for hedge-mode execution.

Invalid account evidence now produces a failed report and no inferred exchange
exposure. It does **not** classify verified local inventory as stale or remove it.
An exposure count of zero in a failed report is not flat-account proof. Invalid
quantity tolerances are rejected instead of allowing infinity to erase mismatches.
Ancillary entry-price/notional diagnostics, authentication/freshness proof and
fee accounting are outside this patch's quantity-validation claim.

## Verification and critical review

[Canonical source bindings and commands](binance-account-quantity-evidence.json)
record 205 distinct affected checks across stages, including 46 new validation
cases and four CLI checks. The main affected run passed 199; two subsequently
added Spot/Futures lifecycle cases prove a valid-flat baseline permits entry and
the otherwise identical malformed account blocks only on reconciliation. Their
initial fixture also hit mandatory diversification; the fixture was corrected to
the supported three-symbol configuration without weakening the production rule.
All four original reproductions now fail closed. Ruff and diff checks pass.

The review followed quantity flow into reconciliation, then checked consumers of
stale-position and invalid-account counters. That identified the need to keep
unknown evidence separate from a genuine local-only mismatch. Existing valid-flat
and local/exchange mismatch behavior remains covered. Python/regression skills
guided the shared pure validator and paired controls; the documentation skill
kept this new evidence separate from historical reports.

## Priority decision and remaining work

Before this repair, the current queue and research boundaries were inspected.
The September 6 block-trade/CXMT time gates were still future at September 5
12:31 UTC. Inspected Paradex/Backpack rows require distinct prospectively frozen
populations or material changes, not rolling retries; no sufficient new
information-gain case was established in this pass. The full wallet buy audit
requires a complete causal cash/inventory ledger, not another selected-lock grid.
This is a scoped routing decision, **not a claim that every hypothesis is blocked**.
The demonstrated false-flat capital-risk path took priority over another capture.

No venue requests, real credentials, orders, protected payloads, historical-result
changes, GPU workload or unrelated process interference occurred. This is not
profitability, terminal-order recovery, rearm, or enterprise-readiness evidence.
Continue exact scope-bound terminal recovery, explicit rearm, inventory/fee
accounting and independent supervision; keep financial R&D first when an
informative eligible test becomes available.
