# Closing native components and unsupported Spot opening scope

The review demonstrated a capital boundary mismatch: the opening intent accepted
Spot `SHORT`, while owned closing supports only Spot `LONG`. The rejection test
failed before repair. Request validation now stops unsupported Spot shorts before
journal creation or durable exchange submission. Spot longs and Futures shorts
remain supported. Historical unsupported UNKNOWN obligations remain unchanged
and block admission; they are not silently deleted, migrated or rearmed.

The [source-bound record](closing-native-components.json) also binds the new
offline `retain_closing_inventory` library boundary. It revalidates retained
terminal closing fills against the parent request and scope, then stores allowed
native components in the existing intent journal. Opening and closing share
explicit instrument-unit validation. The existing write helper already owns
the immediate transaction; there is no second authority database.

Spot sale principal and commissions retain their actual asset units. Futures
reduction changes derivative quantity, not a principal quote cash purchase or
sale. Native fee assets and signed rebates stay separate without guessed FX.
Reported Futures PnL is retained separately with the supplied margin-asset unit;
it is not combined with fees into posted cash or bot-lot profit. A synthetic
110-quote close with reported PnL 10 and 0.11 commission therefore retains three
distinct facts, not 110 principal proceeds or an admitted 9.89 account profit.

Two schema searches surfaced an official
[COIN-M document](https://developers.binance.com/en/docs/products/derivatives-trading-coin-futures/user-data-streams)
and then no matching result. COIN-M examples do not qualify USD-M cash semantics;
no example value or search result became economic evidence. These were search
discovery, not repository byte-bound primary captures. Independent USD-M
settlement/fee and account-attribution proof remains required.

562 affected checks pass, including 28 new cases; Ruff passes. Cases cover
principal/contract distinction, fee assets/rebates, long/short reduction, exact
decimals, partial and zero fills, persisted idempotence, concurrent writers,
scope/unit mismatch, missing evidence, tamper and pre-submission refusal. An
extra transaction start and a test argument-order error were corrected locally;
no risk rule was weakened. No full suite, hosted CI, benchmark or fit was run.

The observation is not wired into the installed CLI, normal live Futures fee
accounting, inventory application or rearm. Metadata is caller-supplied and
scope-bound, not authenticated or freshness-qualified by this function. Every
financial/account/lot/inventory/rearm qualification flag remains false.
Partial observations never erase residual positions or resolve UNKNOWN.
Native Spot base-commission effects may exceed the recorded sale quantity;
that is a reconciliation obligation, not permission to consume another lot.

This advances the full-hedge native-component prerequisite, not a profitable
strategy, repaired stateful objective or qualified training dataset. No venue
request, credential, order, protected-data change, old-result rewrite or
automation occurred. Counts remain 202 observations / 65 hypotheses / 37 scoped
mechanisms / zero qualified stable profitable edges.
