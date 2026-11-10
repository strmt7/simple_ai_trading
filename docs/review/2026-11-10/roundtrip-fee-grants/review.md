# Explicit roundtrip commission-asset grants

The journal can now retain explicitly allocated third-asset units, including
BNB, without assigning ownership of an entire wallet. Grants are immutable,
bounded, sorted and unique; base and quote retain their existing allocation
fields. Amounts stay exact. No exchange rate, future fee amount, current balance,
account eligibility or asset-ownership certificate is inferred.

Each observation debits its actual commission asset. Both legs share one grant;
the second leg cannot reuse the first leg's spent fee units. Split fills can
contain both a discount-asset debit and received-asset fallback. Their base/quote
fees still reduce the net receipt that caps the second principal. Missing,
foreign or excessive positive fee debits cannot clear UNKNOWN or release opening
admission. They remain reconciliation obligations.

Records with extra grants use payload format 2. Existing format-1 records keep
their exact six request fields and remain readable; absent grants are never
manufactured. Older strict readers reject the new format instead of interpreting
its fees as free. The paired execution namespace remains version 4; no actual
user ledger was migrated.

All **435** affected checks pass in one final lane, including both orderings,
restart, two-leg grant exhaustion, mixed payment assets, legacy format and
malformed stored grants. The source was copied before launch; unchanged code
and tests bind to the base commit. The [result](result.json) retains that exact
phase. Earlier 407-case journal and 230-case fee-reader reports retain their own
source phases; overlapping counts are not added into an imaginary full run.

This records reconciliation limits; it does not enforce a fee-asset cap at the
exchange. An actual debit above the grant may already have occurred. Native
balance/ownership qualification, fee/FX precision and freshness, bounded dispatch,
receipt reconstruction, partial recovery/rearm, independent storage identity
and session supervision remain open. COMPLETE means two reconciled filled
orders, not restored wealth or profitable cash after FX. The CLI roundtrip is
still unchanged and unwired to this journal and the shared runtime authority.

No account, order, credential, model fit, market sample, benchmark or persistent
worker was used. Changes remain local for the next substantive batch. Financial
counts remain **204 observations / 65 hypotheses / 37 scoped mechanisms / 0
qualified stable edges**. This is capital-accounting work, not edge acceptance.
