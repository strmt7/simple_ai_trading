# Available cash after quote reservations

Fill-only prefunding can accept a supplied path even when other unfilled orders
consume its available quote cash. `spot_reserved_cash` now composes the existing
unchanged fill accounting with an explicit declaration of maximum remaining
quote reservations at every fill prefix. It is a research capital constraint;
it neither reserves live funds nor changes order submission authority.

For cumulative net fill cash `D_i`, buffer `B`, supplied initial quote cash `C`
and declared maximum remaining lock `L_i` while that prefix's cash is constant:

```text
available_quote_lower_bound_i = C + D_i - L_i
minimum_initial_quote_cash = B + max_i(L_i - D_i)
```

Include prefix zero, the final prefix, and every reservation peak between fills,
including an interval with no fill. Each reserve declaration must bind the exact
spot-input digest and a source-body digest. Missing intervals, wrong path binding,
negative/coerced amounts and mutable lists are rejected. Digests alone do not
prove native origin, coverage or actual lock timing. Unknown lock amounts cannot
be silently filled with zero. A cancel request alone does not prove release.

Remaining locks are distinct from original order totals. For a 100-unit buy,
a 40-unit partial fill followed by a remaining 60-unit lock still requires 100
units, not 140 or 200. Releasing a lock changes availability without adding cash
or income. Fees enter existing fill cash once; base commission changes received
inventory without an invented second quote debit. Derivative wealth or funding
never supplies spot-wallet credit through this calculation.

| Synthetic same-unit declaration | Required initial quote, zero buffer |
| --- | ---: |
| Buy for 100, later sell for 120, no locks | 100 |
| Same fills, peak pre-fill locks 160 | 160 |
| Buy fills 40 then 60, remaining locks 100 then 60 then zero | 100 |
| Buy for 100, another 80 remains locked afterwards | 180 |
| No fills, 75 remains locked | 75 |

The result retains its fill-only comparator, available-cash bounds, binding
prefix, additional initial cash and increment over fill-only capital. It reports
the exact declared scenario. If maxima are conservative upper bounds, the cash
criterion is conservative too; it does not prove the minimum native requirement.
Immediate fill settlement, no non-fill wallet cash changes and complete reserve
maxima are explicit conditions. Native reservation evidence, delayed proceeds,
financing, third-asset fees, derivative margin and continuous solvency remain
unqualified. All native/solvency/training/edge qualification flags stay false.

The existing live Polymarket reconciliation separately blocks new exposure while
any owned order remains unresolved. This work does not claim that the live path
lacked that protection or that a source declaration replaces it. Concurrency and
supervisor/recovery correctness require their own complete review.

Focused verification covers peak locks without fills, partial fills, release,
fees, cash-shift invariance, zero-lock equivalence, invalid coverage and source
binding. The old seven-control prefunding report reproduces byte-for-byte;
its modules, frozen intent and historical results are unchanged. Exact sources,
synthetic controls and test results are in [verification](verification.json).
No new market observation, account call, capture, fit or order occurred. Financial
counts remain 203 observations / 65 hypotheses / 37 scoped mechanisms /
**0 qualified stable edges**. This closes a research-accounting omission, not the
whole native capital or profitability qualification.
