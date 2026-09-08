# MRNA: negative entry headroom, no escalation

[Frozen contract](contract.json), [canonical result](result.json),
[source classification](../mrna-source-classification.md),
[registry amendment](registry-amendment-plan.json).

The separate rank-12 September 8 13:35 UTC exception selected MRNA before
prices, after CRWD's earlier rejection. The retained exact-one inventory
contains CRWD, MRNA and SQQQ equity TradFi matches in that order. No current
MRNA price was inspected before the freeze. Exact canonical URL/method searches
over retained contracts and journals, including ignored data, found no prior
matching request. The existing bounded public transport was reused unchanged;
neither the old CRWD runner nor its results were edited.

Two public unauthenticated GETs began at 13:37:15.995 and 13:37:16.451 UTC,
456 ms apart. Both returned HTTP 200; raw bodies are 118 and 150 bytes.
Exclusive fsynced intents, complete bodies and receipts are retained under
this directory. No redirect, retry, proxy, key or signed request was used.

| Frozen long-bStock / short-perpetual screen | USDT per underlying share |
| --- | ---: |
| MRNABUSDT displayed spot ask | 141.23 |
| MRNAUSDT displayed perpetual bid | 140.87 |
| Gross entry headroom | -0.36 |
| Fixed 50-bip stress on spot ask | 0.70615 |
| Stressed entry headroom | -1.06615 |

Displayed entry quantities were 0.938 bStock shares and 1.14 perpetual shares;
their common displayed size is 0.938, not a guaranteed executable capacity.
The stressed subtotal is approximately -75.4903 bips of spot acquisition cost.
No direction forecast was needed for this arithmetic.

## Financial meaning

The observation fails even before fees or financing: the acquisition leg costs
more than the short entry. Positive future funding could in principle offset
an entry deficit, but none is credited or observed under this fixed screening
contract. A negative entry screen is not proof that every holding horizon,
funding strategy, maker policy or future MRNA state has negative expected value.
The 50-bip stress is a chosen screening hurdle, not a measured fee schedule.

Request timing does not prove internal quote freshness, simultaneous execution,
or last book update. Futures `time` is transaction time. The September 4
inventory establishes historical exact-one identity, not current corporate
actions or account eligibility. A positive prefilter would still need fresh
contract/multiplier, conversion, depth, adverse funding, exit basis, fees,
capital, margin/liquidation, tax and account qualification. None is bypassed.

This exact MRNA observation is terminal. No depth, funding, fees, account,
credential, order, funds, wallet or protected-data access followed. SQQQ is
unscreened, not a fallback and not inferred negative from MRNA. Reopening
requires the remaining literal rank-12 material/source/authority trigger.

## Verification and continuity

Before capture, 34 focused offline checks passed, including parsing both
hash-bound retained CRWD raw responses through the new production decoder,
exact threshold/quantity/skew cases and the unchanged bounded transport.
After capture, four additional checks cover exact zero-network reconstruction,
prospective freeze and complete journals, refusal to repeat a consumed capture,
and registry/audit lineage. No training or performance benchmark was run.

The registry amendment verifies prior Git checkpoint
`c0dfb1d99260b8dd9cb3d5dfe98575e73326ac8c`, both preceding canonical ledger
hashes, exact terminal count, new source bindings and full reconstruction.
Before writing, it proves all other JSON values unchanged by reversing only
the declared edits in memory. The prior rank-12 status/next action also remain
in its dated history field. Current totals: 37 accepted mechanism scopes, 65
hypotheses, 197 terminal observations and zero qualified stable profitable edges.
Native-fee application, fenced recovery/rearm, independent supervision and the
final comprehensive code review remain unfinished; this checkpoint claims none.
