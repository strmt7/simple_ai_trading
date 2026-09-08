# Terminal closing evidence without replay or rearm

September 8, 2026. [Source-bound checkpoint](binance-terminal-closing-evidence.json).

The preceding durable-close repair prevents another order while an earlier
close is uncertain, but did not collect its later terminal executions. This
checkpoint adds that evidence path. It is not native inventory application or
permission to clear the unresolved obligation.

## Exact ownership and execution semantics

`collect_closing_recovery` requires an exact closing client ID and an existing
scope-bound UNKNOWN close intent. It reconstructs and validates the original
owned position template and request, including product, configured key identity
and reduction scope. It does not adopt current settings as legacy ownership.
Invalid bot close IDs now share one pre-persistence/pre-recovery validator.

One exact-client-ID order query must establish FILLED, CANCELED, EXPIRED,
EXPIRED_IN_MATCH or REJECTED before trade access. PARTIALLY_FILLED is not terminal.
A nonzero execution then requires one exact-order trade page, at most 1,000
rows, whose unique trade IDs, times, sides, exact quantities and quote totals
reconcile. Zero execution needs no trade query. The existing bounded transport
GET retry policy is unchanged: these are logical query bounds, not a promise
of one or two HTTP transmissions. No quote, balance or order-submission method
is called by this collector.

The shared execution validator takes an explicit closing client ID, not a
position with falsified opening fields. Spot long closes are sells; one-way
futures long/short closes require the opposite side and reduceOnly true. The
original opening exchange ID is not mistaken for the closing exchange ID.
Default opening semantics and its persisted evidence shape are preserved.

Native commissions and rebates remain exact by asset. Signed futures realized
PnL is summed with bounded exact decimal arithmetic and retained separately.
It is venue-reported PnL, not proof of this managed lot's PnL, a known margin
currency, a balance change or account flatness. Instrument/margin identity,
preexisting account inventory and attribution still need reconciliation.

The original close intent quantity remains the comparison target after a
partial fill reduces local inventory. A test records a first 0.001 fill against
a 0.002 intent, then observes a terminal cumulative 0.0015; collection leaves
the existing active/closed pair unchanged. The eventual applier must apply only
the verified incremental movement, including native-fee differences, exactly
once. It must not replay the entire cumulative fill or fabricate zero fees.

## Persistence and shared retention

The same intent database stores an additive `closing_recovery` row with exact
request binding, validated order/trade fields and normalized evidence. Before
commit, it rechecks the still-pending request under the write lock. Identical
concurrent observations produce one row; conflicts cannot overwrite it.
Reopening a retained row revalidates it without refetching. Invalid schemas,
duplicate identities, conflicting requests or corrupt cached evidence reject.
An actual child exit after retention leaves both the observation and UNKNOWN.

Opening and closing collectors now use one product-specific field encoder.
Only fields validated for the actual product are retained; unrelated response
fields and the other product's fields are excluded. This avoids duplicating the
earlier product-allowlist mistake. Both products' output was compared byte for
byte with the baseline opening projection, and opening/native-inventory checks
were rerun after extraction. Historical sources and results were not rewritten.

These are allowlisted validated observations, not byte-complete HTTP archives.
Failed/nonterminal response capture, receipt-time provenance, coordinated
database rollback/loss detection and authenticated current-account evidence
remain unfinished. Cached evidence is non-authorizing even if another worker
changes the intent after a read. Any future applier must revalidate it inside
its own fenced transaction.

## Verification and next action

414 distinct affected checks passed across stages, including 75 new closing
cases. A 414-case domain run covered durable closes, opening/closing recovery,
native opening observations, execution scope, paired storage, lifecycle and
autonomous behavior. After shared-encoder extraction, its 203 affected cases
passed again. Coverage includes Spot/futures long/short, canceled partials,
native asset fees and rebates, a 30-decimal signed PnL residual, malformed or
oversized pages, scope rotation, concurrent conflicts, interrupted persistence,
cached corruption and actual child `os._exit(74)`. Ruff passes. Test durations
are not benchmark evidence; no unrelated workstation process was controlled.

All venue interactions used offline fakes. No actual account, credential,
user ledger or protected research capture was touched. No full-repository
suite, training or repeated market sample was launched for this checkpoint.
The financial registry and every protected/retry boundary remain unchanged.

Next integrate native opening movements and incremental terminal closing
movements through the authoritative paired store, with current account,
instrument, policy and process-generation evidence and explicit rearm. The
collector is not automatically called by the trading loop and does not grant
authenticated-operation authority. CLI/Windows integration, independent
supervision, final comprehensive review and profitable-edge qualification remain
open. Preserve prior source bindings; do not regenerate them against this code.
