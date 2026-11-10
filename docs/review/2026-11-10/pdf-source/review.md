# Long-option terms: useful primary evidence, unresolved payoff identity

The distinct [official Options Contract Specifications](https://bin.bnbstatic.com/static/cms/cg08ou2ak0tn7mcplvfg/file/443bcc67fde7274898ff8e3f7af23c7d89e654c5e609d250772e0ddaa96409be.pdf)
passed the [frozen one-use source contract](contract.json): actual HTTP 200,
186,353 raw bytes, four complete native pages and 7,323 Markdown characters.
The document is dated January 5, 2026. It identifies RIE/RCH European options,
USDT settlement and unit size 1 for BTC/ETH/SOL. It delegates exercise,
settlement and fees to other documents; this is not account/API applicability.

The complete URL lookup then found the exact Clearing Procedures already
[retained on September 4](../../2026-09-04/inverse-clearing-source-review.md).
No second download occurred. All 52 retained pages were extracted offline with
native Crawl4AI; the original PDF hash stayed unchanged. Its effective version
remains unverified. The earlier inverse-PnL sign conflict stays unresolved.

## Findings that change the decision

- Procedure 127 prints Settlement Price for calls but Price Index for puts;
  Procedure 128 defines Settlement Price as a preceding 30-minute average.
  Visual checks of PDF pages 37-38 confirm the difference. Common-scalar payout
  identity therefore needs authoritative reconciliation, not a convenient typo fix.
- Procedure 191 prints an exercise-rate/cap of 0.015%/10%, with a bare `Unit`
  alongside `Contract Unit`. Current units, fees and account treatment still need
  qualification; these numbers were not installed as trading defaults.
- Procedures 162-170 permit some long liquidation after short losses and ADL
  against profitable opposing participants. Procedures 173-184 allow fee changes,
  debt and alternative-asset payment. Fully paid does not establish an
  unconditional protected-to-expiry cash balance.

The exact [positive-path controls](settlement-controls.json), reproduced by
[this script](settlement_controls.py), show why reference identity matters.
For strikes 100/300, one constructed positive index path has mean 100 and final
index 400: call-on-mean plus put-on-final-index pays zero, versus the incorrectly
assumed width floor of 200. Another has mean 200/final 210 and pays 190 instead
of 200. These are gross, supplied scenarios under one literal interpretation;
they do not describe verified exchange payouts, an exploit or market observations.
The 1,800-point sampling schedule is supplied, not a vendor-method claim.

## Native PDF reading and verification

The reviewed HTML CLI overwrites the PDF strategy. The new native PDF API
[worker](../../../../tools/crawl4ai_pdf_source.py) and
[owned controller](../../../../tools/capture_crawl4ai_pdf.py) retain actual HTTP,
raw bodies, complete pages, native success/errors and durable one-use journals.
They reject the native placeholder 200 as transport evidence. Robots policy is
qualified before the document; there are no redirects, retries, proxies or LLM.
The child has an owned hidden process deadline; no browser/service is installed.

The optional pypdf 7.0.0 wheel was hash-locked and installed into a new isolated
extension only. Base provider dependency/repair bytes passed read-only verification;
the shared provider and system packages were unchanged. The runtime manifest
binds every extension file and relevant native modules before access.

Eight actual supervised loopback controls pass: valid PDF, robots refusal/error,
HTTP 202, redirect, non-PDF body, page cap and blank text. Native failure JSON
is retained; the server stopped and temporary copies were removed. Earlier local
qualification failures remain private, including a version-label mismatch,
incorrect static-method import and isolated Windows pipe encoding. The final
source used only the corrected, hash-bound phase. This is scoped reader evidence,
not a trading-engine, whole-suite or benchmark claim.

Final offline review also added an EOF deadline check for future captures. The
original captured worker remains hash-bound and privately retained; that complete
worker session took under four seconds, below the 30-second body budget. No source
was repeated. The final worker revision passed the eight supervised fixtures again;
the result distinguishes capture-time code from the final reviewed code.

## Next decision and preserved boundaries

Require independent authoritative reconciliation of the put reference, fee unit,
effective version and product/account scope before venue-specific floor admission.
Then require a separately frozen genuinely new population, complete acquisition
cash, costs, capital alternatives, fills, capacity and recurrence. Long-only
account segregation alone does not eliminate venue ADL/custody exposure.

Do not refetch either PDF, retry failed FAQ aliases, relax consumed contracts or
reprice the 2,944 excluded option symbols. Preserve both original source phases,
all protected captures and the supplied-law evaluator unchanged. No metadata,
quotes, funding, credentials, accounts, orders, model training or GPU benchmark
followed. Counts remain **204 observations / 65 hypotheses / 37 scoped mechanisms
/ 0 qualified stable edges**. The broader R&D/application goal remains active.

The [source-bound result](result.json) distinguishes fresh capture, retained-source
reuse, exact arithmetic and the remaining qualification gates.
