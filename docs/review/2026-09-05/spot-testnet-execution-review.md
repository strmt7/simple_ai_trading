# Spot-testnet owned execution checkpoint

This is substantive virtual exchange execution, not simulated fills and not
evidence of a profitable strategy. The consumed campaign completed on September
5, 2026, at 02:37:43 UTC. No mainnet, futures or Polymarket orders were placed.

## Outcome

Across BTCUSDT, ETHUSDT and SOLUSDT, the campaign submitted nine orders: three
resting LIMIT_MAKER buys, three filled IOC buys, and three filled IOC sells.
Three exact-owned cancellations were acknowledged and subsequently queried as
CANCELED with zero executed quantity. Six owned trades reconcile exactly.
Final per-symbol queries showed no open orders; acquired campaign base was zero
for each asset. This does not assert that the account has no unrelated holdings.
There were 48 GETs, nine POSTs and three DELETEs, with no mutating-request retry.

The total virtual quote cash delta was **-0.01019850 USDT**. All observed
testnet commissions were zero. These round trips were lifecycle probes, not an
edge strategy, and cannot estimate mainnet costs, fill quality or profitability.

The canonical [original result](spot-testnet-execution/result.json) deliberately
retains `required_live_cases_passed=false`. The original reporting predicate
expected the cancellation client-ID suffix, while the exchange's later order
queries retained the original client ID. Numeric order IDs, successful DELETE
receipts and terminal queries agree. The corrected
[offline adjudication](spot-testnet-execution/adjudication.json) confirms all six
required coverage flags for all three assets. It independently rebuilds cash
from the normalized owned fills, then checks the original journal and result.
It makes zero network requests and adds no market observation or accepted edge.

## Preservation and forward behavior

The [plan](spot-testnet-execution/plan.json),
[hash-chained journal](spot-testnet-execution/journal.jsonl), and original result
are unchanged. The exact executed runner is preserved at
[frozen-runner.py](spot-testnet-execution/frozen-runner.py); its SHA-256 remains
`68456cff03d68b94dda7fc44be71cd6d00def9dc50e9ecfe6fc9ed382300fdaa`.
[Source disposition](spot-testnet-execution/source-disposition.json) explains
the forward correction without relabeling the consumed implementation.
The plan also bound the API client's local CRLF bytes, whereas Git stores LF.
`frozen-api.py.base64` retains those exact bytes; the verifier decodes and
checks their original hash without altering the unchanged tracked client.

`tools/review_spot_testnet_execution.py` verifies the original implementation
bindings before reconstructing evidence. `spot_testnet_coverage.py` now requires
stable numeric order identity and an acknowledged exact-order cancellation,
not a particular presentation of the client ID. Forward runner invocation
rejects the consumed plan's old implementation hash; it must not be rerun from
these artifacts. The completed process has exited; no recovery is required.

Thirty-three focused offline tests passed, including both cancellation-ID
representations, accepted-order/timeout reconciliation without resubmission,
partial-fill base-fee accounting, foreign-order/inventory denial, inconsistent
evidence rejection, journal tampering and retained-ledger reconstruction.
Synthetic checks remain distinct from observed exchange coverage.

## Limits that remain explicit

- No partial fill occurred on the exchange. Synthetic partial-fill coverage
  does not establish live partial-fill recovery.
- Acknowledgement bodies were deliberately ignored, not dropped by the
  network. The thrown-after-acceptance timeout was tested only offline.
- The durable journal was reloaded while orders could rest; a cold process
  restart, workstation failure, concurrent campaign or real disconnect was not
  tested. This isolated runner is not integrated production recovery.
- Public configuration and private owned responses were normalized, not stored
  as origin HTTP bytes. Private account bodies and foreign order details were
  not retained. Hash chains prove local integrity, not exchange signatures.
- Static quantity, tick and notional checks do not implement every dynamic
  exchange filter. The frozen recovery path still requires zero testnet fee
  schedules and can refuse changed conditions. Do not call it universal recovery.
- Keys were used transiently through masked input, never persisted by the
  campaign. Conversation/tool-input records are outside that claim. Rotate
  credentials shared in conversation; process exit is not secure-erasure proof.

## Why this advances the economic work

Round 58's equal quote notionals and full-fill flags cannot prove a flat,
after-fee maker position. This checkpoint provides a small owned-fill ledger
with exact base and quote cash reconciliation to support the forward fix.
It does not retroactively repair historical labels or justify training on them.
The next economic work should use explicit partial quantities, fees, orphan
inventory and executable liquidation alternatives before fitting a maker model.
Keep this ledger separate from carry capital requirements and from Polymarket's
conditional-token settlement/ownership rules; their accounting is not identical.

No GPU benchmark or training was required for this arithmetic. No unrelated
processes were modified. Canonical counts stay 37 accepted scoped mechanisms,
65 ranked hypotheses, 189 terminal market observations, and zero stable current
account-qualified after-all-cost edges. Protected captures and retry triggers
are unchanged. This is a verified component, not an enterprise-readiness claim.
