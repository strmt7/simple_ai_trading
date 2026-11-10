# Scoped Spot diagnostic

`spot-roundtrip` uses the paired durable journal and enrolled Binance runtime
control. It supports BTC/ETH/SOL against USDC/USDT on official testnet/Demo only.
It is an execution diagnostic, not a profitable strategy or a flat-inventory
guarantee. Native account/order use still needs its separately authorized session.

## Authority and lifetime

The session owner must already have enrolled the control store, reconciled its
owned obligations, enrolled the execution journal and obtained a valid running
lease. It supplies that lease to
the command through `SIMPLE_AI_TRADING_BINANCE_RUNTIME_LEASE`. Never print,
persist or recover its value from logs. The command does not acquire or release
the lease, clear Stop, enroll control storage, or install a worker or restart
mechanism. Its owner remains responsible for heartbeat and session shutdown.

The independently supervised Binance session owner is not yet integrated into
the app. Missing authority rejects locally; this document is not an enrollment
or rearm shortcut. Follow [control storage enrollment](RUNTIME_CONTROL_STORAGE.md).
An enrollment marker or empty wallet response is not proof of flat exposure.

Each command owns and closes its HTTP session, including on admission failure,
lost responses and cancellation. No process or thread is created. Request-level
timeouts/freshness checks do not replace the pending independent hard-deadline
supervisor. Do not infer application-wide crash or reboot protection from this
command integration.

## Execution storage setup

Before acquiring an opening lease, inspect the shared execution directory and
explicitly enroll its journal while the controller has no owner and Stop is
latched. These actions are local: they send no account or order request, do not
clear Stop and do not establish flat exposure. Setup constructs and closes one
client solely to validate the configured testnet/Demo scope.

```text
python -m simple_ai_trading.entrypoint spot-roundtrip --action inspect-storage
python -m simple_ai_trading.entrypoint spot-roundtrip --action provision-storage --yes
```

An existing journal requires the exact `journal_snapshot_sha256` returned by
inspection, supplied as `--expected-snapshot-sha256`. This is explicit adoption,
not permission to migrate a foreign or unscoped live ledger. Changed snapshots,
already assigned UUIDs and existing markers reject. Adoption preserves pending
obligations; it neither reconciles native exposure nor authorizes a new order.

The independent marker defaults beside the journal. `--execution-enrollment`
selects another explicit location; keep it consistent across setup and execution.
Preserve it along with the journal and existing controller interlocks. Normal
opening rejects missing/replaced journal files, missing/changed UUIDs, unfinished
or missing markers and different enrolled interlocks. It never recreates them.

If interruption left a **committed** prepared enrollment, inspect it and use
`--action recover-storage --expected-store-id <exact-id>
--expected-record-sha256 <exact-digest> --yes`. Both values must match the prepared
record, and its committed UUID, scope and resources must still verify. An
uncommitted or conflicting preparation stays blocked for reviewed restoration;
do not delete its marker or provision over it. Coordinated rollback of all local
authority remains outside this protection.

## Explicit allocations

Use exactly one amount: `--quote-budget` for a BUY followed by SELL, or
`--quantity` for a SELL followed by BUY. Inputs remain decimal strings. `auto`
resolves these explicit units; it never selects inventory from wallet balances
or converts a price estimate into a spending allowance. The first allocation
is never enlarged to meet an exchange minimum.

Examples within an already authorized, enrolled session:

```text
python -m simple_ai_trading.entrypoint spot-roundtrip --quote-budget 10.00 --mode buy-sell --yes
python -m simple_ai_trading.entrypoint spot-roundtrip --quantity 0.001 --mode sell-buy --yes
```

The native instrument must report matching assets, active MARKET and quote-BUY
capability, quote precision and usable quantity filters. The second principal
is limited to the first leg's net received asset. Native lot/quote rounding can
leave dust; it cannot increase the receipt-owned amount. Other exchange filters,
including price-dependent eligibility, can still reject an order. An ambiguous
failure remains UNKNOWN rather than being treated as a safe retry.

Account trading permission, exact free balances and the scoped fee schedule
are checked before dispatch. This command supports received-asset commissions;
eligible external fee routes reject pending native conversion/cost admission.
Point-in-time free balances do not prove ownership, atomic reservation, future
fees, financing or profitability.

## Status and restart

`--data-dir` selects the shared owned execution directory; the default is the
application-relative `data/autonomous`. `--runtime-control` and
`--runtime-enrollment` configure the existing control and independent marker.
Keep these locations consistent across the session. Local status needs neither
credentials nor a client and creates no missing files:

```text
python -m simple_ai_trading.entrypoint spot-roundtrip --action status
```

Every new operation gets a durable 29-character ID; `--operation-id` can supply
one prospectively. Retain that ID for restart. Do not select a new ID to bypass
an unresolved operation. The following actions use its stored allocation, not
new amount flags:

| Action | Behavior |
| --- | --- |
| `recover` | Observe the exact UNKNOWN leg and complete trade evidence; never submit, complete or rearm |
| `continue` | Submit the remaining leg only after a fully reconciled first FILLED order, or finish local completion after both orders |
| `complete` | Complete only two already reconciled FILLED orders; send no order and leave Stop/lease unchanged |

Use `--action recover`, `continue` or `complete` with the exact
`--operation-id` and `--yes`. A receipt-owned second SELL can use closing
authority during opening Stop. A sell-buy second BUY remains an acquisition
and needs the active opening lease. Lost controller storage does not authorize
fresh setup; loss of the enrolled closing lock rejects without recreation.

`execute` and `continue` finish local completion automatically when both exact
orders are reconciled FILLED. Repeating an action on a completed operation sends
no new order. Native not-found, partial/canceled execution, incomplete fees,
missing namespace or contradictory evidence never authorize a replacement
write or automatic rearm. Unsupported partial resolution remains blocked.

Output distinguishes observed movements from complete net asset movements.
UNKNOWN output leaves total movements missing; it does not report zero cash or
zero PnL. `COMPLETE` means the two orders reconciled, not restored initial assets,
flat risk or profit. Execution-store loss/replacement now blocks paired opening;
coordinated rollback resistance, full writer fencing, verified relocation and
native capital qualification remain open. Recovery and the receipt-owned second
SELL do not require this opening enrollment; they retain their exact journal,
scope, receipt and closing-interlock checks. A second BUY requires it.
