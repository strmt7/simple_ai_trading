# Native opening inventory observations

The new `retain_opening_inventory` stage derives exact native movements from
the terminal evidence already retained in the scoped Binance intent journal.
It adds `opening_inventory` in that same SQLite database, not a competing
account ledger. It never requests data, trades, applies positions or rearms.

For a Spot buy, gross purchased base is reduced by any base-denominated fee;
quote spend includes quote fees, and third-asset fees/rebates remain separate.
For a linear futures opening, signed position quantity is separate from fee
cash movements: the contract notional is not a Spot purchase. A base-asset
futures commission therefore does not reduce the derivative's position size.

The concrete offline example is 0.002 BTC filled with a 0.000002 BTC Spot fee:
the execution contributes 0.001998 BTC, not 0.002 BTC, alongside its actual
quote outflow. A fee exceeding newly acquired base remains a negative movement,
not a fabricated positive lot. These are changes attributable to execution,
not account balances or proof of quantity currently available to sell.

A futures trade's signed delta also does not prove it increased gross exposure:
it could reduce a preexisting opposite position. Zero reported realized P&L
does not prove prior flatness. Account/ownership reconciliation must resolve
that before any active-position application or rearm.

The caller supplies explicit instrument metadata and matching execution scope.
BTC/ETH/SOL and USDT/USDC units are checked literally; futures additionally
require linear crypto-perpetual quote/margin units. This does not authenticate
the caller's metadata source, establish freshness or prove current eligibility.
Only admitted unit fields are stored; arbitrary extra metadata is excluded.

## Durable boundary and evidence

The writer rechecks the exact UNKNOWN intent and reconstructs terminal fill
evidence inside a serialized SQLite transaction. Repeated calls return the same
observation, not additive duplicate balances. Conflicting native records, damaged
terminal evidence, changed intents, unknown schemas and write failure reject.
UNKNOWN remains intact, including a zero-fill terminal order.

190 distinct affected checks passed across staged runs, including 46 new cases.
They cover Spot, futures long/short, all declared asset/quote units, partial and
zero terminal fills, fees/rebates, low caller precision, restart, four concurrent
writers, rollback and an actual test-owned child exiting abruptly with code 71.
The [source-bound evidence record](binance-native-opening-inventory.json) lists
the exact verification stages. No user ledger, account, credential, protected
capture or live endpoint was accessed. Ruff passes.

Next integrate these observations into atomic active-position/native-fee/close
accounting, then explicit recovery/rearm under current account, policy and process
fences. The legacy automatic fill adapter is not changed by this stage and must
not be described as using net native inventory yet. CLI/Windows integration,
legacy migration, missing-file detection, supervision and reboot protection
remain unfinished. Do not repeat this unchanged test domain as market research.

The parallel completion-source review found no new capture authority: retained
CTF mint/merge matching support does not identify public quote levels as the
same underlying orders. Do not infer shared-liquidity identity or independent
capacity without order-level evidence. No prior economic outcome was changed.
