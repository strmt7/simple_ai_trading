# September 10 TradFi funding change: counterpart gate

September 8, 2026. [Canonical result](result.json), [frozen contract](source-contract.json),
[retained source receipt](source-result.json).

The new [official Binance notice](https://www.binance.com/en/support/announcement/detail/aff73212559d4951945600e973240c52)
announces an eight-hour to four-hour funding interval change effective September
10 at 08:15 UTC. The per-settlement cap changes from 2% to 1%; the first listed
changed settlement is September 10 at 12:00 UTC. Automatic acceleration to hourly
funding upon hitting the cap is excluded for these contracts. These are future
terms, not evidence that the change is already active.

All nine announced siblings were checked together: MEITUANUSDT, KUAISHOUUSDT,
GIGADEVUSDT, POPMARTUSDT, TENCENTUSDT, HK1810USDT, HK0700USDT, ZHIPUUSDT and
MINIMAXUSDT. None has an exact `base_asset` label in the hash-bound 67-instrument
Polymarket snapshot captured earlier on September 8 at 10:27:29 UTC. This is a
snapshot-specific exact-label rejection, not proof that aliases are equivalent,
that all venues lack counterparts, or that a future listing cannot appear.
Even a matching label would require share-class, unit, quote and settlement proof.

The nominal full-day sum of absolute rate caps is unchanged: three settlements
at 2% versus six at 1%, both 6%. This is neither expected funding nor a fixed-base
cash-PnL bound: realized rates, mark notionals, basis, fees and margin can differ.
No positive carry inference follows from doubling settlement frequency.

## Decision and resource use

One new unauthenticated public source GET retained 54,035 bytes, with its intent
journal written before access. Today's existing instrument bytes were reused;
there was no new inventory, funding, price, book, account or order request.
The initial decoded-text display hit Windows cp1252 encoding, not source failure;
ASCII-escaped inspection reused the same retained bytes without a retry.

Rank 43 now records this separate terminal counterpart screen. Do not treat the
September 10 clock alone as a funding-study trigger for these nine contracts.
Require an independently observed exact counterpart listing or source-proved
conversion architecture, or a later material exact economic/settlement change,
then separately freeze the relevant qualification. CXMT and every previous
consumed study remain unchanged; no rolling retry or orientation reversal.

The other inspected funding branches did not justify another sample: Paradex's
prior gross role spreads failed even its execution-only allowance, and merely
moving a window is not an information-gain case. No new Paradex or Backpack
request was made. The MRNA gate remains not-before September 8 at 13:35 UTC.

Fourteen focused source/chronology/raw/registry checks pass, including a positive
control proving that an exact label does not authorize downstream access. Counts
remain 37 accepted mechanism scopes, 65 hypotheses and zero qualified stable
profitable edges; terminal observations increase from 194 to 195. This is new
source-bound financial routing evidence, not an additional accepted strategy.
