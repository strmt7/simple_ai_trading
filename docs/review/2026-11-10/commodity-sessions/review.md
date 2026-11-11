# Commodity option hedge and perpetual session basis

The retained [Binance clearing notice][notice] applies to XAUUSDT and XAGUSDT
perpetuals from September 15, 2026 at 21:00 UTC. It specifies a once-per-second,
vendor-constituent weighted price index in the regular session and order-book
EWMA on weekends and public holidays. The perpetual remains tradable across
the mode change. The stated weekend runs Friday 17:00 to Sunday 18:00 Eastern
time. Under that regular schedule, the October 30–November 1 clock change
produces 50 elapsed hours from Friday 21:00 UTC to Sunday 23:00 UTC. The
following Friday switches at 22:00 UTC. Holiday exceptions and how the exact
boundary sample is assigned require separate evidence.

The [historical option contract](../../../model-research/action-value/binance-commodity-option-perpetual-lower-bound-contract-v1.json)
specifies a 17:00 Eastern option settlement clock. At a Friday expiry this
nominally coincides with the perpetual's index-method switch. The notice
does not supply the option settlement sample or prove the perpetual exit is
available at that price. A fixed 20-bp exit stress in the old screen remains
a historical sensitivity. A hard payout floor needs an independently
supported bound for the relevant side, quantity, exit deadline and session.

Let `S` be the option settlement price, `K` its strike and `C` a
conservative per-unit upper bound on premium, fees, adverse funding and
other modeled costs. For a fully filled equal-base hedge that survives to
expiry, let `B` be the perpetual entry bid and `A` the entry ask.

- A long call and short perpetual have conditional lower bound
  `B - K - C - U`, where `U` bounds the executable buy-back ask minus
  `S` from above.
- A long put and long perpetual have conditional lower bound
  `K - A - C + L`, where `L` bounds the executable exit bid minus
  `S` from below.

The inequalities follow from `max(S - K, 0) - S >= -K` and
`max(K - S, 0) + S >= K`. A constructed settlement and strike of 100 USDT,
cost of 3 USDT, and entry of 110 short or 90 long yield +7 USDT if the exit
equals settlement. An exit ask of 108 for the short or exit bid of 92 for the
long yields −1 USDT instead. Entry-leg mismatch, margin and liquidation
survival, cash availability and failover remain separate conditions.

The [evidence](evidence.json) binds one exact native CMS response and the
offline arithmetic. This historical method change warrants a source check
before future commodity screens. It supplies no active expiry, matched
settlement sample, executable basis bound, funding history, account
qualification or new economic observation. Prior population, costs and
retry rules stay intact.

[notice]: https://www.binance.com/en/support/announcement/detail/f69c05ebdba7422e9c614685abe84dc5
