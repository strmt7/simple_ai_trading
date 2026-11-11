# Binance TradFi perpetual fee supersession

Binance's October 28 notice planned to cut the VIP 4–9 taker discount from
50% to 40% on November 5 at 08:00 UTC. Its November 4 notice superseded the
planned change before that instant and stated that the existing 50% discount
would continue until further notice. Regular through VIP 3 retained the
existing 20% discount. Both notices displayed zero maker fees for the
ordinary promotion tiers, with separate market-maker and taker-program
terms. [October 28 notice][original] · [November 4 notice][replacement]

Applying the cancelled table would overstate advertised non-BNB fees by
0.17–0.30 basis points per side for VIP 4–9. At two equal notionals the
overstatement is 0.34–0.60 basis points in total. For example, two VIP 4
taker fills of 1,000,000 USDT each would be overstated by 60 USDT.
Keep the cancelled schedule as evidence of what was announced and use
the continuing schedule for advertised fee replay. Account commissions,
rebate-program entitlement and exact fee receipts require separate evidence.

The [replay](derive.py) reconciles all ten publisher table rows against
[retained source digests](evidence.json). The cancellation preserves a
revocable fee term. It supplies no incremental fee reduction against the
prior effective schedule and does not reopen a consumed funding or price
population. No trading-engine constants or new economic observations follow.

[original]: https://www.binance.com/en/support/announcement/detail/4f406308840549ffb8fdaac9873c25a8
[replacement]: https://www.binance.com/en/support/announcement/detail/e3424d3a781841eb8b507bc694c0ca3a
