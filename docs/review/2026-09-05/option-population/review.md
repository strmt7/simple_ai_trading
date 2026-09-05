# September 5 distinct option population: no floor survivor

One prospectively frozen metadata gate proved **70 genuinely new BTC/ETH/SOL
option contracts**, outside the union of 2,274 previously retained names.
Of 1,436 currently eligible names, 1,366 were already known. One separately
frozen two-request price screen found 23 positive-entry rows, zero positive
gross floors and zero survivors after the unchanged 33.5-bip stress. The other
47 rows have zero entry quotes; they are incomplete, not free options.

Exactly three public unauthenticated GETs were used. Complete responses,
intent/completion journals, source receipts, implementation hashes, population
and all 70 economic rows are retained beside this review. No depth, funding,
fee, account, credential, order, fund or protected capture access followed.
The request-start skew was 1,984 ms within the frozen 10,000-ms gate; receipt
timing does not establish internal quote freshness, depth or executable size.

The idealized per-unit call floor is `perpetual bid - strike - option ask`;
the put floor is `strike - perpetual ask - option ask`. Among eligible rows,
the best floor after stress relative to perpetual entry was
`BTC-260908-79500-P`: strike 79,500, option ask 595, perpetual ask 79,662.30,
gross floor -757.30 USDT/BTC and fixed stress 266.8687050 USDT/BTC, leaving
-1,024.1687050 USDT/BTC. This is a modeled unit payoff, not a realized loss.
Funding, settlement-basis mismatch, collateral and executable quantity are
not proved. The stress is a rejection hurdle, not an actual account fee.
A negative lower bound is not proof of negative expected value; this snapshot
neither qualifies an edge nor rejects every future member of the family.

## Implementation and provenance

The new generic population gate verifies all six prior populations before
access and subtracts their complete union. The new generic price screen keeps
all frozen symbols, checks contract unit/strike/expiry, uses the acquisition
side and conjoins row and run-level gates. Old fixed-population runners and
results are preserved. The forward transport persists bounded chunks, retains
HTTP failures, refuses redirects/retries and writes durable source receipts.
Its 30-second read budget is checked between reads, with a 10-second socket
timeout; it is not a strict 30-second end-to-end deadline.

The frozen price contract's phrase "strict positive gross less fixed stress"
is awkward prose: its numerical rule and tested implementation require
`gross - stress > 0`. Neither the consumed contract nor its result was edited.

The original public baseline
`data/binance-commodity-option-perpetual-lower-bound-v1/raw/options-exchange-info.raw`
was present locally but untracked. Publication includes its unchanged bytes
(SHA-256 `81a8cb419821562c96537ec310e7043f40e2a63c7e96fb33d86a225f13b26449`)
so the frozen exclusion union reconstructs from Git. No historical result is
regenerated. All new JSON/JSONL and that baseline have exact-byte Git attributes.

Verification: all 37 focused tests pass, covering transport, exclusion-union
and price-screen branches plus zero-network artifact reconstruction. Ruff
passes on the seven new Python files. No GPU workload or broad CI is
needed for this bounded screen; no unrelated process was modified.

## Routing

Both stages and all 70 contracts are consumed. Do not poll, repair missing
quotes, reprice or escalate them. Future distinct populations must exclude
these 70 in addition to every earlier union member. Rank 47's literal retry
trigger is unchanged. The registry amendment changes only this branch and
its audit binding: 37 accepted mechanism scopes, 65 hypotheses, 192 terminal
observations, **zero fully qualified stable profitable edges**. These counts
are bookkeeping, not evidence that more economically independent strategies
have been discovered. Polymarket protected partial captures remain untouched;
that restriction does not stop unrelated eligible R&D.
