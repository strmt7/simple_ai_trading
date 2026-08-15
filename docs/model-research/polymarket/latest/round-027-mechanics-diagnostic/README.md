# Round 27 mechanics diagnostic

This target-free screen used 53 BTC five-minute markets and 268,393 synchronized quote states from the preregistered Stage 0 cohort. Exact message batches were fully applied before evaluating Up and Down together, and each condition was replayed independently with bounded memory.

The same-state screen found 6 after-fee complete-set episodes. 0 survived the recorded venue delay and 0 survived the optimistic two-delay sequential floor. The best delayed cost was 1.014726 pUSD; the best sequential cost was 1.002034 pUSD per complete set, before network or order-response latency.

Extreme-price states occurred in 53 markets and late-favorite states in 44 markets. These are candidate observations, not trades or an edge. Public quotes cannot prove maker fills, queue position, settlement value, or profitability.

![Round 27 mechanics diagnostic](mechanics-diagnostic.svg)

The [canonical JSON](mechanics-diagnostic.json) and [latency table](complete-set-latency.csv) are hash-bound source data. Stage 0 permits mechanics screening only and is not promotion eligible. No market edge or after-cost profitability is claimed.
