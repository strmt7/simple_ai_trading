# Round 27 mechanics diagnostic

This target-free screen used 11 BTC five-minute markets and 54,983 synchronized quote states from the condition-eligible portion of Round 26 v2. Exact message batches were fully applied before evaluating Up and Down together.

Four after-fee complete-set episodes remained. None survived the recorded 250 ms taker delay, and none survived the optimistic 500 ms sequential two-leg floor. The best delayed cost was 1.012066 pUSD; the best sequential cost was 1.002744 pUSD per complete set, before network or order-response latency.

Extreme-price and late-favorite states occurred often enough for prospective model-value testing. Public quotes cannot prove maker fills, queue position, or profitability.

![Round 27 mechanics diagnostic](mechanics-diagnostic.svg)

The [canonical JSON](mechanics-diagnostic.json) and [latency table](complete-set-latency.csv) are hash-bound source data. This superseded cohort predates the preregistration and is not promotion eligible. No market edge or after-cost profitability is claimed.
