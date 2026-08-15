# Round 27 settlement mechanics

The target-access claim was persisted before querying any result. All 53 eligible BTC five-minute markets were terminal on both official public sources, and Gamma and CLOB agreed on all 53 winners. The observed labels were 26 Up and 27 Down.

![Round 27 settlement mechanics](settlement-mechanics.svg)

The [canonical audit](settlement-mechanics-audit.json) and [per-market labels](settlement-labels.csv) bind every row to the two raw payload hashes and an evidence hash. Raw public receipts remain in the compact local evidence database. This validates settlement mechanics only: Stage 0 is not model-fitting data and makes no edge or profitability claim.
