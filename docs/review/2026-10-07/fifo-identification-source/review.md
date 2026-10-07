# FIFO ambiguity: qualify the execution advantage

The new primary [paper v2](https://arxiv.org/html/2609.13597v2), Danait, Zamora
and Boier, revised September 17, was retained through two frozen public document
GETs on October 7. [Source and repair evidence](review.json) binds the exact
version, local-only bodies, public receipts/journals, source change and tests.
Abstract and Sections 1–10, including the Proposition 3.1 sketch, were reviewed;
author code, proprietary data, independent proof and JAX replay were not replicated.

Aggregate books can admit multiple FIFO histories. The paper varies cancellation
allocation while holding common inputs fixed. Its tagged-fill ordering is
conditional on one touch-price spell and ordered ahead cancellations; its
compiler is a subset of compatible histories, not universal profit bounds.
Experiments use Tokyo equities with zero fees/latency and price-taking shadow
orders. Those fill rates, hardware and execution savings do not transfer as
Binance/Polymarket evidence. More favorable fills need not improve inventory PnL.

Repository audit confirmed a separate bug: the shared virtual helper returned a
completed marker that its next validation rejected. Completion now remains valid
and later prints cannot overfill. Zero-total initial markers and malformed prints
still reject. Exact integer timestamps replace boolean/string/fractional coercion.
Existing side, price, activation and expiry semantics remain. A synthetic common
background path `[10, 20, 10, 9]` gives one versus zero virtual fills under two
compatible cancellations; it is not an owned historical fill or market return.

Twelve of eighteen new cases failed before; thirty-six focused final cases pass,
including existing shared paper and Binance paper controls. Ruff passes. Exact
symbol inventory found definitions/tests only, no runtime integration; initial
queue provenance, print identity/deduplication and causal ordering remain caller
requirements. No cancellation compiler, new profit evaluator or queue probability
was invented. Frozen full-fill targets were not changed into cash ledgers.

Next maker R&D must pair policies on qualified causal L1/L2 paths, retain partial
base quantities and every unresolved episode, and evaluate fees, latency, hedge
and orphan cash through existing accounting components across matched histories.
Do not treat zero full-fill support as zero partial inventory, or front/back fills
as arbitrary policy-profit extrema. No existing market retry gate reopens solely
because this paper exists. No fit, venue capture, account, order, credential,
protected access, campaign or GPU benchmark occurred. Counts remain
201 observations, 65 hypotheses, 37 scopes and zero qualified stable edges.
