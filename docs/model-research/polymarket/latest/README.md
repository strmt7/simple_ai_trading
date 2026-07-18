# Polymarket model status

![Optimization evidence progression](charts/optimization-progress.svg)

## Current boundary

Round 13 is frozen but has not started. It is a one-use prospective
BTC/ETH/SOL five-minute confirmation of the unchanged Round 11 calibration,
with explicit FOK worst-price limits, exact recorded fees and depth, seven
latency/fee/tick/depth scenarios, a raw-market-prior control, and conjunctive
activity, utility, uncertainty, drawdown, and exposure gates.

Round 12 is not performance evidence. Its recorder captured
`142494` messages, but the evaluator
and publication chain had not been preregistered. It was invalidated before
outcome access; every return, drawdown, and fill field is therefore unavailable,
not zero.

Round 11 remains the latest scored result. Its simulated after-cost utility was
`+22.44105` quote on 42 development conditions, but
maximum drawdown was
`12.36399` and the 95% moving-block-bootstrap lower mean-group
utility was `-1.38152`. It failed uncertainty
and raw-market-prior gates. No profitability, ROI, acceptable-drawdown, paper,
AI-uplift, or trading claim exists.

## Evidence

- [Round 13 frozen contract](../round-013-sealed-confirmation-contract.json)
- [Round 12 invalidation](../round-012-invalidated-capture-evidence.json)
- [Round 11 contract](../round-011-single-leg-directional-value-contract.json)
- [Round 11 report](../round-011-single-leg-directional-value-report.json)
- [Round 11 model artifact](../round-011-single-leg-directional-value-artifact.json)
- [Optimization data](tables/optimization-progress.csv)
- [Publication integrity](publication-integrity.json)

Regenerate these exact tables, charts, and hashes with
`python tools/publish_polymarket_round11.py`. Round 13 can acquire no paper or
live authority until its untouched capture passes every frozen gate and the
authenticated order lifecycle, balance ownership, settlement delay, and
redemption overhead are separately proven.
