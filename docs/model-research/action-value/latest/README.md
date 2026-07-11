# Action-Value Round 10 Evidence

Status: **rejected**. This is checksummed Binance USD-M discovery evidence, not
a profitability, execution, or trading-authority claim.

- UTC window: 2023-09-04 through 2023-09-10 (now consumed for selection)
- Precommitted candidates: 12
- Statistical fit failures: 9
- Trained candidates: 3
- Unrejected candidates: 0
- Policy trades across trained candidates: 0
- Selection trades across trained candidates: 0
- Positive predicted-edge policy rows: 158
- Design SHA-256: `a2aa45f8245a12a85ea94365333f621fc2824a425ad6731105253a138fb0e049`
- Corpus certificate SHA-256: `5782bd80e2de50fe651471d5fb7e89b4449c584299731fb80967378e732639ab`
- Implementation commit: `58e6ac5f75bccb75739c6084c4861ba2ecc981fe`

No trained candidate selected an executable trade; per-trade mean return
is undefined and no equity curve is generated.

Fit errors and every trained artifact hash are retained verbatim
in the source table. A failed fit is not counted as a zero-return model, and an
abstaining model is not presented as profitable.

## Charts

![After-cost performance](charts/after-cost-performance.svg)

![Forecast quality](charts/forecast-quality.svg)

![Action funnel](charts/action-funnel.svg)

![Research progress](charts/research-progress.svg)

The source tables are [candidates.csv](candidates.csv) and
[progress.csv](progress.csv); reconstructed class support and top-score outcomes
are in [diagnostics.json](diagnostics.json) when required by the round. Every
trained artifact SHA-256 and every fit error is retained in `candidates.csv`;
no zero-trade equity curve is fabricated.
Regenerate by passing this round's `--design`, `--evidence-root`, and required
`--diagnostics` to `python tools/publish_action_value_discovery.py`.
