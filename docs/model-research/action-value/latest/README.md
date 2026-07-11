# Action-Value Round 11 Evidence

Status: **rejected**. This is checksummed Binance USD-M discovery evidence, not
a profitability, execution, or trading-authority claim.

- UTC window: 2023-08-14 through 2023-09-24 (now consumed for selection)
- Precommitted candidates: 12
- Statistical fit failures: 0
- Trained candidates: 12
- Unrejected candidates: 0
- Policy trades across trained candidates: 0
- Selection trades across trained candidates: 0
- Positive predicted-edge policy rows: 1468
- Design SHA-256: `c7cfe43512104388577fc3730a6963f19253b800088eec70c3e18573d1ac5d64`
- Corpus certificate SHA-256: `3a3851f51a117901eb414762021824aafc1e2a6f4f409c390adfb8db3b1a191e`
- Implementation commit: `745cdb6062e0a8b6a26950053dd9db844e1b0806`

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
