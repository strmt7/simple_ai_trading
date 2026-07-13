# Round 54: Sequential Distributional Action Value

> **Rejected before evaluation.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Three DirectML-trained dueling causal TCNs reduced matched early-stop Bellman residual loss by `25.76%` to `26.20%`, but that was not directional or economic proof. Calibration directional rank peaked at `0.044239`. The median-Q controller held a position for up to `791` hours and returned `-30.87%` under stress with `43.96%` drawdown.

A post-hoc finite-hold diagnostic found its least-bad point at `8h`: `+1.07%` stress return across `99` trades, `12.43%` drawdown, and a `-0.6336` bps/hour bootstrap lower bound. It remains rejected and selection-contaminated.

Round 55 must forecast finite-horizon return distributions directly, keep the controller bounded, and gate directional skill, proper scores, path risk, net action value, bootstrap evidence, activity, and asset breadth separately. The evaluation interval remains unread.

## Evidence

| View | Graph | Source |
|---|---|---|
| Bellman fit | [SVG](charts/model-skill.svg) | [CSV](models.csv) |
| Directional rank | [SVG](charts/directional-rank.svg) | [CSV](directional-rank.csv) |
| Policy economics | [SVG](charts/policy-economics.svg) | [CSV](policies.csv) |
| Holding duration | [SVG](charts/holding-duration.svg) | [CSV](holding-summary.csv) |
| Bounded-hold screen | [SVG](charts/finite-hold.svg) | [CSV](finite-hold.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`screen.json`, `failure-diagnostic.json`, and `holding-runs.csv` preserve the full verified evidence. `report.json` binds every publication artifact to the exact external reports, design, dataset, model artifacts, and diagnostic implementation.
