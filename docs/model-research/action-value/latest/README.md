# Round 40: causal meta-label screen rejected

**The stacked model found weak repeatable discrimination, but not a six-month policy.** All 24 OpenCL LightGBM artifacts reloaded exactly. Meta-label evaluation AUC stayed above chance in every month, yet only December's prior-month threshold cleared the frozen economic and uncertainty gates. Five months correctly remained flat.

| Evidence | Verified result |
| --- | ---: |
| Source / evaluation span | Binance USD-M 1m / 2024-07-01 to 2024-12-31 UTC |
| Primary / meta GPU artifacts | 18 / 6 |
| Threshold cells / months selected | 216 / 1 of 6 |
| Meta evaluation AUC range | 0.564 to 0.587 |
| Selected evaluation actions | 70 (10/29/31 BTC/ETH/SOL) |
| Conditional action result | +36.061 mean net bps; PF 2.008 |
| Six-month day-block lower 95% | -4.307 bps |
| AI cases / AI models run | 0 / 0; ML gate failed first |
| Compute / runtime / peak working set | opencl:auto / 149.3s / 4.36 GiB |
| Trading authority / leverage | none / none |

![Meta-label AUC](charts/meta-label-auc.svg)

![Calibration economics](charts/calibration-economics.svg)

![Evaluation activity](charts/evaluation-activity.svg)

![Research progress](charts/research-progress.svg)

The `+36.061` bps action mean is not a profitability or ROI claim. It comes from 70 December actions after five zero-action months, its stationary day-block lower bound is negative, and the repeated development period is selection-contaminated. The in-sample meta-fit AUC was materially higher than every later role, which identifies short meta training as the next defect to address. Selection-confirmation 2025-H2 and terminal 2026 remain sealed.

Data: [candidate.csv](candidate.csv) | [monthly.csv](monthly.csv) | [thresholds.csv](thresholds.csv) | [models.csv](models.csv) | [sources.csv](sources.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
