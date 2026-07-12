# Round 34: three-action calibration rejected

**Rejected without trading authority.** The model now keeps action-class probabilities separate from side-profit probabilities. Opportunity, side-profit, multiclass log-loss, and side-profit Brier gates passed; conditional direction and both selected-action economic tails failed. Threshold selection and every later role remained withheld.

| Evidence | Verified result |
| --- | ---: |
| Source window | 2023-05-16 to 2023-07-06 UTC |
| Causal one-second rows | 877,894 |
| CUSUM events / valid barrier outcomes | 230,941 / 229,000 |
| Train / early-stop / calibration rows | 128,307 / 21,934 / 28,581 |
| Opportunity ROC AUC / gate | 0.6637 / 0.6500 |
| Conditional direction ROC AUC / gate | 0.5245 / 0.5500 |
| Side-profit ROC AUC / gate | 0.6066 / 0.5500 |
| Multiclass log-loss / prior ratio | 0.9727 |
| Side-profit Brier / prior ratio | 0.9789 |
| Selected-action side AUC / Spearman IC | 0.5235 / -0.0127 |
| Top-100 / top-500 stress mean | -6.32 / -10.76 bps |
| Eligible rows: conservative / regular / aggressive | 0 / 10 / 39 |
| Architecture gates passed | 4 / 7 |
| Final profiles | none |

![Stage access](charts/stage-access.svg)

![Calibration architecture gates](charts/architecture-gates.svg)

![Calibration eligibility](charts/eligibility.svg)

![Forecast diagnostics](charts/forecast-quality.svg)

![Research progress](charts/research-progress.svg)

The nonzero regular and aggressive eligibility counts show that the corrected expected-return semantics removed Round 33's universal policy bottleneck. They do not rescue the negative ranked tails and are not a reason to loosen risk controls. DirectML tensor execution and OpenCL FP64 LightGBM training were attested. No leverage, testnet or live execution, untouched-period claim, or profitability claim is permitted.

Data: [stages.csv](stages.csv) | [profiles.csv](profiles.csv) | [architecture.csv](architecture.csv) | [forecast.csv](forecast.csv) | [models.csv](models.csv) | [progress.csv](progress.csv) | [integrity report](report.json)
