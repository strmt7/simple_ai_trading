# Latest Financial Foundation Benchmark

This directory contains the latest committed Kronos candidate evidence. It is
real post-pretraining Binance USD-M archive data for BTCUSDT, ETHUSDT, and
SOLUSDT. It is **not** a profitability result and grants no trading authority.

![Kronos benchmark](benchmark.svg)

| Field | Result |
|---|---:|
| Status | `rejected` |
| Observations | 1536 |
| Raw model MAE | 0.0042225031 |
| Random-walk MAE | 0.0018330693 |
| Raw MAE improvement | -130.3515% |
| Raw information coefficient | -0.053405 |
| Raw direction accuracy | 51.987% |
| Calibrated selection MAE | 0.0016923834 |
| Calibrated random-walk MAE | 0.0016922277 |
| Calibrated uplift probability | 42.350% |
| Calibrated 95% CI | [-0.0000019992, 0.0000014356] |
| Fault worker restarts | 1 |
| Planned worker rotations | 25 |
| First rejection reason | `no_symbol_passed_causal_amplitude_calibration` |

Files:

- `observations.csv` is the source table for replotting.
- `report.json` records data provenance, immutable model/source hashes, metrics,
  causal calibration, bootstrap evidence, seeded repeatability, and worker
  recovery evidence.
- `benchmark.svg` is generated from the CSV and is not the numerical authority.
- `manifest.json` binds the promoted files by SHA-256.

The benchmark intentionally leaves data from 2026 onward sealed as terminal
evidence. A rejected forecast model must remain advisory/research-only.
