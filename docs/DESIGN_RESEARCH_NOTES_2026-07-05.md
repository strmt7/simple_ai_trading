# Design Research Notes - 2026-07-05

These notes summarize the source-backed design pass used for the AI-uplift
risk-gate hardening work.

## High-confidence Findings

1. **Autonomous AI trading needs deterministic guardrails before model output.**
   Recent central-bank and market-structure commentary on AI agents in markets
   emphasizes kill switches, circuit breakers, accountability, and controls for
   herding or correlated behavior. The app should not let an LLM approval step
   override deterministic risk evidence.

2. **AI uplift must be risk-adjusted, not only return-positive.**
   Backtest-overfitting literature continues to emphasize false discoveries,
   multiple testing, and out-of-sample deterioration. AI-assisted trading
   evidence therefore has to prove that AI improves the ML baseline without
   worsening drawdown, liquidation, loss-streak, profit-factor, win-rate, or
   downside return/risk evidence when those metrics exist.

3. **Accepted AI-uplift evidence must be complete, not selectively reported.**
   Accepted AI-uplift artifacts must include finite baseline, AI, and delta
   metrics for return, drawdown, expectancy, trade count, profit factor, win
   rate, liquidation events, loss streaks, and downside return/risk. Missing
   contract fields now fail before model-lab promotion or AI review.

4. **Aggregate AI uplift is not enough.**
   A higher aggregate AI P&L can still be selection noise or one lucky trade.
   Accepted uplift now needs paired holdout deltas with enough samples,
   positive-delta breadth, an exact one-sided sign-test gate, and positive mean
   paired improvement.

## Implemented In This Pass

- Extended `AIUpliftPolicy` and `assess_ai_uplift` with tail-risk criteria for
  liquidation events, loss-streak deterioration, profit factor, win rate, and
  downside return/risk ratio.
- Extended model-lab financial sanity checks so a stale or hand-written report
  cannot claim `ai_uplift.accepted=true` while still carrying rejection reasons
  or bad accepted AI tail-risk deltas.
- Extended model-lab financial sanity checks so accepted AI uplift also requires
  complete baseline, AI, and delta metric groups plus model-size evidence.
- Extended AI uplift and model-lab financial sanity checks with paired holdout
  statistical evidence so aggregate AI improvement cannot pass without
  sample-level breadth.
- Updated README and model research docs to state that AI remains
  advisory/review-only unless risk-adjusted uplift evidence passes.

## Sources

- AI Agents in Financial Markets, arXiv:
  https://arxiv.org/html/2603.13942v2
- AI in Trading, FMSB, February 2026:
  https://fmsb.com/wp-content/uploads/2026/02/FMSB-AI-in-Trading_Final_12.02.26_FINAL.pdf
- Deflated Sharpe Ratio:
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Backtest Overfitting in the Machine Learning Era:
  https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf?abstractid=4686376&mirid=1
- IOSCO Supervisory Toolkit for AI Use in Capital Markets:
  https://www.iosco.org/library/pubdocs/pdf/IOSCOPD823.pdf
