# Model and local-LLM improvement: source-to-implementation cycle 1

Scope: a bounded technical comparison and two demonstrated integration repairs,
not a trained model, a financial experiment, a full-repository review, or proof
of superiority. Starting revision: `56bacc15b36124de0bb9c7676e19ed8830939053`.
The existing Round 71 institutional/microstructure research and similar-repo
review are reused rather than repeated. Durable payoff mechanisms are a useful
target; a permanently positive executable spread is not promised.

## What changed

The active profitability audit still referenced the registry preceding the
native full-hedge frontier checkpoint. The existing integration test reproduced
one failure. Its source binding and self-hash are corrected; classifications,
frontier returns, acceptance decisions and historical results are unchanged.
`audit-before.json` preserves the prior canonical payload; original file bytes
remain in the starting Git revision. The workflow now requires the existing
two-test ledger check after a registry amendment, not a repeated broad suite.

The current live AI provider checked `done=true` but ignored `done_reason`.
Nine rejection cases failed before the repair, including `length`, unknown,
missing and non-string reasons despite syntactically valid approval JSON.
The provider now requires exact `done=true` and `done_reason=stop` before parsing
an action or issuing the subsequent residency request. Natural completion is
necessary, not sufficient: the existing schema, causal identity, evidence,
digest, risk, token and GPU checks still apply. No inference or model training
was run. Prior AI records and frozen research implementations are unchanged;
this repair does not retrospectively qualify them or prove paired uplift.

54 focused checks pass across live AI, matched shadow uplift and the active
profitability audit. Ruff format/check pass on the changed Python files.
No benchmark, market-data request, account access, credential use or order.

## Primary-source comparison, not blind copying

Four frozen bounded GETs retained 98,024 response bytes with intent/completion
journals: two single-record revision selections and the two fixed source paths.
All passed HTTP, byte, encoding, contract and response-hash verification.
Two further immutable root-license GETs retain 36,199 bytes. The complete MIT
and GPLv3 terms and attribution accompany the unmodified research sources in
`UPSTREAM_NOTICES.md`; nothing is imported into the trading runtime or relicensed.

| Inspected source | Concrete observation | Decision for this repository |
| --- | --- | --- |
| [Ollama API types at efe43c5](https://github.com/ollama/ollama/blob/efe43c5561047d8baf8c862730b2151249d51fd5/api/types.go#L539) | ChatResponse exposes completion status and a separate finish reason. Retained native local host evidence already contains `stop`. The type alone does not enumerate every possible reason. | Require the observed natural-completion reason; reject absent/other reasons rather than coercing them to successful evidence. |
| [FreqAI lifecycle at 73b23ef](https://github.com/freqtrade/freqtrade/blob/73b23ef9673e2e28bf7bfe6174eb175222cb29df/freqtrade/freqai/freqai_interface.py#L190) | A scanning thread handles retraining; shutdown can join it without a timeout. | Threaded background work is not independent fault containment. Do not copy this as the requested supervisor/process architecture. |
| [FreqAI prediction routing](https://github.com/freqtrade/freqtrade/blob/73b23ef9673e2e28bf7bfe6174eb175222cb29df/freqtrade/freqai/freqai_interface.py#L507) | This function's first-prediction branch returns before its later expiry check; that later branch returns null predictions and `do_predict=2`. | Audit first-use/reload/stale-model paths and their callers. A signal requiring strategy cooperation is weaker than a deterministic pre-entry gate. This bounded reading does not establish a system-wide upstream defect: loader/caller protections were not audited. |

Our current AI reviewer has bounded queue/cache capacity and a short join, but
is still a daemon thread, not a terminable child process. The AI wrapper binds
exact cases and checks review age. Review age is not model age, training-label
maturity, quote age, news-publication age or economic validity. These are
separate admission requirements; neither framework popularity nor a functioning
HTTP provider proves any of them. Current readiness/ML callers were only
partially inspected, so their complete lifecycle remains unqualified.

## Sequenced research and implementation cycles

These route existing requirements; they do not authorize a new market capture,
change a consumed result, loosen a retry trigger or enable trading. Each cycle
needs a specific new question and an implementation/evidence delta before more
requests, model fits or expensive backtests. Existing Round 71 designs and
acceptance contracts remain controlling.

1. **Complete cash and causal labels before more fits.** Route the five affected
   rate-only consumers through qualified native funding/mark/quantity clocks
   using the shared cash law, preserving historical implementations/results.
   Reconcile partial fills, native fee currency, residual inventory and actual
   entry/exit basis. The retained 288-event period and full-hedge frontier are
   diagnostics, not a complete cash-labelled training dataset. Unknown event
   coverage or entitlement remains unknown, not zero; no candle substitution.
   Deliver a forward label adapter and source-bound integration evidence before
   expanding models. Rejection consequence: no cash-profit label or fit.
2. **Predict tradable surplus, not price accuracy.** Compare an abstaining
   structural/no-change baseline with existing shallow action-conditioned
   probability/payoff/tail/fill-time models, on cash-qualified chronological
   inputs. Distinguish cash-flow opportunity from basis, latency, queue and
   capital risk. For Polymarket use exact settlement/payoff identity, complete
   outcome coverage and side-specific acquisition costs; for Binance use
   complete hedge cash, collateral and native costs. Test conservative opportunity
   ceilings before deep networks. Deliver fixed-cost, fixed-capacity paired
   comparisons; no benefit means no added complexity.
3. **Ensembles only when complementary out of sample.** Build on existing
   regime/multi-objective routing, not another unbounded parameter search.
   Candidate specialists may cover executable surplus, fill/adverse-selection,
   inventory/capital tail risk and event-rule uncertainty. Abstention is a valid
   action; disagreement is not an excuse for leverage. Freeze model membership,
   causal feature availability and resource budget on development data; compare
   against the best simple baseline on untouched chronological roles with
   aligned observation endpoints and selection-risk evidence. Remove a member
   that adds compute without after-cost incremental value.
4. **Local LLM plus internet as evidence services, not execution authority.**
   Keep retrieval off the execution loop. Any future external context needs
   immutable source receipts, actual publication/availability clocks, bounded
   refresh/expiry, instruction isolation and a target-free structured view.
   Missing/conflicting facts must reduce confidence or abstain. Freeze a paired
   no-LLM/LLM comparison with identical action opportunities, causal completion
   deadlines, costs, inventory and risk limits. Explicitly account for delayed
   decisions and missing completions; do not award veto credit from post-hoc
   realized winners/losers. No independent after-cost uplift means LLM disabled
   for economic decisions. Larger reasoning models are not automatic alpha.
5. **Fault containment, persistent model health and measured acceleration.**
   Implement the already-required supervisor, terminable inference/training
   children and deterministic gateway; only that gateway can act on ownership,
   reconciliation and risk authority. Persist recovery/model-health state across
   restarts; expired/invalid evidence blocks new exposure but never a safe close.
   GPU-batch qualified development workloads when actual backend identity,
   precision parity, memory headroom and measured throughput justify it. Keep
   decision-path deadlines independent of training utilization. Use passive
   workstation load observation for timing claims; leave other tasks untouched.

Large-scale backtesting follows cash/data qualification and low-cost baseline
information gain. It must track strategy trials, dependence, selection,
turnover, capacity, margin and liquidity stress—not count repeated tests as
independent evidence. No AI-model benchmark or library feature list establishes
long-term profitability. Full semantic code review and final exhaustive bug
hunting remain outstanding release gates.

## Process corrections and limits

Two documentation searches returned no results; the revision-bound native
sources above were used instead. One local revision classifier had an expression
ordering error; the saved response was reused without another GET. A new audit
snapshot was initially constructed from truncated tool output and rejected by
JSON validation. It was replaced from the parsed original Git payload before
publication; no historical source was lost. The workflow now explicitly forbids
using truncated output to construct evidence. No failed artifact was published.

Semantic-search MCP smoke passed, but no active repository index exists. Exact
file/symbol routing was used; no cold index, external-code execution, broad CI
or GPU benchmark was launched. This cycle creates no new economic observation,
accepted mechanism or stability claim: current counts remain **201/65/37/0**.
