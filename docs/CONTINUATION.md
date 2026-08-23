# Continue Development

This is the authoritative handoff. Verify every drift-prone claim before acting.
Development belongs only on `main`; do not create another development branch.

## Closeout State

- The last fully hosted-verified baseline before this Round 75 closeout is
  `fc416b7112ae05d668871bd0ebeca19501f594b6`. CI, Ruff, Vulture,
  Super-Linter, CodeQL, and DeepSource passed that exact revision. GitHub exposed
  only `main`, and the available APIs reported zero open Dependabot,
  code-scanning, and secret-scanning alerts. Reverify these claims for the new
  closeout commit; zero alerts never proves zero undisclosed vulnerabilities.
- The repository is beta `0.1.0-beta.1`. No model has production authority or a
  demonstrated long-lived after-cost edge. Binance remains paper/testnet/Demo;
  Polymarket remains independent, disabled by default, and unpromoted.
- The one historical cutoff is `2026-08-14T00:00:00Z`. Do not move it or fetch
  the newest history on each iteration. Prospective experiments remain isolated
  from that frozen snapshot.

## Round 75 Terminal Verdict

The campaign ended at `2026-08-23T12:00:00Z`. A read-only host audit observed no
owned service or capture process, a released lease, and a valid
`campaign_terminal` service state. The scheduled task's triggers ended at the
same boundary and must not restart the campaign.

Canonical evidence:

- `round-075-terminal-campaign-audit-2026-08-23.json`: rejected incomplete
  campaign; artifact SHA-256
  `94a556887adf33996168de260a8a172952bc64040273a7ff2dfb373f2c2f50d6`.
- `round-075-wal-copy-recovery-2026-08-23.json`: controlled recovery on copies
  only; artifact SHA-256
  `18fa37db65aa19fed16128d4aaf0af10cb0606c95192b46dfca0b0a28e932751`.
- `round-075-post-campaign-amendment-v1.json`: frozen-source preservation and
  non-restartable supervisor v3; artifact SHA-256
  `f1a1bc536cf37e7ec867f4d8ddf450537cfa83ae7ed17dd061421db5fa7c0db3`.

Facts: 720 slots were preregistered; 35 produced results, 33 were admitted, 684
were missed, and slot 67 remained incomplete. All admitted epochs are training
role. Tuning and test have zero admitted epochs. Raw training-role eligible
anchor time is `28,903,469,878,300 ns` against `394,740,000,000,000 ns`
required. Shard 002 retains a `9,431,058` byte WAL. WAL recovery on an exact
copy added 196 frames and 62 REST rows but no terminal report, confirming the
payload belongs to the incomplete run and is inadmissible.

Consequences: do not open the original databases, replay or delete the original
WAL, use slot 67, materialize targets, train, tune, inspect sealed tests, or make
accuracy, AI-uplift, edge, profitability, ROI, or trading-authority claims from
Round 75.

## Protected Local Work

`C:\trader\simple_ai_trading-model-dev` remains detached at
`c42219d47dc781a46411a4ec96838f8a26c3924c`. Its terminal evidence is frozen.
The latest read-only preservation snapshot reports:

- 99 tracked status entries and 218 untracked paths;
- binary diff Git hash `bf7f896c3fa2b17a7a7a34887b2d3fe04cb4be54`;
- untracked manifest SHA-256
  `2b81dd7f8c70bf319ac4b40725c1dd06fc6d0d2be119e45dd064e83d6428d50f`,
  calculated as SHA-256 of sorted UTF-8 records containing path, NUL, lowercase
  file SHA-256, and LF;
- newest untracked write `2026-08-10T21:43:27.4494537Z`.

Never clean, reset, switch, commit, or blindly copy that worktree. Review its
remaining content paths against current `main` with a three-way comparison and
integrate only work that is both unpublished and still valid. The exact Round
75 v4 implementation has already been preserved in `main`; do not duplicate it.

The Round 21 sidecar worktree
`C:\trader\simple_ai_trading-round21-sidecar-v2` remains protected through
`2026-08-29T23:40:00Z`. Its process IDs are ephemeral. Do not touch it until a
contract-defined terminal audit proves the boundary has passed and the process,
lease, state, database, and WAL agree.

## Next Work

1. Reverify `main`, GitHub branches, alerts, and exact-SHA hosted workflows for
   this closeout.
2. Preserve and three-way review the remaining model-dev changes. Integrate on
   `main` only, in coherent batches with source-specific tests.
3. Run Polymarket Round 29 only after every source-bound input passes. Keep Stage
   1 outcomes and economics sealed until the probability gate permits access.
4. Reject any candidate that fails bull, bear, sideways, choppy, high-volatility,
   liquidity-stress, or latency-stress after-cost slices. Abstention is required
   where evidence is unsupported; no strategy can guarantee profit or prevent
   every future loss.
5. Keep Binance and Polymarket strategies, capital, ownership ledgers, Stop, and
   promotion evidence independent. Binance data may be a causal Polymarket
   feature only when timing provenance proves it arrives first.
6. Evaluate the night-effect idea as a separate stock-market hypothesis using
   exact exchange calendars, auction mechanics, overnight gaps, spreads, fees,
   taxes, borrow, capacity, and causal timestamps. It has no current crypto or
   trading authority.
7. Perform final walk-forward validation only after source continuity,
   representative train/tune/test coverage, after-cost economic gates, and
   cross-regime gates pass. Walk-forward is not a substitute for those gates.

## Verification Scope

The focused Round 75 closeout currently passes 20 tests plus Ruff and formatting.
Run the smallest affected checks during development, then full CI once before
publication. Do not repeat unchanged expensive suites between adjacent edits.

The previous verbose handoff and chronology are preserved byte-for-byte in:

- `docs/archive/agent-history/AGENT_START-before-2026-08-23-closeout.md.txt`
  (SHA-256 `2ba0ee28f38a9f5d2a177cf4b270fe924517e88f6a9511dd7acb3507ab7907c5`)
- `docs/archive/agent-history/CONTINUATION-before-2026-08-23-closeout.md.txt`
  (SHA-256 `2170f14bcfdf49674c576b8fd7d42aa02dc4569c48ba1f643ec6ad43c8d30b18`)
