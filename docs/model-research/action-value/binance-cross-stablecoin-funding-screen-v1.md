# Binance Cross-Stablecoin Perpetual Funding Screen

> **Rejected. No edge, profitability, paper-trading, or live-trading claim.**
> The full-history frozen gate rejected both BTC and SOL. Do not repeat this
> mechanism without materially new account-specific execution evidence that
> changes the failed economics; another public backfill is prohibited.

This screen tested equal-base opposite positions in Binance linear USD-M
perpetuals quoted and margined in USDT versus USDC. It does not forecast whether
the crypto asset rises or falls. Binance documents public USD-M market data in
its [API catalog][catalog], and its [funding explanation][funding] states that a
positive rate transfers value from longs to shorts while a negative rate does
the reverse. Funding remains dynamic and historical rates do not guarantee
future payments.

## Efficient Evidence Path

The frozen v1 screen made nine public GET requests: one contract catalog, one
all-symbol mark/funding snapshot, one all-symbol top-book snapshot, and six
recent funding histories. BTC and SOL passed its rate-only gate; ETH failed.
The rate-only result is superseded because funding cash is mark price times rate
and the two native quote assets are not one numeraire.

The v2 adjudication reused every v1 futures payload and made one USDCUSDT daily
kline request. After exact funding marks and conservative daily FX bounds, BTC
and SOL still passed the recent 500-settlement screen. Their newer-half
equal-per-leg fee ceilings after the captured spread were only `1.58 bps` and
`4.75 bps`, respectively, so this remained a candidate rather than an edge.

The v3 full-history run then failed because the requested USDCUSDT daily series
did not cover every funding day. Its generic failure writer discarded already
fetched payloads instead of preserving them. That violated the repository's
one-use evidence rule. The failure artifact is retained, and the workflow now
requires a self-hashed journal to be written before and after every request.

The separately frozen v4 recovery was the only deliberate regeneration. It
removed the unnecessary assumption that funding must be converted on every
settlement day. Instead, it retained cumulative native `(USDT, USDC)` cash,
evaluated each slice at fixed USDCUSDT stresses of `0.98` and `1.02`, and
reported the exact FX break-even threshold. It atomically retained all 20
funding responses before economic evaluation.

## Full-History Result

Each candidate had 2,898 common settlement epochs. Orientation was chosen only
on the oldest 966 rows; the middle 966 were validation and the newest 966 were
test. The funding API begins at `2024-01-02 16:00 UTC`, about 20 hours before
the current catalog's USDC contract onboarding timestamps. That source
inconsistency is retained rather than reconciled by assumption. It cannot
rescue either failure because SOL fails both later partitions and BTC fails
selection, validation, and multiple later regime slices.

| Candidate | Fixed orientation | Selection worst stress | Validation worst stress | Test worst stress | Test fee ceiling | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BTC | long BTCUSDT, short BTCUSDC | -126.84 USDT/BTC | -330.56 USDT/BTC | +312.90 USDT/BTC | 9.69 bps/leg | rejected |
| SOL | short SOLUSDT, long SOLUSDC | +0.3071 USDT/SOL | -1.2677 USDT/SOL | -1.4680 USDT/SOL | -36.67 bps/leg | rejected |

BTC also failed its validation second half, 2025 Q2/Q3, and the up, sideways,
regular-volatility, and continuation slices. SOL failed validation, test, both
test halves, multiple complete quarters, every required regime except up, and
the fee gate. Zero of two candidates passed the frozen full-history gate.

The canonical terminal result is
[`binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json`](binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json),
result SHA-256
`8e30be61daaecabd3546e41cdc204d20b8ad38e0fc80c3c9aa96092266a3abe5`.
The self-hashed source journal is
[`binance-cross-stablecoin-funding-recovery-journal-v4.json`](binance-cross-stablecoin-funding-recovery-journal-v4.json).

This result does not weaken the separately ranked Binance quarterly
cash-and-carry or account-specific maker-rebate hypotheses; those mechanisms
have different cash-flow identities and explicit account-evidence triggers.

[catalog]: https://developers.binance.com/en/docs/catalog
[funding]: https://www.binance.com/en/academy/articles/what-are-funding-rates-in-crypto-markets
