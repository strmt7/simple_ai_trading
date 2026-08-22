# Security

Simple AI Trading is beta research software. Binance execution is restricted to
paper or testnet/Demo environments. The independent Polymarket live-capable
boundary is disabled by default and has no capital authority. No model is
approved for real-money trading.

## Security boundaries

- Credentials must never enter prompts, logs, artifacts, tests, documentation,
  or Git history.
- A configuration error must not widen execution authority. Any future
  real-money boundary requires explicit operator opt-in and direct regression
  tests.
- Only provably bot-owned orders and positions may be changed. Unknown state
  blocks new exposure and forces reconciliation before recovery.
- Risk, ownership, reconciliation, Pause, Stop, and close controls remain
  deterministic. AI cannot override them or block a close.
- Dependencies and GitHub Actions are treated as supply-chain boundaries and
  require pinned, reviewed updates.
- Exchange requests must use bounded timeouts, backoff, idempotency, and
  rate-limit controls.

## Credential handling

Runtime secrets are stored outside the repository in
`~/.config/simple_ai_trading/runtime.json` and loaded only when needed.
`RuntimeConfig.public_dict()` is the only supported source for persisted
runtime snapshots. Logging and request errors must pass through the repository's
redaction layer before they are displayed or written.

If a credential has appeared in a prompt, log, artifact, or commit, treat it as
compromised and rotate it immediately. Redaction after disclosure is not a
substitute for rotation.

## Scanner results

A clear scanner is evidence about that scanner and revision, not proof that no
vulnerability exists. Unsupported or unavailable scanners are reported as
unverified. Static-analysis findings are reviewed at their source and trust
boundary; they are not hidden through broad suppressions.

The current verified inventory and remaining review queues are recorded in
`docs/CONTINUATION.md`.

## Reporting

Report non-public issues privately to the repository owner with a minimal
reproduction and affected revision. Never file a public issue containing a
valid credential, signed request, wallet secret, or unredacted account data.
