# Polymarket Exact Duplicate-Contract Screen v1

> **Rejected current snapshot. No edge, profitability, or trading-authority
> claim.** One repeated question had two different canonical payout-rule sets;
> zero cross-condition pairs qualified for pricing.

This screen tests whether separate Polymarket conditions express byte-identical
binary payout rules. Such a duplicate could support a direction-independent
bundle screen, but only after contract identity is proven. A shared title or
question is a discovery key, not proof of equivalent resolution.

## Exact Identity Contract

Different condition IDs advance only when all of these canonical market fields
are exactly equal:

- question;
- description;
- market end date;
- resolution source;
- group-item title;
- ordered outcomes.

The tool permits no synonym replacement, template substitution, whitespace
normalization, semantic model judgment, or parent-event fallback. It fetches
the official Bitcoin, Ethereum, and Solana tag pages within a fixed 1,000-event
offset bound, identifies repeated byte-exact questions from embedded markets,
then fetches canonical event endpoints only for those candidates.

## Source-Bound Result

The canonical fetch contained 1,478 unique active scoped events and 607 active,
order-accepting, order-book-enabled, non-negative-risk binary markets. There was
one repeated exact question:

`Will MicroStrategy announce bankruptcy before 2027?`

It appeared under two separate conditions and events. Their deadlines,
resolution-source strings, and ordered outcomes matched, but two payout-rule
fields did not:

- `description` used separate company-specific and listed-company wording;
- `group_item_title` was empty in one condition and `MicroStrategy` in the
  other.

The resulting payout-rule SHA-256 fingerprints were different. The screen
therefore found zero exact payout-rule duplicate groups and stopped before
order books, fees, persistence, fills, or promotion. This prevents a small
price discrepancy from being mislabeled as a guaranteed bundle when the
underlying resolution contracts are not identical.

The authoritative evidence is
[`duplicate-contract-parity-snapshot-v1-2026-08-25.json`](duplicate-contract-parity-snapshot-v1-2026-08-25.json),
result SHA-256
`7eab53089f904d647538de29193dcfa33bfabaa73440161d5fdec706b7bcb5b1`.

Do not rerun this current-state screen. Reopen it only under a frozen
prospective sampling contract or when canonical payout terms, independently
verified resolution lineage, or execution evidence materially changes the
identity boundary.
