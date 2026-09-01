> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.us/llms.txt
> Use this file to discover all available pages before exploring further.

# Directional Collateral Return

> How portfolio margin optimization works for directional event outcomes

Directional collateral return is a portfolio margin optimization that reduces your margin requirement when you hold offsetting positions in instruments from the same directional event.

## What Are Directional Events?

Directional events have multiple instruments with ordered strike levels where outcomes are logically linked. If a higher threshold is true, all lower thresholds must also be true. Examples include:

* **Point spreads**: Will the team win by more than 3.5? More than 6.5? More than 10.5?
* **Totals**: Will the combined score exceed 40.5? Exceed 47.5? Exceed 50.5?
* **Price levels**: Will Bitcoin exceed 50K? 75K? 100K?

Each instrument has an ordinal rank (rank 1 = lowest threshold). If rank 3 resolves Yes, then ranks 1 and 2 must also resolve Yes.

## How Collateral Return Works

When collateral return is enabled on your account and you hold a lower-ranked long position that offsets a higher-ranked short position in the same directional event, your margin requirement is reduced.

Consider a Bitcoin price hit event with three instruments:

| Instrument                           | Question                              |
| ------------------------------------ | ------------------------------------- |
| `cphc-btc-hit-2026-03-31-120000pt00` | What price will Bitcoin hit in March? |
| `cphc-btc-hit-2026-03-31-130000pt00` | What price will Bitcoin hit in March? |
| `cphc-btc-hit-2026-03-31-140000pt00` | What price will Bitcoin hit in March? |

If Bitcoin hits 140,000, it must have also hit 130,000 and 120,000. This directional relationship is what enables collateral return.

**Without Collateral Return:**

* long 3 contracts of `cphc-btc-hit-2026-03-31-120000pt00`
* short 2 contracts of `cphc-btc-hit-2026-03-31-130000pt00`
* Margin requirement: 2 (full short position)
* Buying power reduced by 2

**With Collateral Return:**

* long 3 contracts of `cphc-btc-hit-2026-03-31-120000pt00`
* short 2 contracts of `cphc-btc-hit-2026-03-31-130000pt00`
* Margin requirement: 0 (short fully offset by lower-ranked long)
* Buying power: no reduction

## Why This Matters

Your long position at the lower strike guarantees a payout in any scenario where your short position at the higher strike loses. If Bitcoin hits 130,000 (your short loses), it must have also hit 120,000 (your long wins), so the long payout covers the short's loss.

**Maximum loss calculation:**

* Bitcoin below 120,000: Long loses, short wins = net depends on entry prices
* Bitcoin between 120,000 and 130,000: Both positions win
* Bitcoin above 130,000: Long wins, short loses - but long payout offsets short loss

The exchange recognizes this natural hedge and reduces your margin accordingly.

## Directionality Matters

Only a lower-ranked long can offset a higher-ranked short. The reverse does not work.

* long `cphc-btc-hit-2026-03-31-120000pt00`, short `cphc-btc-hit-2026-03-31-130000pt00` - collateral return applies
* long `cphc-btc-hit-2026-03-31-130000pt00`, short `cphc-btc-hit-2026-03-31-120000pt00` - NO collateral return

This is because Bitcoin hitting 130,000 guarantees it also hit 120,000, but Bitcoin failing to hit 120,000 tells you nothing useful - 130,000 already failed too, so there's no offset.

## Multiple Offsets

A single lower-ranked long position can offset multiple higher-ranked short positions. Using the Bitcoin example with more strike levels:

* long 10 contracts of `cphc-btc-hit-2026-03-31-120000pt00`
* short 2 contracts of `cphc-btc-hit-2026-03-31-130000pt00`
* short 3 contracts of `cphc-btc-hit-2026-03-31-140000pt00`
* short 6 contracts of `cphc-btc-hit-2026-03-31-150000pt00`
* Total short: 11, Collateral return: 10 (2 + 3 + 5 from remaining long)
* Margin requirement: 1

Multiple long/short pairs across different ranks can offset simultaneously. The exchange matches the highest-ranking short with the highest available long first, working down.

## Using Freed-Up Buying Power

Freed-up buying power can be deployed into other markets (different events). This allows for more efficient capital utilization across your entire portfolio.

However, you cannot use this freed-up buying power to increase your position in the same directional event that generated the collateral return.

## Closing Offsetting Positions

When you close one of the offsetting positions, you must "return" the collateral that was freed up.

**Example:**

* You are long 3 contracts of `cphc-btc-hit-2026-03-31-120000pt00`, short 2 contracts of `cphc-btc-hit-2026-03-31-130000pt00` (collateral return of 2)
* You use that freed buying power to trade in a different market
* If you try to sell your long position, you must return the collateral
* If that buying power is already deployed elsewhere, the order will be rejected

Hypothetical collateral return from new orders is not factored into buying power checks. The exchange only considers your current positions when calculating collateral return.

## Key Points

* Collateral return applies to directional events where lower thresholds must be true if higher ones are
* A lower-ranked long offsets a higher-ranked short (not the reverse)
* One long position can offset multiple short positions across higher ranks
* Freed-up buying power can be used in other markets, not the same event
* Closing offsetting positions requires returning the freed collateral
* This is a portfolio margin optimization, not a reduction in actual risk
