> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.us/llms.txt
> Use this file to discover all available pages before exploring further.

# Mutually Exclusive Collateral Return

> How portfolio margin optimization works for mutually exclusive event outcomes

Mutually exclusive collateral return is a portfolio margin optimization that reduces your margin requirement when you hold offsetting short positions in instruments from the same event where only one outcome can occur.

## What Are Mutually Exclusive Events?

Mutually exclusive events have multiple possible outcomes, but only one can happen. Examples include:

* **Elections**: Only one candidate can win
* **Championships**: Only one team can win the title
* **Award winners**: Only one nominee can win the award

When you're short on multiple instruments from the same mutually exclusive event, the exchange recognizes that your maximum loss is capped because only one outcome can occur.

## How Collateral Return Works

When collateral return is enabled on your account and you hold short positions across multiple instruments in the same mutually exclusive event, your margin requirement is reduced by the smaller position size.

**Without Collateral Return:**

* Short 9,000 contracts of Candidate A
* Short 1,000 contracts of Candidate B (same election)
* Margin requirement: 10,000 (9,000 + 1,000)
* Buying power: 0 (if you started with 10,000 balance)

**With Collateral Return:**

* Short 9,000 contracts of Candidate A
* Short 1,000 contracts of Candidate B (same election)
* Margin requirement: 9,000 (reduced by the 1,000 offsetting position)
* Buying power: 1,000 (freed up capital)

## Why This Matters

Because only one candidate can win, your worst-case scenario is having the larger short position lose. The smaller offsetting position guarantees you'll win that amount, effectively reducing your net exposure.

**Maximum loss calculation:**

* If Candidate A wins: You lose 9,000 but gain 1,000 = Net loss of 8,000
* If Candidate B wins: You lose 1,000 but gain 9,000 = Net gain of 8,000

Your true maximum loss is 8,000, not 10,000. Collateral return recognizes this and only requires 9,000 in margin instead of 10,000.

## Using Freed-Up Buying Power

The 1,000 in freed-up buying power can be deployed into other markets (different events). This allows for more efficient capital utilization across your entire portfolio.

However, you cannot use this freed-up buying power to increase your position in the same mutually exclusive event that generated the collateral return.

## Closing Offsetting Positions

When you close one of the offsetting positions, you must "return" the collateral that was freed up.

**Example:**

* You have 9,000 short Candidate A, 1,000 short Candidate B
* Collateral return gives you 1,000 buying power
* You use that 1,000 to trade in a different market
* If you try to buy back the 1,000 contracts of Candidate B, you must return the 1,000 in collateral
* If that 1,000 is already deployed elsewhere, the order to close will be rejected

## Portfolio Margin Optimization

Collateral return is part of Polymarket US's portfolio-level margining system. The exchange calculates margin requirements across your entire portfolio, recognizing natural hedges and offsetting positions to maximize capital efficiency.

This differs from position-by-position margining where each position is treated in isolation, requiring full margin regardless of offsets.

## Key Points

* Collateral return only applies to mutually exclusive events where one outcome must occur
* Your margin requirement is reduced by the smaller offsetting position size
* Freed-up buying power can be used in other markets, not the same event
* Closing offsetting positions requires returning the freed collateral
* This is a portfolio margin optimization, not a reduction in actual risk
