> For clean Markdown of any page, append .md to the page URL.
> For a complete documentation index, see https://docs.paradex.trade/llms.txt.
> For AI client integration (Claude Code, Cursor, etc.), connect to the MCP server at https://docs.paradex.trade/_mcp/server.

# Trading Fees

#### Announcement

An updated Pro maker fee rate for Perps & Spot API traders takes effect on July 3 at 12:00 SGT. See [below](#maker-fees).

Retail trading stays at 0 fees across Perps, Spot, and Options.

* Retail Takers and Makers have 0% fees on all products: Perps, Spot and Options
* Pro fees follow fee schedule outlined below

## Perps & Spot Fees

### Retail

|       | Perps | Spot |
| ----- | ----- | ---- |
| Taker | 0%    | 0%   |
| Maker | 0%    | 0%   |

### Pro

#### Maker fees

Flat 0.003% (0.3 bps) fee on all Perps and Spot maker orders, effective July 3 at 12:00 SGT. Fees are charged on trade notional (Size × Trade Price).

#### Taker fees

Spot & Perps taker fees are based on your 14D volume, less any applicable discounts.

`Your taker fee = base rate × (1 − total discount)`

* Pro taker fees cannot go below 1.75 bps (0.0175%), regardless of total discount.
* Fees are charged on trade notional (Size × Trade Price).

##### Base rate

Your base rate is set by your order classification and combined 14D volume across your main account and subaccount(s).

| Tier  | 14D Volume (USD) | Taker Fee |
| ----- | ---------------- | --------- |
| Pro 0 | —                | 0.045%    |
| Pro 1 | ≥ 1M             | 0.042%    |
| Pro 2 | ≥ 5M             | 0.039%    |
| Pro 3 | ≥ 10M            | 0.037%    |
| Pro 4 | ≥ 25M            | 0.035%    |

##### Discounts

Three discounts are additive and apply to your volume-based taker fee:

#### [Stake \$DIME](#stake-dime)

Up to 30% off, scaling with the amount staked

#### [Pay fees in \$DIME](#pay-fees-in-dime)

40% off for all orders during initial rollout

#### [FastFills](#fastfills)

30% off for Pro orders matching Retail

###### Stake \$DIME

Stake \$DIME for tiered Spot & Perps taker fee discounts, up to 30%.

| Staking Tier | \$DIME Staked | Fee Discount |
| ------------ | ------------- | ------------ |
| Base         | 0             | 0%           |
| Wood         | ≥ 10K         | 5%           |
| Bronze       | ≥ 50K         | 10%          |
| Silver       | ≥ 100K        | 15%          |
| Gold         | ≥ 500K        | 20%          |
| Platinum     | ≥ 1M          | 25%          |
| Diamond      | ≥ 2M          | 30%          |

To unstake, submit an unstaking request at any time. A 24-hour cooldown follows, after which you must confirm to receive your funds. For more on \$DIME, see [\$DIME Utility](/docs/getting-started/dime-utility).

###### Pay fees in \$DIME

Pay fees in \$DIME for a flat 40% discount. Fees are deducted from your spot \$DIME balance at the \$DIME ↔ USDC mark price at transaction time.

#### Example

A user who stakes 100K \$DIME (15% staking discount) and pays fees in \$DIME (40% \$DIME fee discount) gets a total discount of 55% (15% + 40%), applied to their volume-based taker fee.

###### FastFills

Pro taker fees get an additional 30% off when matching Retail orders, stackable with staking and \$DIME discounts. The 1.75 bps (0.0175%) minimum fee still applies.

See [FastFills](/trading/fastfills) for details.

##### Fee scenarios

How volume tiers, staking, \$DIME fee payment, and FastFills combine to determine a Pro user's final taker fee.

| Scenario | Volume Tier | Base Taker Fee | \$DIME Fee Discount | Staking Amount | Staking Discount | FastFills Discount | Total Discount | Final Fee |
| -------- | ----------- | -------------- | ------------------- | -------------- | ---------------- | ------------------ | -------------- | --------- |
| Pro A    | Pro 1       | 0.042%         | 40%                 | 10K \$DIME     | 5%               | 30%                | 75%            | 0.0175%\* |
| Pro B    | Pro 4       | 0.035%         | 40%                 | 2M \$DIME      | 30%              | —                  | 70%            | 0.0175%\* |

\*Minimum Pro fee of 1.75 bps (0.0175%) applies.

## Options Fees

### Retail

| Maker Fee | Taker Fee |
| --------- | --------- |
| 0%        | 0%        |

### Pro

Fees are capped at 12.5% of the option price (premium). The FastFills 30% discount applies to Pro taker fees matching Retail orders.

| Maker Fee (RPI & API) | Taker Fee |
| --------------------- | --------- |
| 0.01%                 | 0.01%     |

## Settlement

When a market is delisted, a settlement fee applies to any positions still open.

| Market                     | Fee     |
| -------------------------- | ------- |
| Spot and Perpetual Futures | 0.015%  |
| Dated Options              | 0.0075% |