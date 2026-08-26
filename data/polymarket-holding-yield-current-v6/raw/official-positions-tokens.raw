> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Positions & Tokens

> Understanding outcome tokens and how positions work on Polymarket

Every prediction on Polymarket is represented by **outcome tokens**. When you trade, you're buying and selling these tokens. Your **position** is simply your balance of tokens for a given market.

<Frame>
  <img src="https://mintcdn.com/polymarket-292d1b1b/FOMte3ewbG-LVy3k/images/core-concepts/token-lifecycle.png?fit=max&auto=format&n=FOMte3ewbG-LVy3k&q=85&s=cad279109c43c68c541123c2d348c4c5" alt="" className="dark:hidden" width="1596" height="952" data-path="images/core-concepts/token-lifecycle.png" />

  <img src="https://mintcdn.com/polymarket-292d1b1b/FOMte3ewbG-LVy3k/images/dark/core-concepts/token-lifecycle.png?fit=max&auto=format&n=FOMte3ewbG-LVy3k&q=85&s=204d16c1d89892a3c8573060aa04780e" alt="" className="hidden dark:block" width="1596" height="952" data-path="images/dark/core-concepts/token-lifecycle.png" />
</Frame>

## Outcome Tokens

Each market has exactly two outcome tokens:

| Token   | Redeems for | If...                    |
| ------- | ----------- | ------------------------ |
| **Yes** | \$1.00      | The event occurs         |
| **No**  | \$1.00      | The event does not occur |

Tokens are **ERC1155** assets on Polygon, using the [Gnosis Conditional Token Framework](https://github.com/gnosis/conditional-tokens-contracts/) (CTF). This means they're fully onchain and function as standard ERC1155 tokens.

<Note>
  Outcome tokens are always fully backed. Every Yes/No pair in existence is
  backed by exactly `$1` of pUSD collateral locked in the CTF contract.
</Note>

### Split

Convert pUSD into outcome tokens. Splitting \$1 creates 1 Yes token and 1 No token.

```
$100 pUSD → 100 Yes tokens + 100 No tokens
```

Use this when you want to:

* Create inventory for market making
* Obtain both sides of a market

### Trade

Buy or sell tokens on the order book. This is how most users acquire positions.

* **Buy Yes** at `$0.60` → Pay `$0.60`, receive 1 Yes token
* **Sell Yes** at `$0.60` → Give up 1 Yes token, receive `$0.60`

You can sell your position at any time before resolution.

### Merge

Convert a complete set of tokens back into pUSD. Merging requires equal amounts of Yes and No tokens.

```
100 Yes tokens + 100 No tokens → $100 pUSD
```

Use this when you want to:

* Exit a position without trading
* Convert accumulated tokens back to collateral

### Redeem

After a market resolves, exchange winning tokens for pUSD.

| Outcome             | Yes tokens     | No tokens      |
| ------------------- | -------------- | -------------- |
| Event occurs        | Worth \$1 each | Worth \$0      |
| Event doesn't occur | Worth \$0      | Worth \$1 each |

```
100 winning tokens → $100 pUSD
```

### Position Value

The value of your position depends on the current market price:

```
Position value = Token balance × Current price
```

If you hold 100 Yes tokens and Yes is trading at \$0.75:

```
Position value = 100 × $0.75 = $75
```

## Profit and Loss

Your profit depends on how the market resolves compared to your entry price.

### Example - Buying Yes at 0.40

| Scenario            | Outcome  | Return | Profit                    |
| ------------------- | -------- | ------ | ------------------------- |
| Event occurs        | Yes wins | \$1.00 | +\$0.60 per token (150%)  |
| Event doesn't occur | No wins  | \$0.00 | -\$0.40 per token (-100%) |

### Holding Rewards

Polymarket pays a **4.00% annualized** Holding Reward based on your total position value in eligible markets. Your total position value is randomly sampled once each hour, and the reward is distributed daily. The rate is variable and subject to change at Polymarket's discretion.

### Example - Selling Before Resolution

You can lock in profits or cut losses by selling before the market resolves:

* Bought Yes at `$0.40`
* Price rises to `$0.70`
* Sell at `$0.70` → Profit of `$0.30` per token (75%)
