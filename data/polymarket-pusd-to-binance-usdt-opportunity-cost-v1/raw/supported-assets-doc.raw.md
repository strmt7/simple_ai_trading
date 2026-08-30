> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Supported Assets

> Chains and tokens supported for deposits to Polymarket

The Bridge API supports deposits from multiple chains and tokens. All deposits are automatically converted to **pUSD on Polygon**, which is used as collateral for trading on Polymarket.

## Get Supported Assets

Retrieve the full list of supported chains and tokens with their minimum deposit amounts.

```bash theme={null}
curl https://bridge.polymarket.com/supported-assets
```

```json theme={null}
{
  "supportedAssets": [
    {
      "chainId": "137",
      "chainName": "Polygon",
      "token": {
        "name": "USD Coin",
        "symbol": "USDC",
        "address": "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB",
        "decimals": 6
      },
      "minCheckoutUsd": 2
    }
  ]
}
```

## Supported Chains

The bridge supports deposits from these blockchain networks:

| Chain           | Address Type | Min Deposit | Example Tokens                              |
| --------------- | ------------ | ----------- | ------------------------------------------- |
| Ethereum        | EVM          | \$7         | ETH, USDC, USDT, WBTC, DAI, LINK, UNI, AAVE |
| Polygon         | EVM          | \$2         | POL, USDC, USDT, DAI, WETH, SAND            |
| Arbitrum        | EVM          | \$2         | ETH, ARB, USDC, USDT, DAI, WBTC, USDe       |
| Base            | EVM          | \$2         | ETH, USDC, USDT, DAI, cbBTC, AERO, USDS     |
| Optimism        | EVM          | \$2         | ETH, OP, USDC, USDT, DAI, USDe              |
| BNB Smart Chain | EVM          | \$2         | BNB, USDC, USDT, DAI, ETH, BTCB, BUSD       |
| Solana          | SVM          | \$2         | SOL, USDC, USDT, USDe, TRUMP                |
| Bitcoin         | BTC          | \$9         | BTC                                         |
| Tron            | Tron         | \$9         | USDT                                        |
| HyperEVM        | EVM          | \$2         | HYPE, USDC, USDe, stHYPE, UBTC, UETH        |
| Abstract        | EVM          | \$2         | ETH, USDC, USDT                             |
| Monad           | EVM          | \$2         | MON, USDC, USDT                             |
| Ethereal        | EVM          | \$2         | USDe, WUSDe                                 |
| Katana          | EVM          | \$2         | AUSD                                        |
| Lighter         | EVM          | \$2         | USDC                                        |

<Note>
  Supported assets change over time. Always call `/supported-assets` for the
  current list before initiating a deposit.
</Note>

## Minimum Amounts

Each asset has a `minCheckoutUsd` value, the minimum deposit amount in USD equivalent. Deposits below this threshold may fail to process, so check it before you tell a user how much to send.

Most L2 chains (Polygon, Arbitrum, Base, Optimism) have low minimums of $2, while Ethereum deposits require $7 minimum. Bitcoin and Tron have \$9 minimums due to higher bridging costs.

## Next Steps

<CardGroup cols={2}>
  <Card title="Create Deposit" icon="arrow-right-to-bracket" href="/trading/bridge/deposit">
    Generate bridge addresses for your wallet.
  </Card>

  <Card title="Check Status" icon="clock" href="/trading/bridge/status">
    Track your deposit progress.
  </Card>
</CardGroup>
