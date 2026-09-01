> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Fees

> Tiered maker and taker trading fees for Polymarket Perps

Perps trading fees are tiered by an account's trailing 30-day trading volume.
Higher-volume accounts pay lower taker fees, and the top tier earns a maker
rebate instead of paying a maker fee.

## Fee Calculation

For each fill, the fee is calculated on the notional value of the trade:

```text theme={null}
Fee = abs(Price * Quantity) * Rate
```

Fees are denominated in the instrument's quote asset (pUSD). The rate applied
to a fill is set by the account's current volume tier.

| 30-Day Volume ≥ | Taker   | Maker    |
| --------------- | ------- | -------- |
| \$0             | 0.0400% | 0.0125%  |
| \$1M            | 0.0370% | 0.0100%  |
| \$5M            | 0.0350% | 0.0080%  |
| \$25M           | 0.0300% | 0.0050%  |
| \$100M          | 0.0270% | 0.0020%  |
| \$500M          | 0.0250% | 0.0000%  |
| \$1B            | 0.0200% | -0.0050% |

New accounts start at the \$0 tier and move up as trailing 30-day volume crosses
each threshold. Fee tiers are re-evaluated every UTC day.

A negative maker fee is a rebate: the maker receives the rebate amount, and the
fee recipient's internal ledger is debited by the same amount.

<Note>
  A subset of accounts created during the Perps beta are temporarily on the
  top-tier fee schedule regardless of trailing 30-day volume. Standard
  volume-based tiering applies to these accounts once the transition period
  ends.
</Note>

If you're integrating Perps, read the current fee schedule from
[Trading Fees](/perps/trading#trading-fees).

## Fee Metrics

Trailing 7-day activity metrics are available for visibility. They are a
rolling view of recent activity and do not, on their own, determine the volume
tier used to set fees.

| Metric              | Meaning                                                                        |
| ------------------- | ------------------------------------------------------------------------------ |
| Total volume        | Total Perps trading volume                                                     |
| Taker volume        | Perps volume that removed liquidity                                            |
| Maker volume        | Perps volume that added liquidity                                              |
| Account maker share | Account maker volume divided by total exchange volume                          |
| Entity maker share  | Entity maker volume divided by total exchange volume, when the account has one |

These metrics are cached by UTC day and may be stale by up to 24 hours.

If you're integrating Perps, read account metrics from
[Account Stats](/perps/account-management#account-stats).

## Fee Accounting

Every fill's fee flows through a single fee-recipient account on the internal
ledger:

* Taker fees credit the recipient.
* Maker fees credit the recipient at every tier where the maker rate is
  non-negative.
* At the top tier the maker rate is a rebate, so it debits the recipient and
  credits the maker.
