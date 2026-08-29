> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Tiers

> Rate limits, rewards, and how to upgrade

The Builder Program uses a tiered system to manage rate limits while rewarding high-performing integrations. Higher tiers unlock increased limits, weekly rewards, and priority support.

## Feature Definitions

| Feature                     | Description                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| **Daily Relayer Txn Limit** | Maximum Relayer transactions per day for deposit wallet, Safe, and Proxy wallet operations |
| **API Rate Limits**         | Rate limits for non-relayer endpoints (CLOB, Gamma, etc.)                                  |
| **Gasless Trading**         | Gas fees subsidized for supported smart-wallet operations                                  |
| **Order Attribution**       | Orders tracked and attributed to your Builder profile                                      |
| **Builder Fees**            | Builders who route orders can charge fees and monetize on flow                             |
| **Leaderboard Visibility**  | Visibility on the [Builder Leaderboard](https://builders.polymarket.com/)                  |
| **Telegram Channel**        | Private Builders channel for announcements and support                                     |
| **Engineering Support**     | Direct access to engineering team                                                          |
| **Marketing Support**       | Promotion via official Polymarket social accounts                                          |
| **Priority Access**         | Early access to new features and products                                                  |

***

## Tier Comparison

| Feature                     | Unverified |  Verified  |  Partner  |
| --------------------------- | :--------: | :--------: | :-------: |
| **Daily Relayer Txn Limit** |   100/day  | 10,000/day | Unlimited |
| **API Rate Limits**         |  Standard  |  Standard  |  Highest  |
| **Gasless Trading\***       |     Yes    |     Yes    |    Yes    |
| **Order Attribution**       |     Yes    |     Yes    |    Yes    |
| **Builder Fees**            |     Yes    |     Yes    |    Yes    |
| **Leaderboard Visibility**  |      —     |     Yes    |    Yes    |
| **Telegram Channel**        |      —     |     Yes    |    Yes    |
| **Engineering Support**     |      —     |  Standard  |  Elevated |
| **Marketing Support**       |      —     |  Standard  |  Elevated |
| **Priority Access**         |      —     |      —     |    Yes    |

***

## Unverified

<Card title="100 Relay transactions/day" icon="seedling">
  The default tier for all new builders. Start immediately with no approval
  required.
</Card>

**How to get started:**

1. Go to [polymarket.com/settings?tab=builder](https://polymarket.com/settings?tab=builder)
2. Create a builder profile
3. Click **"+ Create New"** to generate API keys
4. Attach your [builder code](/trading/place-orders#builder-attribution) to CLOB orders for attribution; use a Relayer API key for gasless wallet operations

**What's included:**

* Gasless trading through deposit wallets for new API users and existing Safe/Proxy wallets
* Gas subsidized on all Relayer transactions up to the daily limit
* Access to all client libraries and documentation

***

## Verified

<Card title="10,000 Relay transactions/day" icon="badge-check">
  For builders who need higher throughput. Requires manual approval.
</Card>

**How to upgrade:**

Contact us at [builder@polymarket.com](mailto:builder@polymarket.com) with:

* Your Builder API Key
* Use case description
* Expected volume
* Other relevant information (links, docs, decks, etc.)

**Unlocks over Unverified:**

* 100x daily Relayer transaction limit
* Monetize with Builder fees
* Leaderboard visibility at [builders.polymarket.com](https://builders.polymarket.com)
* Private Telegram channel for announcements and support
* Weekly USDC rewards based on volume (subject to approval)
* Grants (subject to approval)

***

## Partner

<Card title="Unlimited Relay transactions/day" icon="handshake">
  Enterprise tier for high-volume integrations and strategic partners.
</Card>

**Unlocks over Verified:**

* Unlimited Relayer transactions
* Highest API rate limits
* Elevated engineering support
* Elevated and coordinated marketing support
* Priority access to new features and products

***

## How to Upgrade

<Steps>
  <Step title="Build and Launch">
    Start with the Unverified tier and build your integration.
  </Step>

  <Step title="Generate Volume">
    Route orders through Polymarket and demonstrate consistent usage.
  </Step>

  <Step title="Apply for Verification">
    Email [builder@polymarket.com](mailto:builder@polymarket.com) with your
    builder key and use case.
  </Step>

  <Step title="Get Approved">
    The Polymarket team reviews applications and responds within a few business
    days.
  </Step>
</Steps>

## Contact

Ready to upgrade or have questions?

<Card title="builder@polymarket.com" icon="envelope" href="mailto:builder@polymarket.com">
  Email us with your Builder API Key and use case details.
</Card>

## FAQ

<AccordionGroup>
  <Accordion title="How do I know if I am verified">
    Verification is displayed in your [Builder Profile](https://polymarket.com/settings?tab=builder) settings.
  </Accordion>

  <Accordion title="What happens if I exceed my daily limit">
    Relayer requests beyond your daily limit will be rate-limited and return an
    error. Consider upgrading to Verified or Partner tier if you're hitting
    limits.
  </Accordion>

  <Accordion title="What if I just need more daily Relay transaction limits for my own wallet">
    If you're not routing orders for other users (wallets), you can get unlimited
    daily Relay transactions by obtaining a [Relayer API key](https://polymarket.com/settings?tab=api-keys).
  </Accordion>
</AccordionGroup>
