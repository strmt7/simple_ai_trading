> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.us/llms.txt
> Use this file to discover all available pages before exploring further.

# RFQ API Overview

> Create and manage combo RFQs and quotes through the Retail API

<Note>
  **Beta access required.** The Retail RFQ API is available only to explicitly enabled Retail API users.
</Note>

An RFQ requests two-sided liquidity for the exact symbol of a [combo instrument](/api-reference/combos/overview). All calls use normal [Retail API authentication](/api-reference/authentication) at `https://api.polymarket.us`.

The Retail API derives the participant and account from the API key; clients do not send an account.

## Endpoints

| Method   | Endpoint                                    | Description                                |
| -------- | ------------------------------------------- | ------------------------------------------ |
| `GET`    | `/v1/rfqs/user-id`                          | Get your pseudonymous RFQ user ID          |
| `GET`    | `/v1/rfqs`                                  | Query visible RFQs                         |
| `POST`   | `/v1/rfqs`                                  | Create an RFQ                              |
| `DELETE` | `/v1/rfqs/{rfqId}`                          | Close your open RFQ                        |
| `GET`    | `/v1/rfqs/quotes`                           | Query visible quotes                       |
| `POST`   | `/v1/rfqs/quotes`                           | Create or replace your quote               |
| `DELETE` | `/v1/rfqs/{rfqId}/quotes/{quoteId}`         | Delete your quote                          |
| `PUT`    | `/v1/rfqs/{rfqId}/quotes/{quoteId}/accept`  | Accept one side of a quote                 |
| `PUT`    | `/v1/rfqs/{rfqId}/quotes/{quoteId}/confirm` | Confirm an accepted quote during last look |

On the Retail API, Combo and RFQ creation share an additional [edge rate limit](/api-reference/rate-limits) of 10 requests per 10 seconds, enforced per API key and per IP. RFQ-specific business limits and participant restrictions are enforced separately by the RFQ service.

For `GET /v1/rfqs/quotes`, provide an `rfqId` or exactly one of `userFilter=USER_FILTER_SELF` and `rfqUserFilter=USER_FILTER_SELF`. Cursors are opaque and must be reused with the same filters and authenticated participant.

## Real-time stream

RFQ and quote lifecycle events are available on the [Private WebSocket](/api-reference/websocket/private#rfq-subscriptions). Subscribe with `SUBSCRIPTION_TYPE_RFQ` to receive all seven RFQ and quote event types, from `rfqCreated` through `quoteExecuted`.

The stream is live and best effort, with no replay or separate subscription acknowledgment. On startup or reconnect, reconcile state with `GET /v1/rfqs` and `GET /v1/rfqs/quotes`.

## Execution

`QUOTE_STATUS_EXECUTED` and `quoteExecuted` mean the paired exchange orders were submitted and their order IDs were recorded. They do not mean the orders filled.

Use `SUBSCRIPTION_TYPE_RFQ` for the RFQ lifecycle and `SUBSCRIPTION_TYPE_ORDER` for fills, rejections, cancellations, and expirations. Correlate the orders using `creatorOrderId` for the maker and `rfqCreatorOrderId` for the requester; both IDs are also returned by `GET /v1/rfqs/quotes`.

RFQ orders enter the normal combo order book and may trade with other resting liquidity. Combo instruments can also be traded directly through the [Orders API](/api-reference/orders/overview); using an RFQ is optional. If either `restRemainder` setting is true, unfilled quantity on that side may remain on the book.

## See Also

<CardGroup cols={2}>
  <Card title="Combos API" icon="shuffle" href="/api-reference/combos/overview">
    Create and read combo instruments
  </Card>

  <Card title="Private WebSocket" icon="bolt" href="/api-reference/websocket/private#rfq-subscriptions">
    Receive all seven RFQ event variants
  </Card>

  <Card title="Authentication" icon="key" href="/api-reference/authentication">
    Sign Retail API requests
  </Card>
</CardGroup>
