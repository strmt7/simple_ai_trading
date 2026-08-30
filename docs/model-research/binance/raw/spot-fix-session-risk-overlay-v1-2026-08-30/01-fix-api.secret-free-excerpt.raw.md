# FIX API

> [!NOTE]
> This API can only be used with the SPOT Exchange.

## General API Information

* FIX connections require TLS encryption. Please either use native TCP+TLS connection or set up a local proxy such as [stunnel](https://www.stunnel.org/) to handle TLS encryption.
* APIs have a timeout of 10 seconds when processing a request. If a response from the Matching Engine takes longer than this, the API responds with "Timeout waiting for response from backend server. Send status unknown; execution status unknown." [(-1007 TIMEOUT)](errors.md#-1007-timeout)
  * This does not always mean that the request failed in the Matching Engine.
  * If the status of the request has not appeared in [User Data Stream](user-data-stream.md), please perform an API query for its status.
* If your request contains a symbol name containing non-ASCII characters, then the response may contain non-ASCII characters encoded in UTF-8.
* To ensure uninterrupted connectivity, please make sure that your client sends **SNI (Server Name Indication)** during the TLS handshake and performs certificate validation against the intended hostname. <br>
Clients that do not send SNI may receive an unexpected certificate, which can result in TLS handshake or hostname verification failures.

<details>
<summary>Example implementations</summary>

### NodeJS
If you are using Node.js and connecting via raw TLS sockets (`tls.connect()`), you must explicitly set the servername option. Please refer to the sample below:

```javascript
  const tls = require("tls");
  const hostname = "fix-dc.binance.com"; //EXAMPLE

  const options = {
     host: hostname,
     port: 9002,
     servername: hostname                // enables SNI
   };
```

Note that: NodeJS doesn't enable SNI by default for TLS (See [https://nodejs.org/api/tls.html#tlsconnectoptions-callback](https://nodejs.org/api/tls.html#tlsconnectoptions-callback)). <br>
If you are using standard HTTPS libraries in Node.js (e.g., `https.request()`, `axios`, `fetch`), these typically set SNI automatically when connecting via a hostname/URL.

### Other languages/custom TLS implementations
When using custom TLS agents / TLS APIs, ensure you set the equivalent field (often named `server_hostname`, `hostname`, or `ServerName`) to the endpoint hostname so SNI is sent.
</details>

**FIX sessions only support Ed25519 keys.** </br>

Please refer to [this tutorial](https://www.binance.com/en/support/faq/how-to-generate-an-ed25519-key-pair-to-send-api-requests-on-binance-6b9a63f1e3384cf48a2eedb82767a69a)
on how to set up an Ed25519 key pair.

### FIX API Order Entry sessions

* Endpoint is: `tcp+tls://fix-oe.binance.com:9000`
* Supports placing orders, canceling orders, and querying current limit usage.
* Supports receiving all of the account's [ExecutionReport`<8>`](#executionreport) and [List Status`<N>`](#liststatus).
* Only API keys with `FIX_API` are allowed to connect.
* QuickFIX Schema can be found [here](https://github.com/binance/binance-spot-api-docs/blob/master/fix/schemas/spot-fix-oe.xml).

### FIX API Drop Copy sessions

* Endpoint is: `tcp+tls://fix-dc.binance.com:9000`
* Supports receiving all of the account's [ExecutionReport`<8>`](#executionreport) and [List Status`<N>`](#liststatus).
* Only API keys with `FIX_API` or `FIX_API_READ_ONLY` are allowed to connect.
* QuickFIX Schema can be found [here](https://github.com/binance/binance-spot-api-docs/blob/master/fix/schemas/spot-fix-oe.xml).
* Data in Drop Copy sessions is delayed by 1 second.

### FIX API Market Data sessions

* Endpoint is: `tcp+tls://fix-md.binance.com:9000`
* Supports market data streams and active instruments queries.
* Does not support placing or canceling orders.
* Only API keys with `FIX_API` or `FIX_API_READ_ONLY` are allowed to connect.
* QuickFIX Schema can be found [here](https://github.com/binance/binance-spot-api-docs/blob/master/fix/schemas/spot-fix-md.xml).

### FIX Connection Lifecycle

* All FIX API sessions will remain open for as long as possible, on a best-effort basis.
* There is no minimum connection time guarantee; a server can enter maintenance at any time.
  * When a server enters maintenance, a [News `<B>`](#news) message will be sent to clients **every 10 seconds until disconnection**, prompting clients to reconnect. Upon receiving this message, a client is expected to establish a new session and close the old one. If the client does not close the old session before the server disconnects it, the server will proceed to log it out and close the session.
* After connecting, the client must send a Logon `<A>` request. For more information please refer to [How to sign a Logon request](#signaturecomputation).
* The client should send a Logout `<5>` message to close the session before disconnecting. Failure to send the logout message will result in the session’s `SenderCompID (49)` being unusable for new session establishment for a duration of 2x the `HeartInt (108)` interval.
* The system allows negotiation of the `HeartInt (108)` value during the logon process. Accepted values range between 5 and 60 seconds.
  * If the server has not sent any messages within a `HeartInt (108)` interval, a [HeartBeat `<0>`](#heartbeat)  will be sent.
  * If the server has not received any messages within a `HeartInt (108)` interval, a [TestRequest `<1>`](#testrequest) will be sent. If the server does not receive a HeartBeat `<0>` containing the expected `TestReqID (112)` from the client within `HeartInt (108)` seconds, the server will send a Logout `<5>` message and close the connection.
  * If the client has not received any messages within a `HeartInt (108)` interval, the client is responsible for sending a TestRequest `<1>` to ensure the connection is healthy. Upon receiving such a TestRequest `<1>`, the server will respond with a Heartbeat `<0>` containing the expected `TestReqID (112)`. If the client does not receive the server’s response within a `HeartInt (108)` interval, the client should close the session and connection and establish new ones.

### API Key Permissions

To access the FIX API order entry sessions, your API key must be configured with the `FIX_API` permission.

To access the FIX Drop Copy sessions, your API key must be configured with either `FIX_API_READ_ONLY` or `FIX_API` permission.

To access the FIX Market Data sessions, your API key must be configured with either `FIX_API` or `FIX_API_READ_ONLY` permission.

**FIX sessions only support Ed25519 keys.**

Please refer to [this tutorial](https://www.binance.com/en/support/faq/how-to-generate-an-ed25519-key-pair-to-send-api-requests-on-binance-6b9a63f1e3384cf48a2eedb82767a69a)
on how to set up an Ed25519 key pair.

<a id="orderedmode"></a>

### On message processing order

The `MessageHandling (25035)` field required in the initial [Logon`<A>`](#logon-request) message controls whether messages from the client may be reordered before they are processed by the Matching Engine.

| Mode            | Description                                                                                |
|-----------------|--------------------------------------------------------------------------------------------|
| `UNORDERED(1)`  | Messages from the client are allowed to be sent to the matching engine in any order.       |
| `SEQUENTIAL(2)` | Messages from the client are always sent to the matching engine in `MsgSeqNum (34)` order. |

In all modes, the client's `MsgSeqNum (34)` must increase monotonically, with each subsequent message having a sequence number that is exactly 1 greater than the previous message.

> [!TIP]
> `UNORDERED(1)` should offer better performance when there are multiple messages in flight from the client to the server.

<a id="responsemode"></a>

### Response Mode

By default, all concurrent order entry sessions receive all of the account's
successful [ExecutionReport`<8>`](#executionreport) and [ListStatus`<N>`](#liststatus) messages,
including those in response to orders placed from other FIX sessions and via non-FIX APIs.

Use the `ResponseMode (25036)` field in the initial [Logon`<A>`](#logon-request) message
to change this behavior.

- `EVERYTHING(1)`: The default mode.
- `ONLY_ACKS(2)`: Receive only ACK messages whether operation succeeded or failed. Disables ExecutionReport push.

<a id="timingsecurity"></a>

### Timing Security

* All requests require a `SendingTime(52)` field which should be the current timestamp.
* An additional optional field, `RecvWindow(25000)`, specifies for how long the request stays valid in milliseconds.
  * `RecvWindow(25000)` supports up to three decimal places of precision (e.g., 6000.346) so that microseconds may be specified.
  * If `RecvWindow(25000)` is not specified, it defaults to 5000 milliseconds only for the Logon`<A>` request. For other requests if unset, the RecvWindow check is not executed.
  * Maximum `RecvWindow(25000)` is 60000 milliseconds.
* Request processing logic is as follows:

```javascript
serverTime = getCurrentTime()
if (SendingTime < (serverTime + 1 second) && (serverTime - SendingTime) <= RecvWindow) {
  // begin processing request
  serverTime = getCurrentTime()
  if (serverTime - SendingTime) <= RecvWindow {
    // forward request to Matching Engine
  } else {
    // reject request
  }
  // finish processing request
} else {
  // reject request
}
```


### Message Limits

* Each connection has a limit on **how many messages can be sent to the exchange**.
* The message limit **does not count the messages sent in response to the client**.
* Breaching the message limit results in immediate [Logout `<5>`](#logout) and disconnection.
* To understand current limits and usage, please send a [LimitQuery`<XLQ>`](#limitquery) message.
  A [LimitResponse`<XLR>`](#limitresponse) message will be sent in response, containing information about Order Rate
  Limits and Message Limits.
* FIX Order entry sessions have a limit of 10,000 messages every 10 seconds.
* FIX Drop Copy sessions have a limit of 60 messages every 60 seconds.
* FIX Market Data sessions have a limit of 2000 messages every 60 seconds.

<a id="connection-limits"></a>

### Connection Limits

* Each Account has a limit on how many TCP connections can be established at the same time.
* The limit is reduced when the TCP connection is closed. If the reduction of connections is not immediate, please wait up to twice the value of `HeartBtInt (108)` for the change to take effect.
  For example, if the current value of `HeartBtInt` is 5, please wait up to 10 seconds.
* Upon breaching the limit a [Reject `<3>`](#reject) will be sent containing information about the connection limit
  breach and the current limit.
* FIX Order Entry limits:
   * 15 connection attempts within 30 seconds
   * Maximum of 10 concurrent TCP connections per account
* FIX Drop Copy limits:
    * 15 connection attempts within 30 seconds
    * Maximum of 10 concurrent TCP connections per account
* FIX Market Data limits
  * 300 connection attempts within 300 seconds
  * Maximum of 100 concurrent TCP connections per account
  * A single connection can listen to a maximum of 1000 streams.

### Unfilled Order Count

* To understand how many orders you have placed within a certain time interval, please send a [LimitQuery`<XLQ>`](#limitquery) message.
  A [LimitResponse`<XLR>`](#limitresponse) message will be sent in response, containing information about Unfilled Order Count and Message Limits.
* **Please note that if your orders are consistently filled by trades, you can continuously place orders on the API**. For more information, please see [Spot Unfilled Order Count Rules](./faqs/order_count_decrement.md).
* If you exceed the unfilled order count your message will be rejected, and information will be transferred back to you in a reject message specific to that endpoint.
* **The number of unfilled orders is tracked for each account.**

## Error Handling
| `STOP_LOSS_LIMIT`   | 38, 44, 59, 1102 or 25009       |                           |
| `TAKE_PROFIT`       | 38, 1102 or 25009               | This will execute a `MARKET` order when the conditions are met. (e.g. `TriggerPrice (1102)` is met or `TriggerTrailingDeltaBips (25009)` is activated)   |
| `TAKE_PROFIT_LIMIT` | 38, 44, 59, 1102 or 25009       |
| `LIMIT_MAKER`       | 38, 44                          | This is a `LIMIT` order that will be rejected if the order immediately matches and trades as a taker. <br/> This is also known as a POST-ONLY order. |

<a id="executionreport"></a>

#### ExecutionReport `<8>`

Sent by the server whenever an order state changes.

> [!NOTE]
> * By default, ExecutionReport`<8>` is sent for all orders of an account, including those submitted in different connections. Please see [Response Mode](#responsemode) for other behavior options.
> * FIX API should give better performance for ExecutionReport<code>&lt;8&gt;</code> push.

| Tag   | Name                     | Type         | Required | Description                                                                                                                                                                                                                                                                                                                  |
|-------|--------------------------|--------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
<a id="ordermasscancelrequest"></a>

#### OrderMassCancelRequest `<q>`

Sent by the client to cancel all open orders on a symbol.

> [!NOTE]
> All orders of the account will be canceled, including those placed in different connections.

| Tag | Name                  | Type   | Required | Description                                                                                                    |
|-----|-----------------------|--------|----------|----------------------------------------------------------------------------------------------------------------|
| 11  | ClOrdID               | STRING | Y        | `ClOrdId` of this mass cancel request.                                                                         |
| 55  | Symbol                | STRING | Y        | Symbol on which to cancel orders.                                                                              |
| 530 | MassCancelRequestType | CHAR   | Y        | Possible values: <br></br> `1` - CANCEL_SYMBOL_ORDERS                                                               |

**Sample message:**

```
8=FIX.4.4|9=95|35=q|34=2|49=dpYPesqv|52=20240613-01:24:36.948|56=SPOT|11=1718241876901971671|55=BTCUSDT|530=1|10=243|
```

**Responses:**

* [ExecutionReport`<8>`](#executionreport) with `ExecType (150)` value `CANCELED (4)` for the every order canceled.
* [OrderMassCancelReport`<r>`](#ordermasscancelreport) with `MassCancelResponse (531)` field indicating whether the message is accepted or rejected.
* [Reject`<3>`](#reject) if the message is rejected.
