> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Collateral Return

> Release pUSD from compatible Combo positions before resolution

Quoting Combos can be capital-intensive because pUSD may remain locked across
related positions until every underlying market resolves. Collateral return
identifies capital that no longer needs to remain locked and returns it as pUSD,
making it available for new quotes while preserving any unmatched exposure.

## How Collateral Return Works

Collateral return finds compatible Combo positions that offset one another. It
decomposes those positions, merges complementary exposure back into pUSD, and
leaves the remaining exposure in residual positions.

For example, let `Y(A)` and `N(A)` represent the YES and NO positions for market
`A`, and let `A ^ B ^ C` represent a three-leg Combo. Suppose the wallet holds:

```text theme={null}
N(A ^ B ^ C)
N(N(A) ^ B ^ C)
```

The first leg of each Combo can be decomposed:

```text theme={null}
N(A) + Y(A ^ N(B ^ C))
Y(A) + Y(N(A) ^ N(B ^ C))
```

`Y(A)` and `N(A)` are complementary, so they merge into 1 pUSD per matched
share. The two `Y(…)` positions remain in the wallet, preserving its unmatched
exposure to the other legs.

## Return Collateral

The workflow is plan-based: inspect the proposed position changes, execute that
exact plan, wait for confirmation, and request another plan when more work
remains.

You need a Relayer API key. See [Connect Your
Account](/trading/wallets-auth#connect-your-account) to create one before
continuing.

<Tabs>
  <Tab title="TypeScript">
    Given a `SecureClient` configured with a Relayer API key, ensure the wallet has
    the required trading approvals:

    ```ts theme={null}
    import { createSecureClient, relayerApiKey } from "@polymarket/client";
    import { privateKey } from "@polymarket/client/viem";

    const client = await createSecureClient({
      wallet: process.env.POLYMARKET_WALLET_ADDRESS,
      signer: privateKey(process.env.SIGNER_PRIVATE_KEY),
      apiKey: relayerApiKey({
        key: process.env.RELAYER_API_KEY!,
        address: process.env.RELAYER_API_KEY_ADDRESS!,
      }),
    });

    await client.setupTradingApprovals();
    ```

    <Note>
      Collateral return supports Deposit Wallet, Safe Wallet, and Proxy Wallet
      accounts. EOA accounts are not supported.
    </Note>

    <Steps>
      <Step title="Request a Plan">
        First, request a plan based on the wallet's current Combo positions.

        ```ts theme={null}
        import type { CollateralReturnPlanResponse } from "@polymarket/client";

        const plan: CollateralReturnPlanResponse = await client.planCollateralReturn();
        ```

        <Accordion title="CollateralReturnPlanResponse">
          <CodeGroup>
            ```ts CollateralReturnPlanResponse Type theme={null}
            enum CollateralReturnKnownOperationKind {
              Split = "split",
              Merge = "merge",
              Redeem = "redeem",
              SplitOnCondition = "split_on_condition",
              MergeOnCondition = "merge_on_condition",
              SplitOnEvent = "split_on_event",
              MergeOnEvent = "merge_on_event",
              ConvertOnEvent = "convert_on_event",
              Extract = "extract",
              Inject = "inject",
              ConvertToYesBasket = "convert_to_yes_basket",
              MergeFromYesBasket = "merge_from_yes_basket",
              Compress = "compress",
            }

            type CollateralReturnOperationKind =
              | CollateralReturnKnownOperationKind
              | (string & {});

            type CollateralReturnOperation = {
              kind: CollateralReturnOperationKind;
              conditionId?: ComboConditionId;
              eventId?: EventId;
              positionId?: PositionId;
              conditionIndex: number;
              amount: DecimalString;
            };

            type CollateralReturnPositionAmount = {
              positionId: PositionId;
              amount: DecimalString;
            };

            type CollateralReturnPositionSummary = {
              consumed: CollateralReturnPositionAmount[];
              created: CollateralReturnPositionAmount[];
            };

            type CollateralReturnRouterCall = {
              to: EvmAddress;
              data: HexString;
            };

            type CollateralReturnPlanResponse = {
              planHash: HexString;
              chainId: number;
              wallet: EvmAddress;
              blockNumber: bigint;
              startingPusd: DecimalString;
              netPusdOut: DecimalString;
              finalPusd: DecimalString;
              operations: CollateralReturnOperation[];
              operationCount: number;
              truncated: boolean;
              estimatedCost: number;
              requiredPusdInput: DecimalString;
              requiredPositions: CollateralReturnPositionAmount[];
              positionSummary: CollateralReturnPositionSummary;
              candidatePositionIds: PositionId[];
              routerCall: CollateralReturnRouterCall;
            };
            ```

            ```json CollateralReturnPlanResponse Example theme={null}
            {
              "planHash": "<plan_hash>",
              "chainId": 137,
              "wallet": "<wallet_address>",
              "blockNumber": "74281963",
              "startingPusd": "5.000000",
              "netPusdOut": "1.000000",
              "finalPusd": "6.000000",
              "operations": [
                {
                  "kind": "merge_on_condition",
                  "conditionId": "<condition_id>",
                  "positionId": "<position_id>",
                  "conditionIndex": 0,
                  "amount": "1.000000"
                }
              ],
              "operationCount": 1,
              "truncated": false,
              "estimatedCost": 240000,
              "requiredPusdInput": "0.000000",
              "requiredPositions": [
                {
                  "positionId": "<position_id>",
                  "amount": "1.000000"
                }
              ],
              "positionSummary": {
                "consumed": [
                  {
                    "positionId": "<position_id>",
                    "amount": "1.000000"
                  }
                ],
                "created": [
                  {
                    "positionId": "<residual_position_id>",
                    "amount": "0.500000"
                  }
                ]
              },
              "candidatePositionIds": [
                "<candidate_position_id_1>",
                "<candidate_position_id_2>"
              ],
              "routerCall": {
                "to": "<contract_address>",
                "data": "<calldata>"
              }
            }
            ```
          </CodeGroup>
        </Accordion>
      </Step>

      <Step title="Inspect the Plan">
        Next, inspect the plan before signing. It is an inspectable execution artifact,
        not just a transaction payload. Review its expected return and residual-position
        impact, and apply any application-specific limits.

        | Field                                     | What to inspect                                                      |
        | ----------------------------------------- | -------------------------------------------------------------------- |
        | `startingPusd`, `netPusdOut`, `finalPusd` | The expected pUSD return and resulting wallet balance.               |
        | `requiredPositions`                       | The Combo positions the plan requires.                               |
        | `operations`                              | The ordered position changes.                                        |
        | `positionSummary`                         | The positions consumed and the residual positions created.           |
        | `estimatedCost`                           | The estimated execution cost against any application-specific limit. |
        | `truncated`                               | Whether more work remains after this plan confirms.                  |
        | `routerCall`                              | The exact action that will be submitted.                             |
      </Step>

      <Step title="Execute the Inspected Plan">
        Finally, execute the inspected plan and wait for the transaction to confirm.

        ```ts theme={null}
        const handle = await client.executeCollateralReturnPlan({ plan });
        const outcome = await handle.wait();

        // outcome.transactionHash: TxHash
        ```

        `await handle.wait()` returns after the submitted plan's transaction confirms.
        Request another plan only then, because the next plan is calculated from the
        resulting onchain state.

        <Info>
          If execution fails, discard the plan and request a new one before retrying.
        </Info>
      </Step>
    </Steps>
  </Tab>

  <Tab title="Python">
    Given an `AsyncSecureClient` configured with a Relayer API key, ensure the wallet
    has the required trading approvals:

    ```python theme={null}
    import os

    from polymarket import AsyncSecureClient, RelayerApiKey

    client = await AsyncSecureClient.create(
        private_key=os.environ["SIGNER_PRIVATE_KEY"],
        wallet=os.environ["POLYMARKET_WALLET_ADDRESS"],
        api_key=RelayerApiKey(
            key=os.environ["POLYMARKET_RELAYER_API_KEY"],
            address=os.environ["POLYMARKET_RELAYER_API_KEY_ADDRESS"],
        ),
    )

    await client.setup_trading_approvals()
    ```

    The sync `SecureClient` provides the same methods.

    <Note>
      Collateral return supports Deposit Wallet, Safe Wallet, and Proxy Wallet
      accounts. EOA accounts are not supported.
    </Note>

    <Steps>
      <Step title="Request a Plan">
        First, request a plan based on the wallet's current Combo positions.

        ```python theme={null}
        from polymarket import CollateralReturnPlanResponse

        plan: CollateralReturnPlanResponse = await client.plan_collateral_return()
        ```

        <Accordion title="CollateralReturnPlanResponse">
          <CodeGroup>
            ```python CollateralReturnPlanResponse Type theme={null}
            class CollateralReturnOperationKind(StrEnum):
                SPLIT = "split"
                MERGE = "merge"
                REDEEM = "redeem"
                SPLIT_ON_CONDITION = "split_on_condition"
                MERGE_ON_CONDITION = "merge_on_condition"
                SPLIT_ON_EVENT = "split_on_event"
                MERGE_ON_EVENT = "merge_on_event"
                CONVERT_ON_EVENT = "convert_on_event"
                EXTRACT = "extract"
                INJECT = "inject"
                CONVERT_TO_YES_BASKET = "convert_to_yes_basket"
                MERGE_FROM_YES_BASKET = "merge_from_yes_basket"
                COMPRESS = "compress"


            class CollateralReturnOperation:
                kind: CollateralReturnOperationKind | str
                condition_id: ComboConditionId | None
                event_id: EventId | None
                position_id: PositionId | None
                condition_index: int
                amount: Decimal


            class CollateralReturnPositionAmount:
                position_id: PositionId
                amount: Decimal


            class CollateralReturnPositionSummary:
                consumed: tuple[CollateralReturnPositionAmount, ...]
                created: tuple[CollateralReturnPositionAmount, ...]


            class CollateralReturnRouterCall:
                to: EvmAddress
                data: HexString


            class CollateralReturnPlanResponse:
                plan_hash: HexString
                chain_id: int
                wallet: EvmAddress
                block_number: int
                starting_pusd: Decimal
                net_pusd_out: Decimal
                final_pusd: Decimal
                operations: tuple[CollateralReturnOperation, ...]
                operation_count: int
                truncated: bool
                estimated_cost: float
                required_pusd_input: Decimal
                required_positions: tuple[CollateralReturnPositionAmount, ...]
                position_summary: CollateralReturnPositionSummary
                candidate_position_ids: tuple[PositionId, ...]
                router_call: CollateralReturnRouterCall
            ```

            ```json CollateralReturnPlanResponse Example theme={null}
            {
              "plan_hash": "<plan_hash>",
              "chain_id": 137,
              "wallet": "<wallet_address>",
              "block_number": 74281963,
              "starting_pusd": "5.000000",
              "net_pusd_out": "1.000000",
              "final_pusd": "6.000000",
              "operations": [
                {
                  "kind": "merge_on_condition",
                  "condition_id": "<condition_id>",
                  "position_id": "<position_id>",
                  "condition_index": 0,
                  "amount": "1.000000"
                }
              ],
              "operation_count": 1,
              "truncated": false,
              "estimated_cost": 240000,
              "required_pusd_input": "0.000000",
              "required_positions": [
                {
                  "position_id": "<position_id>",
                  "amount": "1.000000"
                }
              ],
              "position_summary": {
                "consumed": [
                  {
                    "position_id": "<position_id>",
                    "amount": "1.000000"
                  }
                ],
                "created": [
                  {
                    "position_id": "<residual_position_id>",
                    "amount": "0.500000"
                  }
                ]
              },
              "candidate_position_ids": [
                "<candidate_position_id_1>",
                "<candidate_position_id_2>"
              ],
              "router_call": {
                "to": "<contract_address>",
                "data": "<calldata>"
              }
            }
            ```
          </CodeGroup>
        </Accordion>
      </Step>

      <Step title="Inspect the Plan">
        Next, inspect the plan before signing. Review its expected return and
        residual-position impact, and apply any application-specific limits.

        | Field                                         | What to inspect                                                      |
        | --------------------------------------------- | -------------------------------------------------------------------- |
        | `starting_pusd`, `net_pusd_out`, `final_pusd` | The expected pUSD return and resulting wallet balance.               |
        | `required_positions`                          | The Combo positions the plan requires.                               |
        | `operations`                                  | The ordered position changes.                                        |
        | `position_summary`                            | The positions consumed and the residual positions created.           |
        | `estimated_cost`                              | The estimated execution cost against any application-specific limit. |
        | `truncated`                                   | Whether more work remains after this plan confirms.                  |
        | `router_call`                                 | The exact action that will be submitted.                             |
      </Step>

      <Step title="Execute the Inspected Plan">
        Finally, execute the inspected plan and wait for the transaction to confirm.

        ```python theme={null}
        handle = await client.execute_collateral_return_plan(plan=plan)
        outcome = await handle.wait()

        # outcome.transaction_hash: TransactionHash
        ```

        `await handle.wait()` returns after the submitted plan's transaction confirms.
        Request another plan only then, because the next plan is calculated from the
        resulting onchain state.

        <Info>
          If execution fails, discard the plan and request a new one before retrying.
        </Info>
      </Step>
    </Steps>
  </Tab>

  <Tab title="API">
    <Note>
      The following steps show the Deposit Wallet path. For Safe or Proxy Wallet
      flows, use an SDK or see the reference examples at the end of this tab.
    </Note>

    Before requesting a plan, ensure the wallet has the approvals below. Follow [Set
    Up Trading Approvals](/trading/wallets-auth#set-up-trading-approvals) to check,
    encode, and submit the required approval calls.

    | Asset           | Contract to approve      | Contract call                                     |
    | --------------- | ------------------------ | ------------------------------------------------- |
    | pUSD            | Collateral Return Router | `approve(CollateralReturnRouter, maxUint256)`     |
    | Combo positions | Collateral Return Router | `setApprovalForAll(CollateralReturnRouter, true)` |

    Use the following production contract addresses when building the approval
    calls.

    | Contract                 | Address                                      |
    | ------------------------ | -------------------------------------------- |
    | pUSD                     | `0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB` |
    | Position Manager         | `0x006F54F7f9A22e0000CC2AB60031000000ae9fEF` |
    | Collateral Return Router | `0x12121212006e4CD160D18e3f00711DA5c3372600` |

    Follow these steps to request and execute a collateral-return plan.

    <Steps>
      <Step title="Request a Plan">
        First, request a plan for the Deposit Wallet whose Combo positions you want to
        process.

        ```bash theme={null}
        curl -X POST "https://combos-rfq-collateral-return.polymarket.com/v1/collateral-return/plan" \
          -H "Content-Type: application/json" \
          --data '{
            "wallet": "<deposit_wallet_address>"
          }'
        ```

        The plan is calculated from one pinned block and contains the position changes
        and Router call for that state snapshot.

        <Accordion title="Collateral Return Plan">
          ```json theme={null}
          {
            "plan_hash": "<plan_hash>",
            "chain_id": 137,
            "wallet": "<deposit_wallet_address>",
            "block_number": "74281963",
            "starting_pusd": "5.000000",
            "net_pusd_out": "1.000000",
            "final_pusd": "6.000000",
            "operations": [
              {
                "kind": "merge_on_condition",
                "condition_id": "<condition_id>",
                "position_id": "<position_id>",
                "condition_index": 0,
                "amount": "1000000"
              }
            ],
            "operation_count": 1,
            "truncated": false,
            "estimated_cost": 240000,
            "required_positions": [
              {
                "position_id": "<position_id>",
                "amount": "1000000"
              }
            ],
            "position_summary": {
              "consumed": [
                {
                  "position_id": "<consumed_position_id>",
                  "amount": "1000000"
                }
              ],
              "created": [
                {
                  "position_id": "<residual_position_id>",
                  "amount": "1000000"
                }
              ]
            },
            "candidate_position_ids": [
              "<candidate_position_id_1>",
              "<candidate_position_id_2>"
            ],
            "router_call": {
              "to": "<router_address>",
              "data": "<calldata>"
            }
          }
          ```
        </Accordion>

        Nested `amount` values use six-decimal base units, so `1000000` represents one
        position unit.
      </Step>

      <Step title="Inspect the Plan">
        Next, inspect the plan before signing. Review its expected return,
        residual-position impact, and execution cost against any application-specific
        limits.

        | Field                                         | What to inspect                                                      |
        | --------------------------------------------- | -------------------------------------------------------------------- |
        | `starting_pusd`, `net_pusd_out`, `final_pusd` | The expected pUSD return and resulting wallet balance.               |
        | `required_positions`                          | The Combo positions the plan consumes.                               |
        | `position_summary`                            | The positions consumed and created, including residual positions.    |
        | `operations`                                  | The ordered position changes.                                        |
        | `estimated_cost`                              | The estimated execution cost against any application-specific limit. |
        | `truncated`                                   | Whether more work remains after this plan confirms.                  |
        | `router_call`                                 | The exact Router action that the wallet will authorize.              |
        | `plan_hash`                                   | The correlation value to send with the signed wallet action.         |
      </Step>

      <Step title="Fetch the Wallet Nonce">
        Then, fetch a fresh Deposit Wallet nonce for the signer from the Relayer.

        ```bash theme={null}
        curl -G "https://relayer-v2.polymarket.com/v1/account/transactions/params" \
          -H "RELAYER_API_KEY: <relayer_api_key>" \
          -H "RELAYER_API_KEY_ADDRESS: <signer_address>" \
          --data-urlencode "address=<signer_address>" \
          --data-urlencode "type=WALLET"
        ```

        ```json Response theme={null}
        {
          "address": "<signer_address>",
          "nonce": "<wallet_nonce>"
        }
        ```
      </Step>

      <Step title="Create the Batch Typed Data">
        Next, create Deposit Wallet EIP-712 `Batch` typed data containing exactly one
        call. Copy `router_call.to` and `router_call.data` from the plan without
        modification, and use `0` for the call value. Set `deadline` to a future Unix
        timestamp; the complete example below uses ten minutes from the current time.

        ```json walletBatchTypedData theme={null}
        {
          "domain": {
            "name": "DepositWallet",
            "version": "1",
            "chainId": 137,
            "verifyingContract": "<deposit_wallet_address>"
          },
          "types": {
            "Call": [
              { "name": "target", "type": "address" },
              { "name": "value", "type": "uint256" },
              { "name": "data", "type": "bytes" }
            ],
            "Batch": [
              { "name": "wallet", "type": "address" },
              { "name": "nonce", "type": "uint256" },
              { "name": "deadline", "type": "uint256" },
              { "name": "calls", "type": "Call[]" }
            ]
          },
          "primaryType": "Batch",
          "message": {
            "wallet": "<deposit_wallet_address>",
            "nonce": "<wallet_nonce>",
            "deadline": "<unix_seconds>",
            "calls": [
              {
                "target": "<router_call.to>",
                "value": "0",
                "data": "<router_call.data>"
              }
            ]
          }
        }
        ```
      </Step>

      <Step title="Sign the Batch">
        Then, sign the typed data with the signer that controls the Deposit Wallet. The
        example below uses Viem.

        ```ts theme={null}
        import { privateKeyToAccount } from "viem/accounts";

        const signer = privateKeyToAccount(
          process.env.SIGNER_PRIVATE_KEY as `0x${string}`,
        );
        const signature = await signer.signTypedData(walletBatchTypedData);
        ```
      </Step>

      <Step title="Execute the Plan">
        Next, send the plan hash and signed Deposit Wallet envelope to the Collateral
        Return API. Use the same nonce, deadline, and Router call that you signed.

        ```bash theme={null}
        curl -X POST "https://combos-rfq-collateral-return.polymarket.com/v1/collateral-return/submit" \
          -H "Content-Type: application/json" \
          -H "RELAYER_API_KEY: <relayer_api_key>" \
          -H "RELAYER_API_KEY_ADDRESS: <signer_address>" \
          --data '{
            "plan_hash": "<plan_hash>",
            "envelope": {
              "type": "WALLET",
              "from": "<signer_address>",
              "to": "0x00000000000Fb5C9ADea0298D729A0CB3823Cc07",
              "nonce": "<wallet_nonce>",
              "signature": "<signature>",
              "metadata": "Return Combo collateral",
              "depositWalletParams": {
                "depositWallet": "<deposit_wallet_address>",
                "deadline": "<unix_seconds>",
                "calls": [
                  {
                    "target": "<router_call.to>",
                    "value": "0",
                    "data": "<router_call.data>"
                  }
                ]
              }
            }
          }'
        ```

        The Collateral Return service validates and simulates the signed action against
        current state before forwarding it to the Relayer. The response includes the
        transaction ID used in the next step.

        ```json Response theme={null}
        {
          "transactionID": "<transaction_id>",
          "state": "STATE_NEW"
        }
        ```

        <Info>
          If submission returns `409 fresh plan required`, discard the plan and request
          a new one before rebuilding and signing the wallet action.
        </Info>
      </Step>

      <Step title="Confirm the Transaction">
        Finally, poll the Relayer transaction until it reaches `STATE_CONFIRMED`.

        ```bash theme={null}
        curl "https://relayer-v2.polymarket.com/v1/account/transactions/<transaction_id>" \
          -H "RELAYER_API_KEY: <relayer_api_key>" \
          -H "RELAYER_API_KEY_ADDRESS: <signer_address>"
        ```

        ```json Response theme={null}
        {
          "transaction_id": "<transaction_id>",
          "transaction_hash": "<transaction_hash>",
          "state": "STATE_CONFIRMED",
          "error_msg": null
        }
        ```

        | State                                        | Action                                                  |
        | -------------------------------------------- | ------------------------------------------------------- |
        | `STATE_NEW`, `STATE_EXECUTED`, `STATE_MINED` | Continue polling.                                       |
        | `STATE_CONFIRMED`                            | The transaction confirmed successfully.                 |
        | `STATE_FAILED`, `STATE_INVALID`              | Discard the plan and request a new one before retrying. |
      </Step>
    </Steps>

    The complete examples below use Viem.

    <CodeGroup>
      ```javascript Deposit Wallet theme={null}
      import { privateKeyToAccount } from "viem/accounts";

      const COLLATERAL_RETURN_API =
        "https://combos-rfq-collateral-return.polymarket.com";
      const RELAYER_API = "https://relayer-v2.polymarket.com";
      const DEPOSIT_WALLET_FACTORY = "0x00000000000Fb5C9ADea0298D729A0CB3823Cc07";

      const account = privateKeyToAccount(process.env.SIGNER_PRIVATE_KEY);
      const wallet = process.env.POLYMARKET_WALLET_ADDRESS;
      const relayerHeaders = {
        RELAYER_API_KEY: process.env.RELAYER_API_KEY,
        RELAYER_API_KEY_ADDRESS: account.address,
      };

      // Request a plan.
      const planResponse = await fetch(
        `${COLLATERAL_RETURN_API}/v1/collateral-return/plan`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ wallet }),
        },
      );
      if (!planResponse.ok) throw new Error(await planResponse.text());
      const plan = await planResponse.json();

      // Inspect the plan and apply any application-specific limits before signing.
      console.log({
        netPusdOut: plan.net_pusd_out,
        positionSummary: plan.position_summary,
        estimatedCost: plan.estimated_cost,
        truncated: plan.truncated,
      });

      // Fetch a fresh Deposit Wallet nonce.
      const executeParamsResponse = await fetch(
        `${RELAYER_API}/v1/account/transactions/params?address=${account.address}&type=WALLET`,
        { headers: relayerHeaders },
      );
      if (!executeParamsResponse.ok) {
        throw new Error(await executeParamsResponse.text());
      }
      const executeParams = await executeParamsResponse.json();

      // Create and sign the Deposit Wallet batch.
      const deadline = BigInt(Math.floor(Date.now() / 1_000) + 600);
      const calls = [
        {
          target: plan.router_call.to,
          value: 0n,
          data: plan.router_call.data,
        },
      ];
      const signature = await account.signTypedData({
        domain: {
          name: "DepositWallet",
          version: "1",
          chainId: 137,
          verifyingContract: wallet,
        },
        types: {
          Call: [
            { name: "target", type: "address" },
            { name: "value", type: "uint256" },
            { name: "data", type: "bytes" },
          ],
          Batch: [
            { name: "wallet", type: "address" },
            { name: "nonce", type: "uint256" },
            { name: "deadline", type: "uint256" },
            { name: "calls", type: "Call[]" },
          ],
        },
        primaryType: "Batch",
        message: {
          wallet,
          nonce: BigInt(executeParams.nonce),
          deadline,
          calls,
        },
      });
      const envelope = {
        type: "WALLET",
        from: account.address,
        to: DEPOSIT_WALLET_FACTORY,
        nonce: executeParams.nonce,
        signature,
        metadata: "Return Combo collateral",
        depositWalletParams: {
          depositWallet: wallet,
          deadline: deadline.toString(),
          calls: calls.map((call) => ({
            ...call,
            value: call.value.toString(),
          })),
        },
      };

      // Execute the plan.
      const submitResponse = await fetch(
        `${COLLATERAL_RETURN_API}/v1/collateral-return/submit`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", ...relayerHeaders },
          body: JSON.stringify({ plan_hash: plan.plan_hash, envelope }),
        },
      );
      if (!submitResponse.ok) throw new Error(await submitResponse.text());
      const submitted = await submitResponse.json();

      // Confirm the transaction.
      let outcome;
      for (;;) {
        const transactionResponse = await fetch(
          `${RELAYER_API}/v1/account/transactions/${submitted.transactionID}`,
          { headers: relayerHeaders },
        );
        if (!transactionResponse.ok) {
          throw new Error(await transactionResponse.text());
        }
        const transaction = await transactionResponse.json();

        if (transaction.state === "STATE_CONFIRMED") {
          outcome = transaction;
          break;
        }
        if (
          transaction.state === "STATE_FAILED" ||
          transaction.state === "STATE_INVALID"
        ) {
          throw new Error(transaction.error_msg ?? transaction.state);
        }

        await new Promise((resolve) => setTimeout(resolve, 1_000));
      }

      console.log(outcome);
      ```

      ```javascript Safe Wallet theme={null}
      import { hashTypedData, zeroAddress } from "viem";
      import { privateKeyToAccount } from "viem/accounts";

      const COLLATERAL_RETURN_API =
        "https://combos-rfq-collateral-return.polymarket.com";
      const RELAYER_API = "https://relayer-v2.polymarket.com";

      const account = privateKeyToAccount(process.env.SIGNER_PRIVATE_KEY);
      const wallet = process.env.POLYMARKET_WALLET_ADDRESS;
      const relayerHeaders = {
        RELAYER_API_KEY: process.env.RELAYER_API_KEY,
        RELAYER_API_KEY_ADDRESS: account.address,
      };

      // Request a plan.
      const planResponse = await fetch(
        `${COLLATERAL_RETURN_API}/v1/collateral-return/plan`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ wallet }),
        },
      );
      if (!planResponse.ok) throw new Error(await planResponse.text());
      const plan = await planResponse.json();

      // Inspect the plan and apply any application-specific limits before signing.
      console.log({
        netPusdOut: plan.net_pusd_out,
        positionSummary: plan.position_summary,
        estimatedCost: plan.estimated_cost,
        truncated: plan.truncated,
      });

      // Fetch a fresh Safe nonce.
      const executeParamsResponse = await fetch(
        `${RELAYER_API}/v1/account/transactions/params?address=${account.address}&type=SAFE`,
        { headers: relayerHeaders },
      );
      if (!executeParamsResponse.ok) {
        throw new Error(await executeParamsResponse.text());
      }
      const executeParams = await executeParamsResponse.json();

      // Create and sign the Safe transaction.
      const message = {
        to: plan.router_call.to,
        value: 0n,
        data: plan.router_call.data,
        operation: 0,
        safeTxGas: 0n,
        baseGas: 0n,
        gasPrice: 0n,
        gasToken: zeroAddress,
        refundReceiver: zeroAddress,
        nonce: BigInt(executeParams.nonce),
      };
      const digest = hashTypedData({
        domain: { chainId: 137, verifyingContract: wallet },
        types: {
          SafeTx: [
            { name: "to", type: "address" },
            { name: "value", type: "uint256" },
            { name: "data", type: "bytes" },
            { name: "operation", type: "uint8" },
            { name: "safeTxGas", type: "uint256" },
            { name: "baseGas", type: "uint256" },
            { name: "gasPrice", type: "uint256" },
            { name: "gasToken", type: "address" },
            { name: "refundReceiver", type: "address" },
            { name: "nonce", type: "uint256" },
          ],
        },
        primaryType: "SafeTx",
        message,
      });

      const signed = (await account.signMessage({ message: { raw: digest } })).slice(
        2,
      );
      const v = Number.parseInt(signed.slice(128, 130), 16);
      const packedV = v === 0 || v === 1 ? v + 31 : v === 27 || v === 28 ? v + 4 : v;
      const signature = `0x${signed.slice(0, 128)}${packedV.toString(16).padStart(2, "0")}`;
      const envelope = {
        type: "SAFE",
        from: account.address,
        to: message.to,
        proxyWallet: wallet,
        value: "0",
        data: message.data,
        nonce: executeParams.nonce,
        signature,
        metadata: "Return Combo collateral",
        signatureParams: {
          operation: "0",
          safeTxnGas: "0",
          baseGas: "0",
          gasPrice: "0",
          gasToken: zeroAddress,
          refundReceiver: zeroAddress,
        },
      };

      // Execute the plan.
      const submitResponse = await fetch(
        `${COLLATERAL_RETURN_API}/v1/collateral-return/submit`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", ...relayerHeaders },
          body: JSON.stringify({ plan_hash: plan.plan_hash, envelope }),
        },
      );
      if (!submitResponse.ok) throw new Error(await submitResponse.text());
      const submitted = await submitResponse.json();

      // Confirm the transaction.
      let outcome;
      for (;;) {
        const transactionResponse = await fetch(
          `${RELAYER_API}/v1/account/transactions/${submitted.transactionID}`,
          { headers: relayerHeaders },
        );
        if (!transactionResponse.ok) {
          throw new Error(await transactionResponse.text());
        }
        const transaction = await transactionResponse.json();

        if (transaction.state === "STATE_CONFIRMED") {
          outcome = transaction;
          break;
        }
        if (
          transaction.state === "STATE_FAILED" ||
          transaction.state === "STATE_INVALID"
        ) {
          throw new Error(transaction.error_msg ?? transaction.state);
        }

        await new Promise((resolve) => setTimeout(resolve, 1_000));
      }

      console.log(outcome);
      ```

      ```javascript Proxy Wallet theme={null}
      import {
        concat,
        createPublicClient,
        encodeFunctionData,
        http,
        keccak256,
        toHex,
      } from "viem";
      import { privateKeyToAccount } from "viem/accounts";
      import { polygon } from "viem/chains";

      const COLLATERAL_RETURN_API =
        "https://combos-rfq-collateral-return.polymarket.com";
      const RELAYER_API = "https://relayer-v2.polymarket.com";
      const PROXY_WALLET_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052";
      const RELAY_HUB = "0xD216153c06E857cD7f72665E0aF1d7D82172F494";

      const account = privateKeyToAccount(process.env.SIGNER_PRIVATE_KEY);
      const wallet = process.env.POLYMARKET_WALLET_ADDRESS;
      const publicClient = createPublicClient({
        chain: polygon,
        transport: http(process.env.RPC_URL),
      });
      const relayerHeaders = {
        RELAYER_API_KEY: process.env.RELAYER_API_KEY,
        RELAYER_API_KEY_ADDRESS: account.address,
      };

      // Request a plan.
      const planResponse = await fetch(
        `${COLLATERAL_RETURN_API}/v1/collateral-return/plan`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ wallet }),
        },
      );
      if (!planResponse.ok) throw new Error(await planResponse.text());
      const plan = await planResponse.json();

      // Inspect the plan and apply any application-specific limits before signing.
      console.log({
        netPusdOut: plan.net_pusd_out,
        positionSummary: plan.position_summary,
        estimatedCost: plan.estimated_cost,
        truncated: plan.truncated,
      });

      // Fetch a fresh Proxy Wallet nonce and relay address.
      const executeParamsResponse = await fetch(
        `${RELAYER_API}/v1/account/transactions/params?address=${account.address}&type=PROXY`,
        { headers: relayerHeaders },
      );
      if (!executeParamsResponse.ok) {
        throw new Error(await executeParamsResponse.text());
      }
      const executeParams = await executeParamsResponse.json();

      // Encode and sign the Proxy Wallet relay request.
      const data = encodeFunctionData({
        abi: [
          {
            type: "function",
            name: "proxy",
            inputs: [
              {
                type: "tuple[]",
                components: [
                  { name: "typeCode", type: "uint8" },
                  { name: "to", type: "address" },
                  { name: "value", type: "uint256" },
                  { name: "data", type: "bytes" },
                ],
              },
            ],
            outputs: [{ type: "bytes[]" }],
          },
        ],
        functionName: "proxy",
        args: [
          [
            {
              typeCode: 1,
              to: plan.router_call.to,
              value: 0n,
              data: plan.router_call.data,
            },
          ],
        ],
      });
      const relayerFee = 0n;
      const gasPrice = 0n;
      const gasLimit = await publicClient.estimateGas({
        account: account.address,
        to: PROXY_WALLET_FACTORY,
        data,
      });
      const digest = keccak256(
        concat([
          toHex("rlx:"),
          account.address,
          PROXY_WALLET_FACTORY,
          data,
          toHex(relayerFee, { size: 32 }),
          toHex(gasPrice, { size: 32 }),
          toHex(gasLimit, { size: 32 }),
          toHex(BigInt(executeParams.nonce), { size: 32 }),
          RELAY_HUB,
          executeParams.address,
        ]),
      );
      const signature = await account.signMessage({ message: { raw: digest } });
      const envelope = {
        type: "PROXY",
        from: account.address,
        to: PROXY_WALLET_FACTORY,
        proxyWallet: wallet,
        value: "0",
        data,
        nonce: executeParams.nonce,
        signature,
        metadata: "Return Combo collateral",
        signatureParams: {
          gasPrice: gasPrice.toString(),
          relayerFee: relayerFee.toString(),
          gasLimit: gasLimit.toString(),
          relayHub: RELAY_HUB,
          relay: executeParams.address,
        },
      };

      // Execute the plan.
      const submitResponse = await fetch(
        `${COLLATERAL_RETURN_API}/v1/collateral-return/submit`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", ...relayerHeaders },
          body: JSON.stringify({ plan_hash: plan.plan_hash, envelope }),
        },
      );
      if (!submitResponse.ok) throw new Error(await submitResponse.text());
      const submitted = await submitResponse.json();

      // Confirm the transaction.
      let outcome;
      for (;;) {
        const transactionResponse = await fetch(
          `${RELAYER_API}/v1/account/transactions/${submitted.transactionID}`,
          { headers: relayerHeaders },
        );
        if (!transactionResponse.ok) {
          throw new Error(await transactionResponse.text());
        }
        const transaction = await transactionResponse.json();

        if (transaction.state === "STATE_CONFIRMED") {
          outcome = transaction;
          break;
        }
        if (
          transaction.state === "STATE_FAILED" ||
          transaction.state === "STATE_INVALID"
        ) {
          throw new Error(transaction.error_msg ?? transaction.state);
        }

        await new Promise((resolve) => setTimeout(resolve, 1_000));
      }

      console.log(outcome);
      ```
    </CodeGroup>
  </Tab>
</Tabs>

## Handle a Truncated Plan

The service limits how much work one transaction can contain. When more work
remains, each plan is one complete, executable chunk—not a partial transaction.
Execute and confirm each chunk before requesting the next plan because every
new plan depends on the resulting onchain state.

<Tabs>
  <Tab title="TypeScript">
    Use `plan.truncated` to determine whether more work remains after the current
    plan.

    ```ts theme={null}
    let plan: CollateralReturnPlanResponse;

    do {
      plan = await client.planCollateralReturn();

      // Inspect the return and residual-position impact, and apply any
      // application-specific limits before signing.
      const handle = await client.executeCollateralReturnPlan({ plan });
      await handle.wait();
    } while (plan.truncated);
    ```
  </Tab>

  <Tab title="Python">
    Use `plan.truncated` to determine whether more work remains after the current
    plan.

    ```python theme={null}
    while True:
        plan = await client.plan_collateral_return()

        # Inspect the return and residual-position impact before signing.
        handle = await client.execute_collateral_return_plan(plan=plan)
        await handle.wait()

        if not plan.truncated:
            break
    ```
  </Tab>

  <Tab title="API">
    After the submitted transaction confirms, inspect the `truncated` value from
    the plan that produced it. When it is `true`, request a fresh plan, inspect and
    sign its wallet action, submit it, and wait for confirmation again. Continue
    until the confirmed plan has `truncated: false`.
  </Tab>
</Tabs>

## Understand Residual Positions

Collateral return only removes exposure that can be recombined into collateral.
Any unmatched exposure remains in new residual positions. These positions
preserve the wallet's remaining economic exposure after the returned pUSD is
removed.

For example, complementary exposure to one underlying leg may unlock pUSD while
the rest of each Combo remains open. Inspect the created positions before
execution and include them in your normal [Combo position
sync](/trading/combos/market-makers#list-combo-positions) after confirmation.
