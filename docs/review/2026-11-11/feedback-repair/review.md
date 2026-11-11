# Historical evidence and mainline feedback repair

This `BASE-01` phase repairs the terminology blocker, 13 import annotations and
nine unused-parameter diagnostics diagnosed at the November 11 assessment.
It changes no financial result, research population, model, account or trading
authority. The source baseline is `982bd9ac3056137e7361114f9e262ac16d6bc42b`.
The [evidence](evidence.json) binds the source phases and local outcomes.

## Preserve evidence without hiding changed claims

The terminology auditor previously exempted 12 entire paths. That preserved
their historical content but also exempted future edits. Four other retained
before-state/audit/review records were incorrectly treated as current editorial
copy, blocking the hosted Linux job before pytest began.

The auditor now binds all 16 historical texts to SHA-256 digests of UTF-8 text,
normalizing CRLF to LF for portable Git checkouts. Only an exact path and matching
content receive historical treatment. Changed content is audited normally;
copied content at another authored path receives no exemption. Filename checks
remain unconditional. Raw evidence remains unread. Lone-CR changes do not receive
the checkout normalization. No historical file or numerical result was edited.
These content bindings establish an editorial scope, not economic validation or
permission to modify frozen source.

The 13 import diagnostics concern the storage-migration import following existing
direct-script path bootstraps. Each now carries the same explicit `E402` annotation
as neighboring imports. All 13 before/after Python ASTs are identical. No import
was moved above its required path setup, no research runner was executed and no
analyzer-wide rule was disabled. The original source remains reachable at the
baseline commit and is separately retained in the private source-phase record.

Four redirect handlers retain urllib's exact callback signature and explicitly
discard two unused inputs; all redirects still return `None`. The HTML parser
explicitly discards unused attributes while preserving script/style exclusion.
No keyword parameter was renamed. Tests exercise the complete urllib opener
chain with an in-memory transport and prove that every supported redirect code
ends in the original HTTP failure without requesting its destination.

## Verification and corrections

The final affected suite passes **48 cases**. Full-repository Ruff and terminology
checks pass. The tracked production Vulture scope passes with the existing
hash-locked CI analyzer, Vulture 2.16, installed in a new isolated task-owned
environment. The application environment was not changed.

The first test phase retained 20 failures caused by an incorrect test expectation:
calling the redirect handler directly returns `None`; urllib's complete handler
chain raises `HTTPError`. The corrected test exercises that chain without
network access. A first Vulture attempt found its package absent; it was recorded
as an environment failure, never as a passing analysis. Both original failures
and their source/log bindings remain retained. Final source-bound checks followed
the corrections. The unchanged callback suite was not expanded into market tests.

## Remaining work and priority

`BASE-01` remains partially complete. The retained-source Markdown diagnostics
and low workflow advisory need their own provenance/current-contract treatment.
No Super-Linter rerun, hosted workflow dispatch, broad pytest/coverage run or
DeepSource retry was performed. The current full-suite failure count remains
unknown; this local repair does not establish hosted acceptance or security.

Proceed to `EXEC-01` for the pending-close inspection/collection/application
workflow while keeping the remaining feedback issues explicit. Eligible
informative financial R&D and demonstrated capital failures retain priority.
The holding-yield row still requires a material official program, payout or
economic-comparator change; no new qualifying change was established here.
No market request, protected capture, credential, account request, order,
benchmark, training run, scheduler or service was used in this phase.
