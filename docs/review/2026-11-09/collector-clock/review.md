# Collector units and clock-probe repair

The vendor documents a different Spot timestamp mode from Binance's default:
its collection URL explicitly selects microseconds since February 28, 2025.
That retrospective claim covers the fixed October 1 sample's date. The actual
collector deployment is not independently attested by a current document, but
the declared vendor mode now has direct primary-source support. Do not default
this historical adapter to milliseconds or choose a unit by timestamp magnitude.

Three preselected documents passed native Crawl4AI admission, with matching final
URLs, HTTP 200, native success, empty errors and substantive text. The
[intent](intent.json) and [result](result.json) retain selection, exact revisions,
provider/controller identities, result hashes and the durable journal binding.
Full native results and logs remain private. The source controller is terminal;
no market download, account request, credential, order or fit occurred.

## What the source establishes

- [Spot collection details](https://docs.tardis.dev/historical-data-details/binance)
  identify the microsecond URL, multiple connections, generated REST snapshots
  with top-1000-level coverage and native update-ID integrity checks.
- [Collection overview](https://docs.tardis.dev/historical-data-details) claims
  synchronized arrival clocks with 100 ns precision. Precision is not measured
  accuracy, and it supplies no numeric absolute-offset guarantee. Daily restart
  and snapshot collection can leave gaps; the absence of a marker in one minute
  does not establish complete coverage.
- The [immutable pre-sample mapper](https://github.com/tardis-dev/tardis-node/blob/2f13277771f75ef43571025d6cf5d93cee4b9584/src/mappers/binance.ts)
  passes native fields to a microsecond-named helper. This file alone does not
  verify that helper or deployed collector. Its historical mode can suppress
  initial snapshot-overlap errors. Its first Spot bridge checks snapshot ID plus
  one, whereas the reviewed exchange text's initial-buffer instruction names the
  snapshot ID. Preserve that distinction; importing the mapper wholesale would
  not qualify continuity or resolve the exchange wording.

The next useful offline question is the declared microsecond conversion against
retained native Spot depth and collector receipt clocks, under a separate bounded
clock-only intent. Keep unknown clock offsets explicit. No raw market payload was
reread here, no book was reconstructed and no economic population was reopened.
Original source restrictions and numerical results remain unchanged.

## Production repair

The futures clock probe previously coerced `serverTime` with `int()`, accepting
boolean and fractional identities. It also subtracted two epoch-sized floating
point values, losing a synthetic 2 ns offset, and clamped a regressing monotonic
interval to zero. These are independent local defects; the vendor Spot source
does not qualify the futures endpoint's physical clock.

`probe_binance_futures_clock` now requires an exact positive integer server time
inside the existing int64-nanosecond range. It validates policy inputs before
opening a session, rejects clock regression, checks wall elapsed against monotonic
elapsed under an explicit configurable 1 ms default limit, and subtracts integer
epochs before scaling the difference. It records the final wall receipt without
another epoch float conversion. External sessions remain caller-owned; owned
sessions close on success and failure. Other production definitions are AST
identical to the parent revision.

The two regression controls fail against the original implementation for the
claimed reasons: lost offset and missing rejection of boolean server time.
All **86 current clock/capture/normalization checks pass**, with no skip, and Ruff
lint/format checks pass. An initial synthetic fixture provided three clock probes
while requesting the default five; that fixture error was corrected and excluded
from defect proof. This was not a timing benchmark or full repository suite.

This remains a point-offset estimate. Millisecond server quantization, network
asymmetry, timestamp generation timing, between-probe drift and absolute accuracy
remain unqualified. The disagreement limit is policy, not a metrology claim.
No model, execution-latency, margin or stable-edge promotion follows. Financial
counts remain **203 observations / 65 hypotheses / 37 scoped mechanisms /
0 qualified stable edges**.
