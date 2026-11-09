# Spot stream protocol evidence

Two immutable files from Binance's official documentation repository passed
their predeclared native Crawl4AI admission: HTTP 200, native success, matching
final URL, empty error and substantive extracted text. The June 9 file revision
was selected by unauthenticated commit metadata with an October 1 cutoff; the
October 15 revision was the latest returned file commit. Selection preceded
document-body access. Commit dates do not independently prove historical website
deployment or the collector's effective configuration.

The [intent](intent.json) and [result](result.json) bind exact URLs, revisions,
provider/controller identities, native result and section digests, and the
durable journal. Both bounded owned processes completed normally. Full native
responses, source discovery, logs and journal remain in the private relative
directory named in the result. No persistent worker was installed.

## What this resolves

Both reviewed revisions document default millisecond stream timestamps and an
explicit URL option for microseconds. Spot depth's `E` is event time; `U` and `u`
identify update ranges. The documented Spot bookTicker payload has six fields
and no `E` or `T` timestamp. Neither digit length nor an update ID can replace
evidence of the collector's actual timestamp mode. Local receipt is a different
clock from exchange event time.

The source requires buffering updates around the snapshot and replacing level
quantities rather than adding them. Zero removes a level. A bounded snapshot
leaves unchanged levels beyond its coverage unknown. The initial buffered-event
instruction includes the snapshot ID in the retained update range; the subsequent
update rule rejects a first ID above the local ID plus one. These statements
differ at the boundary where the first ID equals the snapshot ID plus one.
Preserve that wording distinction; this review does not qualify that boundary
case by silently rewriting the source or borrowing futures `pu` semantics.

No semantic change was observed in these limited reviewed rules across the two
versions. The documents differ elsewhere; this is not full-version equivalence.

## Consequence for development

Live-source inspection confirms `microstructure_data._fetch_depth_snapshot`
uses `/fapi/v1/depth`, while `microstructure_normalization.prepare_timed_input`
requires native `T` and futures `pu` linkage. These are futures-specific paths.
Spot documentation cannot qualify their clocks or make them a paired Spot
adapter. No runtime change or new fit was justified by this document check.

The next admission task is evidence of the historical vendor collector's exact
stream configuration and clock semantics, followed by a causal Spot adapter
with independently bound snapshot availability. Continuous paired coverage,
costs, fills, financing and margin remain open. These two source receipts add
no economic observation: counts remain **203 / 65 / 37 / 0 qualified stable
edges**. No market/account request, credential, order, model outcome or protected
partial payload was accessed.

The user's explicit resumed-session authority permitted this new documentation
budget without another approval request. Earlier failed website crawls and their
consumed financial-study consequences remain unchanged. The new receipts do not
turn those failures into successful captures or reopen a market population.

## Focused verification

All six affected policy/handoff checks completed with exit zero using a new
private task-owned pytest base. The first invocation reached six passing checks
but exited with a shared temporary-directory cleanup permission error; retain
that failure separately and do not count it as a clean run. No shared files or
host permissions were changed. The root instructions now require a new owned
base for each invocation, avoiding another session's pytest cleanup paths.
Receipt hashes, admission fields and new relative links were checked directly;
no trading, model, GPU or full-suite rerun was needed for this source-only change.
