# Source-bound repository navigation

Native semantic retrieval returned none of the ten existing benchmark's named
expected files in its first five results. An incremental source refresh alone
did not repair that weakness. The installed embedding model and source binding
were valid; generated AI-risk JSON frequently dominated broad results. This is
weak navigation, not proof that the requested implementation is absent.

The CLI and MCP now combine the actual native semantic response with lexical
matches from exactly the same approved CocoIndex chunks. File-level reciprocal
rank fusion uses equal route weights and constant 60; lexical BM25 weights
filename/body 4/1. Each route and the final result retain at most the requested
limit. Duplicate chunks cannot cast multiple votes for one file. There are no
benchmark-specific query mappings, excluded result families, replacement models
or external embedding requests. Ranks express ordering, never confidence.

An exploratory, frozen pilot on the already-observed ten cases improved named
expected-file hits from **0/10 to 7/10**. The integrated implementation was then
frozen before a single paired evaluation: each query's one native response feeds
both rankings, with no retries, query changes or model/weight adjustments.

| Query set | Cases | Native hits in first five | Combined hits in first five |
| --- | ---: | ---: | ---: |
| Already-observed development cases | 10 | 0 | 7 |
| Prospectively fixed developer follow-up questions | 5 | 3 | 4 |

Four integrated cases still missed their frozen expected files: outage recovery,
action-value policy gates, terminal holdout integrity and timed order-book seed.
Relevant implementations may also fall outside a frozen expected list; a miss
must remain recorded regardless. These are developer checks, not an
independent statistical sample, universal recall estimate or financial evidence.
No speed claim is made: concurrent host load and timings were not qualified.

The lexical cache is built only by explicit indexing. Search cannot install,
refresh or build it. Native semantic failure stops the request before lexical
retrieval; there is no lexical-only failure fallback. Native-only reproduction
remains available through CLI `--semantic-only` and MCP `semantic_only: true`.
Raw native receipts link to final receipts. The whole live source fingerprint
and selected candidate bytes must match their admitted mirror before return.

The cache is private and bound to repository owner, mirror/source digest,
implementation digest and native database/WAL state. Explicit construction checks
every source path against the approved mirror before materialization, takes a
read-only native snapshot, caps rows/text/cache bytes, requires disk headroom,
and atomically replaces only its own recognized cache. Queries retain the same
language/path filters. Processes use the admitted isolated runtime, bounded
timeouts and owned hidden child containment; no registration, persistent worker,
startup task or global environment was installed or changed.

**167 focused checks pass** across source scope, existing adapter behavior and
the new fusion/cache/failure contracts. Ruff passes for the changed Python files.
Another 22 documentation contracts pass with an owned temporary directory. The
first invocation's shared host temporary-directory cleanup failed permission
checks after its assertions passed; it is retained, and no host ACL was changed.
The real isolated runtime passed a transport/receipt preflight, four supported
MCP protocol probes, stdio initialization/listing and a default combined search.
The 15-query run completed once, preserving its intent, journal, raw outputs and
linked receipt digests. Winning capital-accounting candidates were checked in
live source. Source and result bindings are in [verification](verification.json).

The preceding [handoff compaction](../handoff-compaction/review.md) is a distinct
source phase: its old adapter hashes and 0/10 result remain historical evidence,
preserved in its own connector-created parent commit. Its reduction measures
handoff bytes, not tokens or runtime. The shared child cleanup repair does not
qualify abrupt parent death before job enrollment or full process supervision.

Financial observations remain **203**, hypotheses **65**, scoped mechanisms **37**
and qualified stable edges **0**. No market capture, protected partial-data read,
fit, trading-performance benchmark, account call or order occurred in this
navigation repair. It does not complete the whole-repository semantic review,
security acceptance or trading R&D goal. Resume informative financial/capital
work rather than repeatedly tuning these observed navigation cases.
