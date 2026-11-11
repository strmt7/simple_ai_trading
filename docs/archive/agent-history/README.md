# Preserved agent handoffs

Current operating guidance is in [Agent Start](../../AGENT_START.md) and
[Continuation](../../CONTINUATION.md). Keep them compact; detailed reviews belong
beside their source-bound evidence.

The [November 9 manifest](handoff-2026-11-09.json) preserves 17,028 original lines
through their immutable published Git objects. It records each original path,
revision, blob identity, SHA-256, byte count and line count. This avoids copying
old content or publishing historical workstation paths again. The objects remain
reachable through the recorded source revision in the repository's ancestry.
Preserve them during any separately authorized future history operation.

The current reader uses the [November 11 manifest](handoff-2026-11-11.json).
It carries both original entries unchanged and adds `continuation-2026-11-11`,
the complete 534-line continuation published at `27e97291`. Its parent-manifest
hashes preserve the earlier binding. The November 9 manifest retains its original
bytes. Together these entries preserve 17,562 source lines.

| Original | Immutable published source |
| --- | --- |
| Agent Start | [5,572 lines at the source/capital checkpoint](https://github.com/strmt7/simple_ai_trading/blob/eaad2d1605f6256ba43cd78ab6b96479294b6155/docs/AGENT_START.md) |
| Continuation | [11,456 lines at the source/capital checkpoint](https://github.com/strmt7/simple_ai_trading/blob/eaad2d1605f6256ba43cd78ab6b96479294b6155/docs/CONTINUATION.md) |
| November 11 continuation | [534 lines at the dispatch checkpoint](https://github.com/strmt7/simple_ai_trading/blob/27e97291d2a7385a3bbf7583cb30644a59565d0d/docs/CONTINUATION.md) |

Relative paths inside an original resolve from `docs/`, not this directory.
Immutable permalinks retain the original rendered relative links and anchors.
The earlier August handoff snapshots in this directory remain unchanged.

Historical progress is provenance. Every non-superseded study-specific access,
retry, account, cost and promotion constraint remains in force. Archiving grants
no authority and does not turn an old result into current evidence. Current user
instructions, AGENTS, exact canonical contracts and the current registry govern
conflicts. Look up the affected study before access; absence from a compact
summary is never permission. Do not recover or use credentials from history.

Use the bounded local reader for exact lookup:

```powershell
uv run --no-sync python -m tools.read_handoff_archive continuation --find "settlement-equivalence"
uv run --no-sync python -m tools.read_handoff_archive continuation --line 1 --count 40
uv run --no-sync python -m tools.read_handoff_archive continuation-2026-11-11 --find "source-dispatch"
```

The reader verifies the manifest, revision/path, blob size, source SHA-256 and
line count before returning text. Searches are exact literals, reporting at most
20 line locations; windows are at most 120 lines and 16,384 output bytes. It
performs no fetch, index rebuild, credential access or background task. Exact
lookup is not a substitute for mandatory semantic routing before broad searches.
Git plumbing runs with `--no-lazy-fetch` and `--no-replace-objects`: partial-clone
settings cannot silently fetch missing objects or substitute replacement content.
An older Git without that option fails rather than falling back to network access.

An object may be absent from a shallow clone. Explicitly fetch the manifest's
recorded source revision, or read that exact revision through the official
connector. The reader fails with an actionable message; it does not silently
fetch, change history, install tools or substitute current text.

See the [compaction verification](../../review/2026-11-09/handoff-compaction/review.md)
for byte measurements, source binding and the limits of this preservation audit.
The [November 11 routing review](../../review/2026-11-11/handoff-routing/review.md)
records the subsequent compactness repair and the added immutable source entry.
