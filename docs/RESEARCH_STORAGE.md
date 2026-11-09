# Historical research storage

Preserved research inputs live under the application-relative
`data/historical-research` directory by default. Set
`SIMPLE_AI_TRADING_HISTORICAL_DATA_ROOT` to an explicit absolute directory when
storage belongs elsewhere. This setting changes location, not research access,
trading authority or dataset qualification. Historical publishers use the same
resolver; they no longer embed a workstation drive mapping.

The resolver rejects relative root overrides, rooted suffixes, parent traversal,
alternate data streams and suffixes that resolve outside the configured root.
It creates no directories and performs no content analysis. Keep immutable
historical inputs separate from current writable databases and operational state.
Do not rerun an old publisher merely to check whether its path resolves.

## Selective relocation

`tools.migrate_research_storage` freezes a metadata inventory and explicit
retention decisions before copying opaque bytes. Its `plan` command requires
separate source, destination and receipt directories. The destination and receipt
directories must be fresh. It rejects overlapping roots, directory reparse points
and unreviewed file links. Review the retention plan before `apply`; model
download caches, obsolete process markers and empty diagnostic logs are distinct
from unique source evidence, results and ownership claims.

The Windows copy worker is suspended until enrolled in a kill-on-close owned
job. It installs no task, service, startup hook or automatic restart. Only that
worker enters background scheduling mode. `--io-mib-per-second` caps application
bytes across source reads, target writes and verification reads; the default is
64 MiB/s with one-MiB chunks and no accumulated idle credit. Filesystem metadata,
writeback and other applications are outside this byte counter, so it is not a
promise about total physical-device activity.

The worker denies source writers, journals completed source hashes durably,
checks any interrupted partial prefix before appending and verifies every
retained target with a full SSD SHA-256 readback. It never overwrites current
working databases or deletes the source. Identical content with identical
modification times may share an immutable hardlink inside the new destination;
all historical names remain. No link is made to current mutable databases.
Copied-file progress does not mean readback verification is complete.

Source deletion requires the terminal `copy-verified.json` receipt and a separate
current-state check with `tools.verify_research_storage_migration`. That check
binds the plan, applied copy implementation, exact retained/discarded population,
source identity and metadata, destination ownership, file identities, sizes,
modification times and read-only attributes. It reuses completed bulk hashes
rather than reading the entire dataset again. It does not delete anything or
turn metadata checks into a fresh content-hash claim. Perform explicitly
authorized removal only against the verified literal source directory, then
verify its absence and termination of the owned worker. Never leave a transfer
running after its authorized development session ends.

## Publication privacy

Physical path mappings, inventories, copy journals and original implementation
snapshots belong in ignored `data/storage-migrations`. Relocated input bytes
remain ignored as well. Publish relative references and sanitized conclusions;
never force-add local receipts, archives, screenshots or raw logs.

Run `uv run python tools/check_portable_paths.py --staged` before committing.
CI checks additions against its explicit base revision. The gate withholds
matching values and rejects private migration content even for binary additions.
It detects candidate workstation paths, not every possible privacy leak. Review
generated artifacts and changed filenames separately. Older hash-bound results
and shared history remain historical evidence; this gate does not certify that
all existing history has been sanitized.

Local model-provider sessions also use explicit configuration and owned,
time-bounded children. The ROCm helper requires an installed library root and a
dedicated literal-loopback endpoint. It refuses an occupied endpoint, leaves
existing providers and device selections alone, and installs no persistence.
It reuses the installed project environment without synchronizing dependencies
or replacing accelerator packages. Requesting ROCm is not evidence of GPU
residency or a speed improvement.
