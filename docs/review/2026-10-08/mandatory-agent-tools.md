# Mandatory agent-tool audit

The user explicitly required Crawl4AI research, then cross-repository mandatory
CocoIndex Code and Caveman. Seven available local repositories were checked;
archived trading capture worktrees and immutable mirrored skills were not edited.
The [machine-readable audit](mandatory-agent-tools.json) binds the exact reviewed
instruction bytes and separates availability, checks and unproved compliance.
The root instruction record preserves its host-byte hash and a separate Git
blob hash: CRLF-to-LF staging changed representation and the first byte-identity
check rejected that mismatch. Publication uses the Git binding, not an assumed
equality with the host record; the retained pre-access intent matches exactly.

| Repository | Before this task | Policy change |
| --- | --- | --- |
| SuperZip | Both mandatory; local skills and CocoIndex wrapper present | No change |
| VulnerabilityScreener | CocoIndex route present; no Caveman skill | Explicit both-mandatory development-agent block |
| simple_ai_trading | CocoIndex mandatory; no Caveman skill | Both-mandatory block and local Caveman overlay |
| ai skills | Neither explicitly mandatory in root rules | Additive both-mandatory block; mirrors untouched |
| qfales | Neither explicitly mandatory in root rules | Additive both-mandatory block |
| omero-docker-extended | No root AGENTS file in this local checkout | New concise root agent rules |
| Raspberry | No root AGENTS file | New concise root agent rules |

The previously empty user-level Codex `AGENTS.md` now requires both skills.
Discoverable shared `caveman` and `cocoindex-code` skills with implicit invocation
enabled were added to the configured user skills directory. This is local Codex
configuration, not a repository publication or a guarantee that an unrelated
agent product reads Codex configuration. Root rules apply to agents using each
repository's AGENTS contract; higher-priority instructions remain controlling.

Caveman is internal lite compression only. Public prose, code, exact quantities,
units, uncertainty, negation, safety and progress updates remain intact. CocoIndex
is mandatory for broad/fuzzy code navigation, not meaningless queries for exact
known instruction files. Actual search, current-root/index binding and live
source confirmation matter. Existing local pinned adapters are preserved; missing
tooling must be diagnosed and disclosed, never silently claimed as used.

This task used exact known paths. No new semantic search, embedding download,
cold index, model/GPU workload, MCP overwrite or application-runtime dependency
was needed or claimed. SuperZip and scanner have local adapters; trading's adapter
also supports explicit `AGENT_COCOINDEX_REPO` binding for a reviewed shared use.
The other checked repos had no local wrapper. Their policy is mandatory, but a
qualified search on each of those repositories has not been executed here.

Shared skills and trading Caveman passed the skill validator. Trading's policy
checks guard required root routes and evidence-preserving boundaries; they do
not measure skill efficacy or prove every future agent's compliance. No agent
instruction file can technically force arbitrary software to obey it. Existing
SuperZip startup/actual-use receipt mechanisms remain unchanged.

Six focused trading policy checks passed; Ruff check and format passed. QFALES
agent-surface and runtime-contract checks also passed. The cross-repository
presence check passed all seven roots, and removing only our ai-skills block
reconstructed its exact prior instruction text after line-ending normalization.
No whole application suite or skill-efficacy benchmark is claimed.

Unrelated changes were preserved. The already-dirty ai-skills AGENTS file was
backed up before adding the new block; no original text was removed. Other repo
edits are local and unstaged, not silently bundled into a trading commit or
published alongside someone else's work. Historical captures and financial
results remain unchanged. Financial counts remain 202/65/37/0; goal remains active.
