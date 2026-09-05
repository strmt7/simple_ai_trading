# Plugin capability routing

The September 5 read-only CLI inventory reports 3,415 entries: 37 installed
and enabled, and 3,378 uninstalled. This is wider than the recommended-plugin
list supplied to the conversation. The [coverage ledger](plugin-capability-audit.json)
records a relevance review of every entry in the 180-plugin local curated
catalog; the separate 29-entry API catalog adds no new names. Installed remote
manifests and all 510 exposed tool definitions were also inventoried. Tool
definitions can belong to multiple plugins, so attribution counts do not sum.

Coverage is deliberately explicit: all remote entries were enumerated, but
their 3,402 complete remote descriptions were **not** reviewed. The CLI list
exposes identity/status, not those descriptions; the advertised search/suggest
plugin tools are not callable in this session. Local curated versions also
differ from current remote installations. Do not call this an exhaustive
security review or a completed semantic audit of the entire public directory.

## Use capabilities for specific gaps

| Capability | Current routing | Boundary |
| --- | --- | --- |
| Binance | Installed/enabled, independently confirmed; 89 exposed market-data tools spanning Spot, USD-M, COIN-M and Options | Manifest says public read-only, no account or transactions. Preserve exact request budgets/raw evidence; no plugin bypass of consumed studies. Prefer existing raw-capture adapters when connector responses cannot establish timing or provenance. |
| GitHub | Existing connector and CLI for repository/PR/dependency work | Use one route per task, no duplicate polling or redundant PRs. |
| Codex Security | Existing skills for source-backed hardening and the eventual exhaustive security phase | A loaded skill is not a completed scan. |
| Build Web Apps, visualization, browser | Existing skills and browser tools for actual UI flows and meaningful diagnostics | Do not launch unrelated apps, publish dashboards, or replace native interfaces merely to exercise tools. |
| NVIDIA | Official-provider catalog candidate; metadata confirms available but not installed, no declared app dependencies | Useful for actual CUDA/backend/profiling gaps. No install, driver mutation or hardware-speed claim follows from discovery. |
| Hugging Face | Official-provider catalog candidate; metadata confirms not installed, optional connector dependency | Useful for model/dataset provenance, license and revision checks; not automatic dataset download, model execution, training uplift or paid compute authority. |
| Deep Research / Undermind | Deep Research skill available; Undermind manifest cached but no callable tool discovered | Use for a genuinely distinct literature question, not repeated broad searches. Primary papers remain necessary evidence. |
| Temporal / Sentry / Datadog | Conditional architecture/observability candidates | No backend migration, subscription or telemetry upload without a demonstrated gap and appropriate authority. A workflow engine is not an exactly-once venue order guarantee. |
| Financial data and literature providers | LSEG, FactSet, S&P, Daloopa, Quartr, Scite and similar catalog leads | Verify entitlement, exact coverage, timestamps, source licensing and cost before selecting one; brand reputation does not prove a usable edge. |
| IBKR | Installed metadata exists, but no callable IBKR tool found | Outside current Binance/Polymarket account scope; do not read portfolios or draft trades. |

Existing local computation remains preferable for reproducible ledgers and
GPU experiments. Spreadsheet/document tools help when a human-facing artifact
needs them, not as a replacement for canonical numeric evidence. Email,
calendar, CRM, healthcare, media, travel and unrelated broker capabilities have
no current task advantage; leave their data and permissions untouched.

The Plugin Management skill drove the built-in/connected/new capability order
and read-only dependency checks. The OpenAI Docs skill led to the
[official plugin documentation](https://learn.chatgpt.com/docs/plugins), which
distinguishes discovery, installation and external-service connection. There
were no installs, purchases or permission changes. The supported CLI fallback
was found through its own help rather than undocumented app commands.

Efficiency correction: initial broad metadata output and an unprojected CLI
listing exceeded output limits. The corrected calls project counts and selected
fields before printing. Future audits should reuse this snapshot, refresh only
when availability changes or a specific gap appears, and never dump the catalog
or call every tool as an activity metric. A PowerShell inventory syntax error
was corrected before any data or account operation; it is not market evidence.
