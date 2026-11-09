# Operator information and native overview

This is a bounded implementation checkpoint in the resumed whole-app revamp.
The goal remains active. No qualified financial edge, new economic observation,
training, GPU acceleration, account access or order is claimed. Counts remain
203 observations / 65 hypotheses / 37 scoped mechanisms / 0 qualified stable
edges. The broad repo review and final exhaustive bug hunt remain incomplete.

## Demonstrated defects and changes

The Python budget summary sorted zero remaining headroom through a truthiness
fallback as though it were missing. With an exhausted REQUEST_WEIGHT window and
a half-used ORDERS window, the compact report selected the latter. Explicit
missing-value handling now preserves zero as the tightest window. The existing
startup guard already blocked that exhausted sample; this repair corrects the
summary rather than claiming a newly discovered order-gate bypass.

The native footer used a short unknown-string blacklist to turn other values
green. Dry-run, exhausted and arbitrary future labels could therefore look
healthy. Presentation is now a separate typed component: unknown, observed,
caution, critical and simulation. Critical budgets/ledger failures are red;
simulation is amber and ordinary observations are informational blue. Unknown
compute values do not acquire a GPU suffix. Inconsistent clear-ledger counts
remain unknown. Cached budget text cannot establish network connectivity.
None of these presentation states grants order or profitability authority.

The overview now says "Performance evidence" and "No verified series loaded",
with a route to Research, instead of showing empty performance axes. Stops
remain independent and visible. A minimum-width native render exposed the
fixed-width API status control overlapping the next footer cell; both now use
the same cell geometry. Dropdowns share the native owner-draw path with explicit
selection, disabled text, focus indication and ellipsis. The preview explicitly
labels simulated ledger, compute and AI status rather than presenting stub GPU
state as actual residency.

## Visual artifacts and reproducibility

- [Generated design concept](overview-concept.png): built-in image generation,
  used to explore hierarchy and clear empty states; it is not a working UI or
  financial evidence. The exact prompt is retained in [design-prompt.md](design-prompt.md).
- [Inspected minimum-size native frame](overview-native-inspected-1280.png):
  1280x860 DIPs, exported as 1920x1290 pixels.
- [Inspected larger native frame](overview-native-inspected-1680.png):
  1680x1020 DIPs, exported as 2520x1530 pixels.

The frames use the actual production renderer/control handlers at the observed
150% scaling, with local dry-run status only. They show legible Paper,
Conservative and 5x selections, visible independent Stops, separated footer
cells and explicit unavailable/simulated evidence. Earlier PNGs are retained
as intermediate artifacts, including the initial frame with misleading
synthetic GPU wording and frames before dropdown print support. They are not
final acceptance evidence. The rendering trials are not market observations.

The new `tools.capture_native_windows_preview` creates an owned hidden native
process suspended, assigns its kill-on-close job before resuming, forces local
dry-run, bounds lifetime to 30 seconds and refuses existing output. Its temporary
BMP directory is removed on exit. Startup-containment failure and timeout clean
the original owned process handle; no worker-reported PID grants termination.
The native exporter waits for initialized controls and uses its own GDI bitmap,
production client renderer, native dropdown bounds and real selected item IDs.
No foreground activation, desktop input, recurring task or visible helper
terminal is involved. The native app rejects preview export without dry-run;
the live negative check exited 2 and created no output.

Four earlier cross-process print attempts produced all-black frames and were
rejected, with process cleanup; no image was admitted from them. Switching to
an in-process production renderer resolved that limitation. Hidden dropdown
printing additionally required the normal owner-draw notification for its
selected field. These failures are not concealed as successful screenshots.
Do not repeat those approaches or use a visible/foreground fallback.

## Verification and limits

The affected Python API-budget, command-contract/native-parity and preview
lifecycle suite passed 62 checks. Preview cases exercise success, blank/missing
output, nonzero exit, timeout, failed containment/spawn/job closure, output
collision, invalid geometry and foreign/missing paths. Release C++ build and
the native presentation test executable pass; its checks use runtime conditions
rather than assertions that disappear under NDEBUG. Ruff check/format pass.
The generated CLI/native command contract is unchanged.

The two final overview frames were visually inspected. The actual preview
processes exited, and the closing scope check found zero such processes and
zero project scheduled tasks. This does not claim that unrelated application
clients or all background processes on the PC were removed. No keyboard-only,
screen-reader, all-page, all-DPI or real-account verification is claimed.
The evidence manifest binds exact implementation and image bytes. Review of
`main.cpp` covers changed UI/preview sections, not all its historical lines;
passing this checkpoint is not whole-app semantic review or release readiness.

The final incremental source index was explicitly refreshed, then an actual
three-result native-source query located the new presentation component and
preview lifecycle. All three spans were confirmed in live source. External
snapshot `fe6fe8508d09754b7a4cd9b42b8508cb` and receipt
`search-c3a40ed010e142b79e17381dd00a2088.json` retain that routing evidence.
No workflow upgrade or repeated routing benchmark was performed. After the
query, the owned runtime had no live handshake; stale PID/socket metadata was
removed under its lock, preserving logs and unrelated application clients.
