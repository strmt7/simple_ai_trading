# Targeted dependency remediation

Reviewed against `78b0189a6f46779d606371fd2c0724be7a4872c1`.
This is three-advisory triage and remediation, not a complete security scan.

## Static triage

| Input | Static product-impact verdict | Review rank | Dependency fix |
| --- | --- | --- | --- |
| [13: CVE-2026-82397 / GHSA-mpf4-983q-p7j4](https://github.com/advisories/GHSA-mpf4-983q-p7j4), urlencoded parsing denial of service | needs_review, medium confidence | 1 | 6.5.8 |
| [12: GHSA-8423-8fgw-73vq](https://github.com/advisories/GHSA-8423-8fgw-73vq), multipart allocation amplification | needs_review, medium confidence | 2 | 6.5.8 |
| [11: GHSA-wwv5-g3v4-889x](https://github.com/advisories/GHSA-wwv5-g3v4-889x), legacy cookie attribute injection | needs_review, low confidence | 3 | 6.5.8 |

All three inputs are retained separately. GitHub's REST responses identify
Tornado 6.5.7 in `uv.lock` as affected. Static `uv tree --frozen --offline
--invert --package tornado` traces the optional `microstructure` extra through
`hftbacktest -> holoviews -> bokeh -> tornado`, with Panel also in that subtree.
`SECURITY.md` treats reviewed dependencies as a supply-chain boundary.

Remote form input could reach the first two vulnerable server parsers if a
Tornado server path is exposed. The third additionally requires a handler to
pass attacker-influenced values through capitalized legacy cookie keywords.
No direct Tornado/Bokeh/Panel server import or `panel.serve` call was found in
`src` or `tools`. The similarly named backtest-panel is not an HTTP server.
These are counterevidence, not proof that every optional upstream path is safe.
Actual product server exposure, supported deployment and full upstream caller
coverage remain proof gaps; repository exploitability is not confirmed.

## Narrow remediation and verification

Outcome: the affected **dependency boundary is fixed** by pinning Tornado
6.5.8 in the existing optional extra and regenerating `uv.lock`. This also
constrains non-uv installations of that extra, without adding a server package
to the base install. No unrelated dependency version changed. The release API
tag endpoint returned 404; patched-version evidence instead comes from all
three advisories, published package metadata and the installed patch source.

The fix-finding workflow was applied inline because repository policy prohibits
delegation. The separate compatibility/bypass review checked optional versus
base installation, direct versus transitive pinning, the bounded split before
allocation, field counting including separator-only input, and legacy keyword
case aliases. No application server or exploit against a remote host was run.

- Syntax/diff gate: Ruff passed; `uv lock --check` passed (140 packages).
- Original-version control: the same socket-free test file against 6.5.7
  produced nine security assertion failures and three legitimate-case passes.
- Patched-version gate: all twelve checks passed against 6.5.8. Excessive form
  fields are rejected; multipart splitting is bounded before list allocation;
  semicolon and CRLF cookie attributes are rejected for Domain/Path/SameSite.
- Compatibility controls: ordinary repeated form values, ordinary multipart,
  and valid named/legacy cookie arguments still pass.

Commands: `uv run --locked --with tornado==6.5.8 python -m pytest -q
tests/test_tornado_dependency_security.py` and the explicit 6.5.7 negative
control; neither enables the microstructure extra or opens network sockets.
The tests skip if the optional library is absent in a base-only test run.
Full optional-stack UI behavior, remote deployment exploitability and hosted
alert closure are separate from these verified library-level results.
