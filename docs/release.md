# Beta Release Workflow

Simple AI Trading publishes beta releases through the manually triggered
`beta-release` GitHub Actions workflow.

The workflow is intentionally Windows-hosted because the desktop app is a native
Win32 build. It performs these release gates before creating a GitHub
prerelease:

- resolve beta metadata from `pyproject.toml` or a manual SemVer input,
- reject non-beta versions,
- require an explicit replacement acknowledgement before deleting an existing
  release or tag,
- install and test the Python package,
- run the coverage gate,
- build `build/windows/SimpleAITrading.exe`,
- smoke CLI and launcher entrypoints,
- build Python wheel and source distributions,
- assemble a portable Windows beta ZIP,
- attach SHA-256 checksum files,
- publish a GitHub prerelease.

Default beta metadata:

- Python package version: `0.1.0b1`
- GitHub release tag: `v0.1.0-beta.1`

To publish the default beta, run the workflow manually and leave
`release_version` empty.

To replace an existing beta only when explicitly required, set:

- `replace_existing`: `true`
- `release_version`: `<existing-beta-semver>`
- `replacement_acknowledgement`: `replace <existing-beta-semver>`

Replacement is intentionally awkward so accidental release/tag deletion does not
happen during routine beta iteration.
