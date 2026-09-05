# Commercial distribution evidence inventory

Inspected September 5, 2026 at source revision
`c3db89ff89aa3430161510af36aeed0c3a8f6f16`. This is a source-derived design
review, not a completed vulnerability scan. No packaged commercial binary was
built or tested. Hashes are SHA-256 of UTF-8 text normalized from CRLF to LF.
The collection digest binds this inventory using the same representation.

| Evidence | Repository source | SHA-256 | Observation |
| --- | --- | --- | --- |
| native-launcher | `native/windows/src/main.cpp` | `5118b5d8afc62a69ebc1cf7ff54e18705f17ec4907a74b700b0d8f922be5461f` | `looks_like_repo`, `python_invocation`, `shell_command_for_cli` discover Python source and invoke the common module; examined lines 2561-2645. |
| native-build | `tools/build_native_windows.ps1` | `21d8216628e38a0e80d590a3b933bac142e8c4295306189c0e36ecd107ad80b8` | Native CMake build verifies the Python entrypoint; it is not a protected backend compiler. |
| python-package | `pyproject.toml` | `457fea135cc6dd8aba9fe9a5564027ad2e7be8a1fb4de13b33b6f51a32cfad7d` | Setuptools package with common Python entrypoint and optional native/GPU dependencies. |
| current-license | `LICENSE` | `1f8f09225488dfffc1541daa5ec7bccc4430add3d995bf92afb7ba9a3b41ecc5` | Repository carries the MIT license; no licensing change is authorized here. |
| beta-distribution | `.github/actions/windows-beta-release/action.yml` | `79816a90868c434d560241a34bdbba9ec59dc01c655b0f163de62a2f9e02aeb9` | Builds wheel and sdist, copies docs and Python distributions into the portable ZIP, and attaches Python artifacts to releases. |

Read-only GitHub metadata reported PUBLIC visibility. No visibility, release,
license, marketplace installation or account setting was changed. The user asks
for resistance to easy source copying in a possible future private/paid edition;
they have not selected an online-only product or commercial obfuscator purchase.

Primary references opened for this review:

- PyInstaller operating-mode documentation, including its bytecode/decompilation
  warning: https://pyinstaller.org/en/stable/operating-mode.html
- Nuitka Commercial documentation, describing compilation and separate
  constant/data protection: https://nuitka.net/doc/commercial.html
- GitHub repository visibility documentation, describing detached public forks:
  https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility

Vendor protection claims are capability descriptions, not verified resistance
measurements for this application. No measured latency, reverse-engineering
effort, target artifact budget or commercial product requirement exists yet.
