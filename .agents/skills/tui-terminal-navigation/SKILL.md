---
name: tui-terminal-navigation
description: Validate and debug terminal UI keyboard navigation bugs, frozen menus, modal focus issues, arrow-key handling, Settings/Funds/Strategy navigation, or cross-platform Windows/Linux TUI behavior. Use whenever Codex changes or investigates Textual terminal menus or user reports that a menu cannot move, select, scroll, or close.
---

# TUI Terminal Navigation

Use real terminal-process probes before trusting in-process Textual tests.

## Required Workflow

1. Reproduce with the actual CLI process, not only `App.run_test`.
2. Drive menus with real arrow keys first: `Up`, `Down`, `Enter`, `Escape`.
3. Treat `j/k` and numeric keys as secondary shortcuts; they never replace arrow-key validation.
4. Verify Settings, Strategy feature selection, Funds, Confirm, and Form modals after any TUI event-loop or focus change.
5. Keep actions non-blocking while a modal is open. A user-launched action must not await a modal result inside Textual's key handler.

## Script

Run:

```powershell
.\.venv311\Scripts\python.exe tools\terminal_navigation_probe.py --cwd .
```

On Linux:

```bash
python3 tools/terminal_navigation_probe.py --cwd .
```

The probe spawns the actual `menu` command in a PTY, answers terminal capability probes, renders the ANSI screen with `pyte`, and asserts:

- root navigation reaches Settings with arrow keys
- `Enter` opens Settings
- `Down` highlights Strategy, then Execution
- `Up` returns to Strategy
- `Enter` opens Strategy feature selection

If this probe fails, fix the app before claiming menu navigation works.
