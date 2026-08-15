# Main-only branch policy

`main` is the sole permitted branch after final consolidation. The repository
must not discard unique commits or dirty worktrees to reach that state.

Run `pwsh -File tools/consolidate_main_only.ps1` first. The default is a dry
run: it fetches current refs, inventories local and remote branches and
worktrees, and fails if any branch contains commits not reachable from `main`.

After all unique history is deliberately integrated and `main` exactly matches
`origin/main`, run:

```powershell
pwsh -File tools/consolidate_main_only.ps1 -Apply -InstallRuleset
```

The apply phase removes only clean secondary worktrees and fully merged branch
refs. It then installs the GitHub `main-only-branch-creation` ruleset, targeting
all branches except `main` with the creation restriction and no bypass actors.
That prevents future branch creation at the repository boundary. Changing or
removing the ruleset is an explicit repository-administration action.
