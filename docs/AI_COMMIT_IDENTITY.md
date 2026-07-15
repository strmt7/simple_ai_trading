# AI Commit Identity

This is the detailed audit policy referenced by `AGENTS.md`. It is read only
when an agent creates or audits Git history.

Every commit, amend, merge, cherry-pick, squash, rebase, or history rewrite
produced by an AI agent must have this literal identity:

```text
Author:     AI agent <>
Commit:     AI agent <>
```

Use command-scoped configuration so no value leaks into repository or global
Git configuration:

```bash
git -c user.name='AI agent' -c user.email= commit ...
```

Any AI co-author trailer must be `Co-authored-by: AI agent` with no email.
AI commits must never use a human identity, the account owner's identity, a
GitHub noreply address, a host or local-placeholder email, a previous commit's
identity, a CI runner identity, a global Git identity, a model/vendor/tool
identity, or the hyphenated name `AI-agent`.

Human contributors continue to use their real GitHub identities. Audits must
check authors, committers, co-author trailers, and fresh anonymous-contributor
results from `GET /repos/{owner}/{repo}/contributors?anon=1`; report PR-head
refs separately. An invalid AI identity or a fake identity attributed to a
human is a policy violation. Do not publish another commit until the violation
has been surfaced. Before a first push, rewrite the affected local history with
`git filter-repo`; if the history is already shared, do not force-rewrite it
without explicit approval and a coordinated correction plan.
