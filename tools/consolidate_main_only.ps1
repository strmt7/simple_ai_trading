[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$InstallRuleset,
    [string]$Remote = "origin"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $output = & git @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed: $($output -join [Environment]::NewLine)"
    }
    return @($output)
}

function Get-GitLine {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $lines = @(Invoke-Git -Arguments $Arguments)
    if ($lines.Count -ne 1) {
        throw "Expected one line from git $($Arguments -join ' ')"
    }
    return ([string]$lines[0]).Trim()
}

function Get-Worktrees {
    $records = @()
    $current = $null
    foreach ($line in Invoke-Git -Arguments @("worktree", "list", "--porcelain")) {
        if ($line -like "worktree *") {
            if ($null -ne $current) {
                $records += [pscustomobject]$current
            }
            $current = @{ Path = $line.Substring(9); Branch = $null }
        }
        elseif ($null -ne $current -and $line -like "branch refs/heads/*") {
            $current.Branch = $line.Substring(18)
        }
    }
    if ($null -ne $current) {
        $records += [pscustomobject]$current
    }
    return @($records)
}

$root = Get-GitLine -Arguments @("rev-parse", "--show-toplevel")
Set-Location -LiteralPath $root
Invoke-Git -Arguments @("fetch", $Remote, "--prune") | Out-Null

$remoteHead = Get-GitLine -Arguments @(
    "symbolic-ref",
    "--short",
    "refs/remotes/$Remote/HEAD"
)
if ($remoteHead -ne "$Remote/main") {
    throw "The default branch must be $Remote/main; found $remoteHead"
}

$currentBranch = Get-GitLine -Arguments @("branch", "--show-current")
$dirty = @(
    Invoke-Git -Arguments @("status", "--porcelain=v1", "--untracked-files=all")
)
if ($Apply -and $dirty.Count -gt 0) {
    throw "Apply requires a clean primary worktree"
}
if ($Apply -and $currentBranch -ne "main") {
    throw "Apply must run from main; current branch is $currentBranch"
}

$localBranches = @(
    Invoke-Git -Arguments @(
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads"
    ) |
        Where-Object { $_ -ne "main" }
)
$remoteBranches = @(
    Invoke-Git -Arguments @(
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/remotes/$Remote"
    ) |
        Where-Object { $_ -notin @($Remote, "$Remote/HEAD", "$Remote/main") }
)
$branchRefs = @(
    $localBranches
    $remoteBranches
) | Sort-Object -Unique

$unmerged = @()
foreach ($ref in $branchRefs) {
    & git merge-base --is-ancestor $ref main
    if ($LASTEXITCODE -eq 1) {
        $count = Get-GitLine -Arguments @("rev-list", "--count", "main..$ref")
        $unmerged += "$ref ($count unique commits)"
    }
    elseif ($LASTEXITCODE -ne 0) {
        throw "Unable to compare $ref with main"
    }
}
if ($unmerged.Count -gt 0) {
    throw "Unique history must be consolidated before cleanup: $($unmerged -join ', ')"
}

$secondaryWorktrees = @(
    Get-Worktrees | Where-Object { $_.Branch -and $_.Branch -ne "main" }
)
foreach ($worktree in $secondaryWorktrees) {
    $worktreeDirty = @(& git -C $worktree.Path status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0 -or $worktreeDirty.Count -gt 0) {
        throw "Secondary worktree is unavailable or dirty: $($worktree.Path)"
    }
}

$mainSha = Get-GitLine -Arguments @("rev-parse", "main")
$remoteMainSha = Get-GitLine -Arguments @("rev-parse", "$Remote/main")
if ($Apply -and $mainSha -ne $remoteMainSha) {
    throw "Apply requires main and $Remote/main to be identical"
}

[pscustomobject]@{
    apply = [bool]$Apply
    current_branch = $currentBranch
    local_main = $mainSha
    remote_main = $remoteMainSha
    local_branches_to_delete = @($localBranches)
    remote_branches_to_delete = @($remoteBranches)
    secondary_worktrees_to_remove = @(
        $secondaryWorktrees | ForEach-Object { $_.Path }
    )
    ruleset_requested = [bool]$InstallRuleset
} | ConvertTo-Json -Depth 4

if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply after all unique history is on main."
    exit 0
}

foreach ($worktree in $secondaryWorktrees) {
    Invoke-Git -Arguments @("worktree", "remove", "--", $worktree.Path) | Out-Null
}
foreach ($branch in $localBranches) {
    Invoke-Git -Arguments @("branch", "-d", "--", $branch) | Out-Null
}
foreach ($branch in $remoteBranches) {
    $remoteName = $branch.Substring($Remote.Length + 1)
    Invoke-Git -Arguments @("push", $Remote, "--delete", $remoteName) | Out-Null
}
Invoke-Git -Arguments @("fetch", $Remote, "--prune") | Out-Null

$remainingLocal = @(
    Invoke-Git -Arguments @(
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads"
    ) |
        Where-Object { $_ -ne "main" }
)
$remainingRemote = @(
    Invoke-Git -Arguments @(
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/remotes/$Remote"
    ) |
        Where-Object { $_ -notin @($Remote, "$Remote/HEAD", "$Remote/main") }
)
if ($remainingLocal.Count -gt 0 -or $remainingRemote.Count -gt 0) {
    throw "Non-main branches remain after cleanup"
}

if ($InstallRuleset) {
    $repository = (& gh repo view --json nameWithOwner --jq .nameWithOwner).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $repository) {
        throw "Unable to resolve the authenticated GitHub repository"
    }
    $rulesetName = "main-only-branch-creation"
    $existingIds = @(
        & gh api --paginate "repos/$repository/rulesets" `
            --jq ".[] | select(.name == `"$rulesetName`") | .id"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect GitHub repository rulesets"
    }
    if ($existingIds.Count -gt 1) {
        throw "Multiple main-only branch-creation rulesets already exist"
    }
    $existingId = if ($existingIds.Count -eq 1) {
        ([string]$existingIds[0]).Trim()
    }
    else {
        ""
    }
    $payload = [ordered]@{
        name = $rulesetName
        target = "branch"
        enforcement = "active"
        bypass_actors = @()
        conditions = [ordered]@{
            ref_name = [ordered]@{
                include = @("~ALL")
                exclude = @("refs/heads/main")
            }
        }
        rules = @([ordered]@{ type = "creation" })
    } | ConvertTo-Json -Depth 8 -Compress
    $endpoint = if ($existingId) {
        "repos/$repository/rulesets/$existingId"
    }
    else {
        "repos/$repository/rulesets"
    }
    $method = if ($existingId) { "PUT" } else { "POST" }
    $payload | & gh api --method $method $endpoint --input - | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to install the main-only branch-creation ruleset"
    }
}

Write-Host "Main-only branch consolidation completed."
