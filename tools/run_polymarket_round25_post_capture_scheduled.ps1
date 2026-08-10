[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Repository,
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$SourceDatabase,
    [Parameter(Mandatory)][string]$Plan,
    [Parameter(Mandatory)][string]$CampaignStateRoot,
    [Parameter(Mandatory)][string]$OutputRoot,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceCommit,
    [Parameter(Mandatory)][string]$StatePath,
    [Parameter(Mandatory)][string]$StdoutPath,
    [Parameter(Mandatory)][string]$StderrPath,
    [string]$TaskName = 'SimpleAITrading-Polymarket-Round25-PostCapture-v2',
    [ValidateRange(1, 512)][int]$MaximumResolutionConditions = 128,
    [ValidateSet('auto', 'cpu', 'cuda', 'directml')][string]$LightGbmBackend = 'auto',
    [ValidateSet('auto', 'cpu', 'cuda', 'directml')][string]$TcnBackend = 'auto',
    [ValidateRange(1, 24)][int]$MaximumRuntimeHours = 8,
    [ValidateRange(1, 16)][int]$MaximumLogMiB = 1
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Utf8NoBom = [Text.UTF8Encoding]::new($false)
$MaximumLogBytes = $MaximumLogMiB * 1MB
$MaximumRuntime = [TimeSpan]::FromHours($MaximumRuntimeHours)
$Tool = Join-Path $Repository 'tools\run_polymarket_round25_post_capture.py'
$SourceRoot = Join-Path $Repository 'src'

function Write-AtomicText {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Text
    )

    $temporary = "$Path.$PID.tmp"
    [IO.File]::WriteAllText($temporary, $Text, $Utf8NoBom)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Write-State {
    param([Parameter(Mandatory)][hashtable]$Values)

    $body = [ordered]@{
        schema_version = 'simple-ai-trading-round25-postcapture-supervisor-state-v1'
        observed_at_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        task_name = $TaskName
        source_commit_oid = $SourceCommit
        repository = $Repository
        source_database = $SourceDatabase
        output_root = $OutputRoot
        credentials_accessed = $false
        orders_submitted = 0
        paper_trading_authority = $false
        live_trading_authority = $false
    }
    foreach ($entry in $Values.GetEnumerator()) {
        $body[$entry.Key] = $entry.Value
    }
    $json = $body | ConvertTo-Json -Depth 8
    Write-AtomicText -Path $StatePath -Text "$json`r`n"
}

function Write-BoundedLog {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Text
    )

    $bytes = $Utf8NoBom.GetBytes($Text)
    if ($bytes.Length -le $MaximumLogBytes) {
        Write-AtomicText -Path $Path -Text $Text
        return
    }
    $tail = $Utf8NoBom.GetString(
        $bytes,
        $bytes.Length - $MaximumLogBytes,
        $MaximumLogBytes
    )
    $firstLine = $tail.IndexOf("`n")
    if ($firstLine -ge 0) {
        $tail = $tail.Substring($firstLine + 1)
    }
    Write-AtomicText -Path $Path -Text ("log_truncated_to_last_complete_lines`r`n" + $tail)
}

function Stop-OwnedProcessTree {
    param([Parameter(Mandatory)][int]$RootProcessId)

    $all = @(Get-CimInstance Win32_Process)
    $owned = [Collections.Generic.HashSet[int]]::new()
    $owned.Add($RootProcessId) | Out-Null
    do {
        $added = $false
        foreach ($candidate in $all) {
            if (
                $owned.Contains([int]$candidate.ParentProcessId) -and
                -not $owned.Contains([int]$candidate.ProcessId)
            ) {
                $owned.Add([int]$candidate.ProcessId) | Out-Null
                $added = $true
            }
        }
    } while ($added)
    foreach ($processId in @($owned) | Sort-Object -Descending) {
        $candidate = $all | Where-Object { [int]$_.ProcessId -eq $processId }
        if (
            $processId -eq $RootProcessId -or
            ($candidate.CommandLine -as [string]).Contains($Tool)
        ) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-WithinPath {
    param(
        [Parameter(Mandatory)][string]$Child,
        [Parameter(Mandatory)][string]$Parent
    )

    $prefix = $Parent.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    return $Child.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

$workerProcess = $null
$mutex = $null
$ownsMutex = $false
try {
    foreach ($value in @(
        $Repository,
        $Python,
        $SourceDatabase,
        $Plan,
        $CampaignStateRoot,
        $OutputRoot,
        $StatePath,
        $StdoutPath,
        $StderrPath
    )) {
        if ([string]::IsNullOrWhiteSpace($value) -or $value.Contains('"')) {
            throw 'Round 25 scheduled path is invalid'
        }
    }
    $Repository = [IO.Path]::GetFullPath($Repository)
    $Python = [IO.Path]::GetFullPath($Python)
    $SourceDatabase = [IO.Path]::GetFullPath($SourceDatabase)
    $Plan = [IO.Path]::GetFullPath($Plan)
    $CampaignStateRoot = [IO.Path]::GetFullPath($CampaignStateRoot)
    $OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
    $StatePath = [IO.Path]::GetFullPath($StatePath)
    $StdoutPath = [IO.Path]::GetFullPath($StdoutPath)
    $StderrPath = [IO.Path]::GetFullPath($StderrPath)
    $Tool = Join-Path $Repository 'tools\run_polymarket_round25_post_capture.py'
    $SourceRoot = Join-Path $Repository 'src'
    if (
        (Test-WithinPath -Child $StatePath -Parent $CampaignStateRoot) -or
        (Test-WithinPath -Child $StdoutPath -Parent $CampaignStateRoot) -or
        (Test-WithinPath -Child $StderrPath -Parent $CampaignStateRoot) -or
        (Test-WithinPath -Child $OutputRoot -Parent $CampaignStateRoot)
    ) {
        throw 'Round 25 scheduled output overlaps capture state'
    }
    foreach ($path in @($Repository, $SourceRoot, $CampaignStateRoot)) {
        if (-not (Test-Path -LiteralPath $path -PathType Container)) {
            throw "Required Round 25 directory is unavailable: $path"
        }
    }
    foreach ($path in @($Python, $Tool, $Plan)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required Round 25 dependency is unavailable: $path"
        }
    }
    foreach ($parent in @(
        [IO.Path]::GetDirectoryName($StatePath),
        [IO.Path]::GetDirectoryName($StdoutPath),
        [IO.Path]::GetDirectoryName($StderrPath)
    )) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $mutex = [Threading.Mutex]::new(
        $false,
        'Local\SimpleAITrading-Round25-PostCapture-v2'
    )
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        exit 0
    }
    $head = (& git -C $Repository rev-parse HEAD).Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0 -or $head -ne $SourceCommit) {
        throw 'Round 25 post-capture source commit differs'
    }
    $worktreeStatus = (& git -C $Repository status --porcelain) -join "`n"
    if ($LASTEXITCODE -ne 0 -or $worktreeStatus) {
        throw 'Round 25 post-capture worktree is not clean'
    }

    $arguments = @(
        '"' + $Tool + '"',
        '--repository', '"' + $Repository + '"',
        '--source-database', '"' + $SourceDatabase + '"',
        '--plan', '"' + $Plan + '"',
        '--state-root', '"' + $CampaignStateRoot + '"',
        '--output-root', '"' + $OutputRoot + '"',
        '--source-commit', $SourceCommit,
        '--maximum-resolution-conditions', [string]$MaximumResolutionConditions,
        '--lightgbm-backend', $LightGbmBackend,
        '--tcn-backend', $TcnBackend
    ) -join ' '
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Python
    $startInfo.Arguments = $arguments
    $startInfo.WorkingDirectory = $Repository
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.EnvironmentVariables['PYTHONPATH'] = $SourceRoot
    $workerProcess = [Diagnostics.Process]::new()
    $workerProcess.StartInfo = $startInfo
    if (-not $workerProcess.Start()) {
        throw 'Round 25 post-capture worker did not start'
    }
    $stdoutRead = $workerProcess.StandardOutput.ReadToEndAsync()
    $stderrRead = $workerProcess.StandardError.ReadToEndAsync()
    $started = [DateTimeOffset]::UtcNow
    $nextHeartbeat = $started
    $timedOut = $false
    while (-not $workerProcess.WaitForExit(5000)) {
        $now = [DateTimeOffset]::UtcNow
        if ($now - $started -gt $MaximumRuntime) {
            $timedOut = $true
            Stop-OwnedProcessTree -RootProcessId $workerProcess.Id
            break
        }
        if ($now -ge $nextHeartbeat) {
            Write-State -Values @{
                status = 'running'
                process_id = $workerProcess.Id
                elapsed_seconds = [Math]::Round(($now - $started).TotalSeconds, 3)
                output_streams_buffered = $true
            }
            $nextHeartbeat = $now.AddSeconds(30)
        }
    }
    if ($timedOut) {
        Write-State -Values @{
            status = 'owned_worker_timeout'
            process_id = $workerProcess.Id
            elapsed_seconds = [Math]::Round(
                ([DateTimeOffset]::UtcNow - $started).TotalSeconds,
                3
            )
        }
        exit 124
    }
    $workerProcess.WaitForExit()
    if (-not $workerProcess.HasExited) {
        throw 'Round 25 post-capture worker exit status is unavailable'
    }
    $stdout = $stdoutRead.GetAwaiter().GetResult()
    $stderr = $stderrRead.GetAwaiter().GetResult()
    $workerExitCode = [int]$workerProcess.ExitCode
    Write-BoundedLog -Path $StdoutPath -Text $stdout
    Write-BoundedLog -Path $StderrPath -Text $stderr
    $runnerResult = $null
    if ($workerExitCode -eq 0) {
        $lines = @($stdout -split '\r?\n' | Where-Object { $_.Trim() })
        if ($lines.Count -eq 0) {
            throw 'Round 25 post-capture worker emitted no result'
        }
        $runnerResult = $lines[-1] | ConvertFrom-Json
        if (
            -not $runnerResult.event -or
            -not $runnerResult.status -or
            [int]$runnerResult.orders_submitted -ne 0 -or
            $runnerResult.paper_trading_authority -ne $false -or
            $runnerResult.live_trading_authority -ne $false
        ) {
            throw 'Round 25 post-capture worker result differs'
        }
    }
    $stdoutHash = (
        Get-FileHash -LiteralPath $StdoutPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $stderrHash = (
        Get-FileHash -LiteralPath $StderrPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    Write-State -Values @{
        status = if ($workerExitCode -eq 0) { 'pass_complete' } else { 'pass_failed' }
        process_id = $workerProcess.Id
        exit_code = $workerExitCode
        elapsed_seconds = [Math]::Round(
            ([DateTimeOffset]::UtcNow - $started).TotalSeconds,
            3
        )
        stdout_sha256 = $stdoutHash
        stderr_sha256 = $stderrHash
        stdout_path = $StdoutPath
        stderr_path = $StderrPath
        runner_event = if ($null -eq $runnerResult) { $null } else {
            [string]$runnerResult.event
        }
        runner_status = if ($null -eq $runnerResult) { $null } else {
            [string]$runnerResult.status
        }
        source_database_opened = if (
            $null -eq $runnerResult -or
            $null -eq $runnerResult.source_database_opened
        ) { $null } else { [bool]$runnerResult.source_database_opened }
    }
    exit $workerExitCode
}
catch {
    $stateParent = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($StatePath))
    New-Item -ItemType Directory -Path $stateParent -Force | Out-Null
    Write-State -Values @{
        status = 'supervisor_error'
        error_type = $_.Exception.GetType().Name
        error = $_.Exception.Message
    }
    exit 1
}
finally {
    if ($null -ne $workerProcess) {
        $workerProcess.Dispose()
    }
    if ($ownsMutex) {
        $mutex.ReleaseMutex()
    }
    if ($null -ne $mutex) {
        $mutex.Dispose()
    }
}
