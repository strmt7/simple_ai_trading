[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Repository,

    [Parameter(Mandatory = $true)]
    [string]$CaptureRepository
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = (Resolve-Path -LiteralPath $Repository).Path
$captureRoot = (Resolve-Path -LiteralPath $CaptureRepository).Path
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$runner = Join-Path $repositoryRoot "tools\run_round17_polymarket_one_use.py"
$publisher = Join-Path $repositoryRoot "tools\publish_round17_polymarket_result.py"
$dataRoot = Join-Path $repositoryRoot "data"
$development = Join-Path $dataRoot "polymarket-round17-development-v1.json"
$claimStore = Join-Path $dataRoot "polymarket-round17-one-use-v1.sqlite3"
$resolutions = Join-Path $dataRoot "polymarket-round17-test-resolutions-v1.json"
$output = Join-Path $dataRoot "polymarket-round17-one-use-result-v1.json"
$log = Join-Path $dataRoot "polymarket-round17-one-use-v1.log"
$done = Join-Path $dataRoot "polymarket-round17-one-use-v1.done"
$publicationRoot = Join-Path $repositoryRoot (
    "docs\model-research\polymarket\evidence\round-017-one-use-v1"
)
$publicationManifest = Join-Path $publicationRoot "publication-manifest.json"
if (
    -not (Test-Path -LiteralPath $python -PathType Leaf) -or
    -not (Test-Path -LiteralPath $runner -PathType Leaf) -or
    -not (Test-Path -LiteralPath $publisher -PathType Leaf)
) {
    throw "Round 17 one-use scheduled runtime is unavailable."
}
New-Item -ItemType Directory -Path $dataRoot -Force | Out-Null

$arguments = @(
    $runner,
    "--repository", $repositoryRoot,
    "--campaign-plan", "docs/model-research/polymarket/round-014-btc-5m-campaign-plan-v1.json",
    "--cohort-plan", "docs/model-research/polymarket/round-017-btc-5m-cohort-plan-v1.json",
    "--admission-spec", "docs/model-research/polymarket/round-014-btc-5m-admission-spec-v1.json",
    "--database", (Join-Path $captureRoot "data\polymarket-round14-prospective-v1.duckdb"),
    "--state-root", (Join-Path $captureRoot "data\polymarket-round14-prospective-v1-state"),
    "--evaluation-contract", "docs/model-research/polymarket/round-017-btc-5m-one-use-evaluation-contract-v1.json",
    "--development-result", $development,
    "--risk-contract", "docs/model-research/polymarket/round-014-btc-5m-prospective-contract-v1.json",
    "--claim-store", $claimStore,
    "--resolution-checkpoint", $resolutions,
    "--output", $output,
    "--memory-limit", "2GB",
    "--database-threads", "2"
)

$commandParts = @($python) + $arguments | ForEach-Object {
    '"' + ([string]$_).Replace('"', '""') + '"'
}
$command = ($commandParts -join " ") + " 2>&1"
& $env:ComSpec /d /s /c $command | Tee-Object -FilePath $log -Append
$runnerExitCode = $LASTEXITCODE
if ($runnerExitCode -ne 0) {
    exit $runnerExitCode
}

$publicationArguments = @(
    $publisher,
    "--result", $output,
    "--output-dir", $publicationRoot
)
$publicationCommandParts = @($python) + $publicationArguments | ForEach-Object {
    '"' + ([string]$_).Replace('"', '""') + '"'
}
$publicationCommand = ($publicationCommandParts -join " ") + " 2>&1"
& $env:ComSpec /d /s /c $publicationCommand |
    Tee-Object -FilePath $log -Append
$publisherExitCode = $LASTEXITCODE
if ($publisherExitCode -ne 0) {
    exit $publisherExitCode
}
if (-not (Test-Path -LiteralPath $publicationManifest -PathType Leaf)) {
    throw "Round 17 publication manifest is unavailable after publication."
}
$manifest = Get-Content -LiteralPath $publicationManifest -Raw |
    ConvertFrom-Json
Set-Content -LiteralPath $done -Value @(
    "completed_utc=" + [DateTimeOffset]::UtcNow.ToString("O"),
    "result_sha256=" + [string]$manifest.result_sha256,
    "publication_sha256=" + [string]$manifest.publication_sha256
) -Encoding Ascii
exit 0
