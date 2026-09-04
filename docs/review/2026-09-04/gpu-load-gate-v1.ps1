param(
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [ValidateRange(10, 60)][int]$SampleCount = 15
)

$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $OutputPath) { throw "Load receipt already exists" }
$Samples = [System.Collections.Generic.List[object]]::new()
$Failure = $null
$Started = [DateTime]::UtcNow.ToString("o")
try {
    # Passive measurement only. No process, priority, affinity or task changes.
    $Paths = @('\Processor(_Total)\% Processor Time',
        '\PhysicalDisk(_Total)\Avg. Disk Queue Length',
        '\Memory\Available MBytes', '\GPU Engine(*)\Utilization Percentage')
    Get-Counter -Counter $Paths -SampleInterval 1 -MaxSamples $SampleCount | ForEach-Object {
        $Set = $_
        $Cpu = @($Set.CounterSamples | Where-Object Path -like '*\processor(_total)\*')
        $Disk = @($Set.CounterSamples | Where-Object Path -like '*\physicaldisk(_total)\*')
        $Memory = @($Set.CounterSamples | Where-Object Path -like '*\memory\available mbytes')
        $Gpu = @($Set.CounterSamples | Where-Object Path -like '*\gpu engine(*)\*')
        if ($Cpu.Count -ne 1 -or $Disk.Count -ne 1 -or $Memory.Count -ne 1 -or $Gpu.Count -eq 0) {
            throw "Required load counters unavailable"
        }
        foreach ($Counter in $Set.CounterSamples) {
            if ($Counter.Status -notin @(0, 1) -or [double]::IsNaN($Counter.CookedValue) -or
                [double]::IsInfinity($Counter.CookedValue) -or $Counter.CookedValue -lt 0) {
                throw "Invalid load counter; do not assume zero load"
            }
        }
        # Sum process shares for each physical engine, then take the busiest
        # engine. Do not sum different engines into a fake GPU percentage.
        $EngineTotals = $Gpu | Group-Object { $_.InstanceName -replace '^pid_\d+_', '' } |
            ForEach-Object { ($_.Group | Measure-Object -Property CookedValue -Sum).Sum }
        $GpuBusy = ($EngineTotals | Measure-Object -Maximum).Maximum
        $Quiet = $Cpu[0].CookedValue -le 10 -and $Disk[0].CookedValue -le 1 -and
            $Memory[0].CookedValue -ge 24576 -and $GpuBusy -le 5
        $Samples.Add([pscustomobject]@{
            time_utc = $Set.Timestamp.ToUniversalTime().ToString('o')
            cpu_percent = $Cpu[0].CookedValue
            busiest_gpu_engine_percent = $GpuBusy
            disk_queue_length = $Disk[0].CookedValue
            available_memory_mib = $Memory[0].CookedValue
            quiet = $Quiet
        })
    }
} catch {
    $Failure = $_.Exception.GetType().FullName
}
$Eligible = $null -eq $Failure -and $Samples.Count -eq $SampleCount -and
    @($Samples | Where-Object { -not $_.quiet }).Count -eq 0
$Receipt = [ordered]@{
    schema_version = 'passive-benchmark-preflight-load-v1'
    implementation_sha256 = (Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    started_at_utc = $Started
    finished_at_utc = [DateTime]::UtcNow.ToString('o')
    can_start_at_measurement_end = $Eligible
    benchmark_launched = $false
    unrelated_processes_modified = $false
    thresholds = @{ cpu_percent=10; gpu_engine_percent=5; disk_queue=1; min_available_mib=24576 }
    failure = $Failure
    samples = $Samples.ToArray()
    scope = 'Preflight only, not a during-run contention monitor. Valid only immediately after measurement. A new background workload invalidates timing attribution; do not use this receipt to bless earlier or later runs.'
}
$Receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutputPath -Encoding utf8
[pscustomobject]@{can_start_at_measurement_end=$Eligible; samples=$Samples.Count; failure=$Failure} | ConvertTo-Json
