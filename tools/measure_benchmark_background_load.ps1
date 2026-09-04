param(
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [ValidateRange(10, 60)][int]$SampleCount = 15,
    [ValidateRange(0, 90)][double]$ReservedCpuPercent = 50,
    [ValidateRange(0, 90)][double]$ReservedGpuPercent = 40,
    [ValidateRange(1, 64)][double]$ReservedMemoryGiB = 12,
    [ValidateRange(0.1, 100)][double]$MaxDiskQueueLength = 4
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
        # Workload budgets are explicit estimates, not measured future demand.
        # Leave 10 percentage points for CPU/GPU and 4 GiB beyond our RAM budget.
        $Headroom = $Cpu[0].CookedValue + $ReservedCpuPercent -le 90 -and
            $Disk[0].CookedValue -le $MaxDiskQueueLength -and
            $Memory[0].CookedValue -ge (($ReservedMemoryGiB + 4) * 1024) -and
            $GpuBusy + $ReservedGpuPercent -le 90
        $Samples.Add([pscustomobject]@{
            time_utc = $Set.Timestamp.ToUniversalTime().ToString('o')
            cpu_percent = $Cpu[0].CookedValue
            busiest_gpu_engine_percent = $GpuBusy
            disk_queue_length = $Disk[0].CookedValue
            available_memory_mib = $Memory[0].CookedValue
            headroom = $Headroom
        })
    }
} catch {
    $Failure = $_.Exception.GetType().FullName
}
$ConsecutivePressure = 0
$LongestPressure = 0
foreach ($Sample in $Samples) {
    if ($Sample.headroom) { $ConsecutivePressure = 0 } else { $ConsecutivePressure++ }
    $LongestPressure = [Math]::Max($LongestPressure, $ConsecutivePressure)
}
$Eligible = $null -eq $Failure -and $Samples.Count -eq $SampleCount -and $LongestPressure -lt 3
$Receipt = [ordered]@{
    schema_version = 'passive-benchmark-headroom-v2'
    implementation_sha256 = (Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    started_at_utc = $Started
    finished_at_utc = [DateTime]::UtcNow.ToString('o')
    can_start_at_measurement_end = $Eligible
    benchmark_launched = $false
    unrelated_processes_modified = $false
    thresholds = @{
        combined_cpu_ceiling_percent=90; reserved_cpu_percent=$ReservedCpuPercent
        combined_gpu_ceiling_percent=90; reserved_gpu_percent=$ReservedGpuPercent
        reserved_memory_gib=$ReservedMemoryGiB; remaining_memory_reserve_gib=4
        disk_queue=$MaxDiskQueueLength; pressure_samples_to_defer=3
    }
    longest_consecutive_pressure_samples = $LongestPressure
    failure = $Failure
    samples = $Samples.ToArray()
    scope = 'Advisory headroom preflight, not a proof of no contention or a during-run monitor. CPU/GPU reservations must reflect the intended workload, including concurrent workers. Allow brief spikes but defer sustained pressure. Valid only immediately after measurement; record actual competing load during the run and do not use this receipt to bless earlier or later timings.'
}
$Receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutputPath -Encoding utf8
[pscustomobject]@{can_start_at_measurement_end=$Eligible; samples=$Samples.Count; failure=$Failure} | ConvertTo-Json
