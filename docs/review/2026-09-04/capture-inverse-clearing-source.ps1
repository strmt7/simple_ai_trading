# One source-only public PDF request. Never retry or redirect this capture.
[CmdletBinding()]
param([switch]$Preflight)
$ErrorActionPreference = 'Stop'
$sourceUrl = 'https://bin.bnbstatic.com/static/cms/cg08ou2ak0tn7mcplvfg/file/53197b612332da02c20b5b7d19b81ff53ee5f4938c6330c72a30a1ca4f91049f.pdf'
$destination = Join-Path $PSScriptRoot 'inverse-clearing-source.pdf'
$journal = Join-Path $PSScriptRoot 'inverse-clearing-source-journal.jsonl'
$ceiling = 8388608
if ((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $journal)) {
    throw 'Source capture is already consumed; preserve existing evidence.'
}
if ($Preflight) { Write-Output 'Source-only request preflight passed; no network access.'; return }
$handler = [System.Net.Http.HttpClientHandler]::new()
$handler.AllowAutoRedirect = $false
$handler.UseCookies = $false
$client = [System.Net.Http.HttpClient]::new($handler)
$client.Timeout = [TimeSpan]::FromSeconds(30)
$journalStream = [IO.StreamWriter]::new([IO.File]::Open($journal, 'CreateNew', 'Write', 'Read'), [Text.UTF8Encoding]::new($false))
$rawStream = $null
$response = $null
$sourceStream = $null
$cancel = [Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds(30))
function Record-SourceEvent([hashtable]$value) {
    $value.time_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $journalStream.WriteLine(($value | ConvertTo-Json -Compress -Depth 5))
    $journalStream.Flush()
    $journalStream.BaseStream.Flush($true)
}
try {
    Record-SourceEvent @{phase='request_started';url=$sourceUrl;method='GET';max_bytes=$ceiling;redirects=$false;script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
    $rawStream = [IO.File]::Open($destination, 'CreateNew', 'Write', 'Read')
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $sourceUrl)
    $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead, $cancel.Token).GetAwaiter().GetResult()
    $sourceStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
    $buffer = [byte[]]::new(65536)
    $count = 0
    do {
        $remaining = [Math]::Min($buffer.Length, $ceiling + 1 - $count)
        $read = $sourceStream.ReadAsync($buffer, 0, $remaining, $cancel.Token).GetAwaiter().GetResult()
        if ($read -gt 0) { $rawStream.Write($buffer, 0, $read); $count += $read }
    } while ($read -gt 0 -and $count -le $ceiling)
    $rawStream.Flush($true)
    $rawStream.Dispose(); $rawStream = $null
    $ok = ([int]$response.StatusCode -eq 200 -and $count -le $ceiling)
    Record-SourceEvent @{phase='request_completed';status=[int]$response.StatusCode;content_type=[string]$response.Content.Headers.ContentType;bytes=$count;transport_passed=$ok;raw_sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()}
    if (-not $ok) { throw 'Source response failed HTTP/size boundary; no retry.' }
    Write-Output "Source retained: $count bytes; no market or account requests."
} catch {
    if ($null -ne $rawStream) { $rawStream.Flush($true); $rawStream.Dispose(); $rawStream = $null }
    Record-SourceEvent @{phase='terminal_failure';error_type=$_.Exception.GetType().FullName;raw_sha256=$(if (Test-Path -LiteralPath $destination) {(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()} else {$null})}
    throw
} finally {
    if ($null -ne $sourceStream) { $sourceStream.Dispose() }
    if ($null -ne $response) { $response.Dispose() }
    $client.Dispose(); $cancel.Dispose(); $journalStream.Dispose()
}
