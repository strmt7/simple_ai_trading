# Exactly two source-only public GETs. No redirects, retries, or market access.
[CmdletBinding()]
param([switch]$Preflight)
$ErrorActionPreference = 'Stop'
$sourceItems = @(
    @{ Name='mechanism'; Url='https://docs.paradex.trade/risk/funding-mechanism' },
    @{ Name='history-api'; Url='https://docs.paradex.trade/api/prod/markets/get-funding-data' }
)
$outputRoot = Join-Path $PSScriptRoot 'paradex-index-source'
$journalPath = Join-Path $outputRoot 'requests.jsonl'
if (Test-Path -LiteralPath $journalPath) { throw 'Source run consumed; do not retry.' }
foreach ($item in $sourceItems) {
    if (Test-Path -LiteralPath (Join-Path $outputRoot ($item.Name + '.html'))) { throw 'Existing raw body; do not overwrite.' }
}
if ($Preflight) { Write-Output 'Two source-only GETs preflighted; no network access.'; return }
[IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$journal = [IO.StreamWriter]::new([IO.File]::Open($journalPath, 'CreateNew', 'Write', 'Read'), [Text.UTF8Encoding]::new($false))
$handler = [Net.Http.HttpClientHandler]::new()
$handler.AllowAutoRedirect = $false
$handler.UseCookies = $false
$client = [Net.Http.HttpClient]::new($handler)
$client.Timeout = [TimeSpan]::FromSeconds(30)
function Record-Event([hashtable]$value) {
    $value.time_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $journal.WriteLine(($value | ConvertTo-Json -Compress -Depth 5))
    $journal.Flush(); $journal.BaseStream.Flush($true)
}
try {
    foreach ($item in $sourceItems) {
        $rawPath = Join-Path $outputRoot ($item.Name + '.html')
        $raw = $null; $body = $null; $response = $null; $request = $null
        $cancel = [Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds(30))
        $ceiling = 5242880
        try {
            Record-Event @{ phase='started'; name=$item.Name; method='GET'; url=$item.Url; max_bytes=$ceiling; redirects=$false; script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() }
            $raw = [IO.File]::Open($rawPath, 'CreateNew', 'Write', 'Read')
            $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $item.Url)
            $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead, $cancel.Token).GetAwaiter().GetResult()
            $body = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $buffer = [byte[]]::new(65536); $count = 0
            do {
                $remaining = [Math]::Min($buffer.Length, $ceiling + 1 - $count)
                $read = $body.ReadAsync($buffer, 0, $remaining, $cancel.Token).GetAwaiter().GetResult()
                if ($read -gt 0) { $raw.Write($buffer, 0, $read); $count += $read }
            } while ($read -gt 0 -and $count -le $ceiling)
            $raw.Flush($true); $raw.Dispose(); $raw = $null
            $ok = ([int]$response.StatusCode -eq 200 -and $count -le $ceiling)
            Record-Event @{ phase='completed'; name=$item.Name; status=[int]$response.StatusCode; bytes=$count; passed=$ok; content_type=[string]$response.Content.Headers.ContentType; raw_sha256=(Get-FileHash -LiteralPath $rawPath -Algorithm SHA256).Hash.ToLowerInvariant() }
            Write-Output "$($item.Name): $count bytes, transport passed $ok"
            if (-not $ok) { throw 'Source failed HTTP/size boundary; remaining requests not launched.' }
        } catch {
            if ($null -ne $raw) { $raw.Flush($true); $raw.Dispose(); $raw = $null }
            Record-Event @{ phase='failed'; name=$item.Name; error_type=$_.Exception.GetType().FullName; raw_sha256=$(if (Test-Path -LiteralPath $rawPath) {(Get-FileHash -LiteralPath $rawPath -Algorithm SHA256).Hash.ToLowerInvariant()} else {$null}) }
            throw
        } finally {
            if ($null -ne $body) { $body.Dispose() }
            if ($null -ne $response) { $response.Dispose() }
            if ($null -ne $request) { $request.Dispose() }
            $cancel.Dispose()
        }
    }
} finally { $client.Dispose(); $journal.Dispose() }
