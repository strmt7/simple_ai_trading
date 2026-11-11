# Mandatory Crawl4AI research

The user explicitly requires Crawl4AI for web-page research. The requested
`crawl4i` name was resolved by inspecting their other local repositories:
SuperZip contains the actual Crawl4AI integration; no separate tool with that
name was found in the available project search. Search engines may discover
sources, but page-content research must use Crawl4AI, not search snippets.

## Reuse the isolated provider

The trading checkout did not contain Crawl4AI. Reuse SuperZip's independently
managed, hash-locked development integration instead of importing a browser,
LLM stack or an unpatched crawler into the trading runtime. Its source is
[SuperZip at the inspected revision](https://github.com/strmt7/SuperZip/tree/847e54719e650981d2432c05416271e1c6b783d0).
Read that checkout's `docs/crawl4ai.md` before setup or provider changes. The
October 8 provider identified itself as `0.9.4+superzip.portable2`. That is
historical evidence, not a permanent assertion about the other checkout.
On November 9 the existing checkout identifies itself as FlitzZip with provider
`0.9.5+flitzzip.portable2`, a Windows download-opener repair and the pinned
official NLTK 3.10.4 distribution. Re-read its current guide and verify its exact
tool, lock and admitted cache identity before reuse; do not infer runtime
qualification from a version label. These are provider-specific repairs.
Preserve its full licenses and attribution. No provider files were copied or
modified, and no new runtime dependency or paid service was installed here.

Configure an available provider checkout and compatible CPython 3.13/3.14
explicitly through private session configuration. Do not put workstation paths
in tracked examples or application defaults.
The provider validates its dependency/repair identities on reuse; its installer
chooses an external OS cache, serializes setup and provisions only when needed.

```powershell
$crawlerCheckout = $env:SIMPLE_AI_TRADING_CRAWL4AI_CHECKOUT
$crawlerPython = $env:SIMPLE_AI_TRADING_CRAWL4AI_PYTHON
if (-not $crawlerCheckout -or -not $crawlerPython) { throw 'Configure the provider checkout and compatible interpreter.' }
& $crawlerPython "$crawlerCheckout\tools\crawl4ai_tool.py" crawl --help
& $crawlerPython "$crawlerCheckout\tools\crawl4ai_tool.py" crawl '<selected public HTTPS page>' -o all -O '<new absolute result.json>' -c check_robots_txt=true,page_timeout=45000 --json-ensure-ascii
```

Use the native `crawl` CLI, upstream headless defaults and robots checks.
`-o all` retains the native result including HTML, Markdown, URL, HTTP status
and errors. `-o json` means extracted structured content, not the full result;
it is unsuitable for this evidence role without a separately qualified strategy.
Do not select personal profiles, visible browsers, proxies, stealth, external
LLMs, deep crawling, scripts, logins or account endpoints in this workflow.
Installation/help success does not prove a successful source extraction.

## Evidence and access rules

1. Read `RESEARCH_CAPTURE_BOUNDARIES.md` and the exact controlling contract.
   Check the canonical URL and stable article identifier against retained result,
   journal, receipt, adjudication and source-binding records first. A lookup
   limited to contract/plan filenames does not cover these records.
   Reuse admissible saved content; changing the crawler is not a retry trigger.
2. Before access, record the distinct research question, selected canonical URL,
   provider/version, new output path, page/time limits and terminal consequence.
   Ensure the output parent exists and the file does not. The upstream CLI can
   overwrite files: operators must reserve a new destination before invocation.
   Preflight the actual controller entry point, project imports, owned process
   lifecycle and durable output writes before freezing access. A private script
   outside the package tree needs a project-root module or reviewed `runpy`
   entry point; do not discover its import failure during capture.
   Declare any allowed same-publisher canonical redirects before access, including
   the intended document scope. Retain original and final URLs and both native
   `status_code` and `redirected_status_code` when present. For an allowed
   redirect, qualify the effective final status; a different document, host,
   challenge or forbidden redirect still fails. Never retrofit this policy into
   a consumed exact-URL contract or fetch its canonical alias to repair failure.
3. Retain the full result outside Git until source and credential-pattern review.
   Record timestamps, exact result hash/size, native success, status, URL and
   content fingerprint. Read bounded relevant sections, not the entire payload.
   An output-size check after extraction is not a hard network-byte ceiling.
   Native `success=true` alone is insufficient: require HTTP 200, substantive
   extracted source text, no challenge and the exact intended document scope.
4. A single page navigation may fetch robots, scripts and other resources; it is
   not exactly one HTTP GET. Rendered HTML is not the original wire body. Neither
   Markdown nor a successful crawl proves exact API bytes, native fills, account
   eligibility, settlement identity, current profitability or complete coverage.
5. Preserve timeout, robots refusal, challenge, empty and product-scope failures.
   Do not bypass them, silently switch tools, alter consumed contracts or scrape
   aliases to get the desired answer. Exact API/stream/archive evidence continues
   through its existing bounded native transport; offline processing of retained
   pages is allowed only with its provenance and no new source-access claim.
6. Keep historical results untouched. Admit financial facts only after the exact
   source, date, product, units, after-cost economics and evaluation gates pass.
   If a contract forbids browser subrequests, use Crawl4AI for separate eligible
   documentation research, not as a replacement transport for that contract.

For the November 9 resumed session, the user's explicit delegation permits
routine public technical-document decisions without another approval request.
A separately selected official source copy or immutable revision may receive a
new bounded documentation-only attempt after a failed page. Freeze the question,
version, URL, limits and admission first; retain the earlier failure unchanged.
This exception supersedes the documentation-only no-alias language above, not
robots/access controls, mandatory crawler use, financial-study retry triggers,
protected captures or account boundaries. A public publisher repository is a
new source receipt, never a retroactive success for a failed website capture.
Use an as-of revision for historical semantics and keep collector configuration
and clock-offset qualification separate from the protocol's documented defaults.

Missing provider setup is an explicit tool limitation, not permission to claim
that Crawl4AI was used. Qualify a provider change only in its own affected lane;
do not repeat a broad website matrix or trading suite for ordinary page reading.

After a Binance HTML source returns an explicit challenge and empty content,
do not cycle through neighboring legal/support pages to recover the same claim.
Reuse the source-bound access result. Further work needs an independently
published relevant document/revision or independently observed access-policy
change, with its own deduplication and frozen admission. This route guidance
does not authorize challenge bypass or change financial/account retry gates.

## Bounded native PDF extraction

For separately selected primary PDFs, the HTML CLI is insufficient: the current
provider's CLI replaces the configured PDF scraper with its HTML strategy.
Use [the owned-process controller](../tools/capture_crawl4ai_pdf.py) and
[native PDF worker](../tools/crawl4ai_pdf_source.py) instead. The worker pairs
`PDFCrawlerStrategy` with `PDFContentScrapingStrategy`; extraction is native
Crawl4AI, with a reviewed Requests adapter for bounded raw downloads. The native
PDF crawler's status 200 is a placeholder and is never HTTP admission evidence.

Keep the existing provider unchanged. Install only the separately reviewed
[optional parser lock](../tools/requirements/crawl4ai-pdf.txt) into a new owned
extension directory with dependency resolution disabled and hashes required.
Bind the extension's complete file inventory, installed parser identity, native
module bytes, read-only provider dependency/repair verification and capture code
before access. This profile is qualified for the exact provider and optional
parser versions in its evidence, not arbitrary future library combinations.

The frozen private plan supplies explicit provider/configuration paths, their
digests, exact URL, new output destinations, text/page gates and code hashes.
It also binds the supervisor's 150-second deadline. Run its real entry point
with `python -m tools.capture_crawl4ai_pdf --plan <private-plan> --preflight`
before the single authorized invocation without `--preflight`. The controller
reserves a durable one-use journal before launching the hidden owned child;
reusing any output is rejected, including after a failed launch.
The supervised worker uses an explicit environment and a fresh import-cache
prefix. `-B` alone prevents cache writes but still permits cached-bytecode reads.

Both robots and document requests have no retries, redirects, proxies, profiles,
authentication or LLM. Robots must be qualified before the document request;
403/429/5xx, redirects, timeouts and malformed policy fail closed. Explicit
404/410 absence is permitted. DNS and the connected peer must be public, TLS
verification remains enabled, and the same user agent is used for policy and
download. One robots request and at most one document request are permitted.
The hard retained-body ceilings are 512 KiB and 8 MiB respectively, plus one
overflow byte; partial bodies and actual status/headers remain evidence.

Keep raw PDF, native scraping/result JSON, timestamps, journals and hashes in
ignored storage. Reject encoded bodies, invalid/encrypted PDF, more than the
declared maximum of 64 pages, missing/empty page extraction and insufficient
text. Check the actual complete PDF page count: the upstream page limit can
silently truncate extraction. Only the temporary copy is deleted; original
source bytes stay retained. Local fixtures use only their explicitly owned
loopback server, never a fallback reader or a remote test population. Technical
admission still requires separate review of title, product, date and legal scope.

The [October 8 retained run](review/2026-10-08/crawl4ai-research-result.json)
used this integration on official USD-M trade documentation. Native success was
true, but HTTP 202 and one Markdown character rejected source admission. No
semantic, economic or account qualification follows; no alias or retry occurred.

The [November 9 source check](review/2026-11-09/structural-change-discovery/review.md)
used the already-installed current cohort and its hidden owned-child cleanup.
Its separate source returned HTTP 202/one Markdown character and was also
excluded, without an alternative reader or retry. No persistent task was created.

The [historical-data source review](review/2026-11-09/paired-data-source/review.md)
preserves an exact-URL failure caused by a 307 redirect despite a substantive
final HTTP-200 document. A separately frozen zero-network review used retained
bytes for documentation routing only. Three distinct technical documents then
passed their own predeclared source gates. Neither the failed document nor its
canonical alias was refetched, and the original second navigation stayed forbidden.

This product includes software developed by UncleCode (https://x.com/unclecode)
as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).
