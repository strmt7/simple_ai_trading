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
verified provider identifies itself as `0.9.4+superzip.portable2`, with a
Windows download-opener repair and a separate NLTK security source build.
These are provider-specific repairs, not features inferred from stock 0.9.4.
Preserve its full licenses and attribution. No provider files were copied or
modified, and no new runtime dependency or paid service was installed here.

Configure an available provider checkout and compatible CPython 3.13/3.14
explicitly. The paths below are host examples, not hardcoded application paths.
The provider validates its dependency/repair identities on reuse; its installer
chooses an external OS cache, serializes setup and provisions only when needed.

```powershell
$crawlerCheckout = 'C:\SuperZip'
$crawlerPython = '<absolute path to compatible Python>'
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
   Check the canonical URL against retained journals and source bindings first.
   Reuse admissible saved content; changing the crawler is not a retry trigger.
2. Before access, record the distinct research question, selected canonical URL,
   provider/version, new output path, page/time limits and terminal consequence.
   Ensure the output parent exists and the file does not. The upstream CLI can
   overwrite files: operators must reserve a new destination before invocation.
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

Missing provider setup is an explicit tool limitation, not permission to claim
that Crawl4AI was used. Qualify a provider change only in its own affected lane;
do not repeat a broad website matrix or trading suite for ordinary page reading.

The [October 8 retained run](review/2026-10-08/crawl4ai-research-result.json)
used this integration on official USD-M trade documentation. Native success was
true, but HTTP 202 and one Markdown character rejected source admission. No
semantic, economic or account qualification follows; no alias or retry occurred.

This product includes software developed by UncleCode (https://x.com/unclecode)
as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).
