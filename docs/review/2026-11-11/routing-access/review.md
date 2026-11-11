# Routing API distinction and capital notices

The retained official documents distinguish Smart Order Routing (SOR) from
Auto Order Routing (AOR). SOR uses interchangeable quotes at fixed 1:1 and
documents dedicated REST/WebSocket entry points. Its fills can include both
ordinary trades and allocations; a future integration must reconcile each
through the appropriate receipt source. The examples omit commission and do
not prove current symbols or capacity. [Official SOR FAQ][sor]

The previously retained AOR FAQ describes intermediary conversion at a rate
that may differ from Spot. Avoiding a separate conversion commission does not
remove spread or prove improvement over direct execution. Its interface/order
descriptions do not establish API or testnet access. No such access is proved
by finding SOR endpoints in a different document. Phrase absence in the FAQ
also does not prove that AOR lacks an API. [AOR source and arithmetic](../binance-auto-routing/review.md)

The November 11 changelog announces surplus quote-lock release for partially
filled resting BUY limits after November 20 at 07:00 UTC. This may change a
future capital path; it is not evidence of an already available balance or
permission to revise consumed cash studies. The separate `quoteNotionalLimit`
feature has no decided enablement time, and SOR rejects that parameter. A
future integration requires actual supported capability and native balance/
order evidence. [Official API changelog][changes]

Both frozen crawls returned substantive official content with HTTP 200, one
direct attempt each and no fallback. Raw native results remain private. The
earlier AOR FAQ was reused without another request. The first local adjudication
check confused a logical-text digest with Windows-translated file bytes; the
terminal evidence binds both separately, preserving the original extraction.

The [evidence](evidence.json) and [registry update](registry-update.json) add
source metadata only. Counts remain 206 observations / 65 hypotheses /
37 historical scoped mechanisms / **0 qualified stable edges**. No market,
configuration, account or order request followed, and no protected partial or
historical result changed. The old SOR screen stays consumed under its exact
material-change condition. AOR still needs authoritative access/eligibility and
separately eligible matched net-proceeds/capacity evidence for legitimate flow.

[sor]: https://github.com/binance/binance-spot-api-docs/blob/master/faqs/sor_faq.md
[changes]: https://github.com/binance/binance-spot-api-docs/blob/master/CHANGELOG.md
