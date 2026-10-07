# Conditional paper inventory cash

October 7, 2026; source revision `d0e42288cefc1ab2842c64aaf258a327a36fab63`.
[Exact bindings and checks](paper-partial-cash.json) accompany this review.

`PaperOrderJournal.inventory_cash` now projects its existing immutable events,
not a parallel order ledger. It validates journal/ownership and lifecycle in one
read transaction, reconstructs incremental cash from recorded cumulative averages,
and uses rational FIFO allocation. A later opening price/fee cannot rewrite an
earlier close's cost. Residual costs and quantities remain explicit. Ambiguous
same-time events, causal over-closes (including within old tolerance), missing
increment source bindings and invalid rehashed economics reject.

The [previous synthetic example](make-take-source-integrity.md) now runs through
the virtual queue, paper journal and existing bot-owned aggressive-close adapter:
four units partially fill at 100 with 0.12 assumed quote charge, close at 98 with
0.2352, and lose 8.3552. The six-unit opening remainder stays blocking until
cancelled. The cash report survives database restart. This is not a native fill,
fee schedule, independent market observation or profitable strategy.

All 51 affected checks pass (21 new cash cases plus existing paper/queue cases);
Ruff check/format pass. One added negative-case fixture initially lacked its
parent binding; corrected that fixture before the final run. No full-suite,
hosted-CI, GPU, timing, training or automatic-protection claim. The diagnostic
requests protection at the first partial fill; it does not issue orders itself.

Remaining: exact native or qualified simulated per-fill cash, source/quote identity,
fee assets, causal event coverage, queue/role qualification, hedge/orphan cash,
automatic protective execution and new maker-target integration. Rounded upstream
averages cannot recover exact per-fill cash. UNKNOWN/blocking states preserve
uncertainty about unreported fills; reported values describe recorded events only.
Shorts/futures reject rather than silently inheriting token/Spot cash semantics.
Native qualification and profitability flags remain false. Legacy labels/results,
capture boundaries, retry gates and 201/65/37/0 research counts are unchanged.
No venue request, venue credential or live/testnet account/order action occurred. GitHub publication
uses the existing authorized path and fresh identity audit, not venue credentials;
previously surfaced shared-history identity violations remain unrewritten.
