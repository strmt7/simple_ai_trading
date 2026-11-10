# Recheck wallet backing before opening exposure

Polymarket checked wallet backing at startup. The venue funding check for each
opening covered only the order's cost and allowance. A synthetic opening with
a 10,000 pUSD configured allocation still reached the venue callback when the
reported balance dropped to 100 pUSD. The order needed 2.58750 pUSD, and the
ledger revision remained current throughout.

The supervisor now carries the approved ledger revision, configured allocation
and verified remaining cost basis of owned positions in one opening approval.
The gateway adds that basis to the available collateral already returned by
the order funding request. It blocks a shortfall before signing. It still
checks the order's own balance and allowance and verifies the approval revision
at reservation. The change requires no additional venue request.

A shortfall updates the reported support and blocks further openings in that
session. A fresh startup rechecks backing. This applies the existing nominal
allocation test throughout the session, so a settled loss can also leave an
allocation underfunded. Operators can restore backing or select a lower
allocation for a fresh startup. The configured allocation is never reduced
automatically. Owned reduction and Stop retain their separate gates.

The synthetic wallet drop now stops before the venue callback. A second test
checks that verified owned basis counts once and that exact full backing admits
the opening. The earlier quote-time revision interleaving is also reproduced
against this combined version. All 162 focused cases passed. Venue collateral
may still change after the funding response, and cost basis is a nominal
accounting measure.
[Verification evidence](evidence.json)
