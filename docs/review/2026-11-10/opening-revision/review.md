# Preserve portfolio approval through order reservation

The Polymarket supervisor checked a proposal against its current portfolio,
then fetched a fresh executable quote. The gateway took its own ledger revision
after that fetch, so a change in between could go unnoticed by both checks.
In the synthetic reproduction, an unrelated prepared order became terminal
while the quote was being fetched. The ledger advanced from revision 0 to 2,
and the opening still reached the synthetic venue callback once.

Promoted opening now carries the exact revision from the supervisor's portfolio
check. The gateway rejects a mismatch before preflight and signing. The same
revision is checked inside the SQLite reservation transaction, including when
the matching reservation already exists. In the repeated reproduction the
ledger made the same change and the callback count fell to zero. An opening
with the current approval still reaches the gateway normally. The independent
owned close path uses its existing ownership and quantity checks.

All 160 focused cases passed. The fixture changes an order record without
creating a realized loss. The revision check covers the local ledger through
reservation. Venue state can change independently, and hard process containment
remains open. [Verification evidence](evidence.json)
