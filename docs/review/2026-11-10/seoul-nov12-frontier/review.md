# Seoul November 12 fixed temperature partition

One frozen Gamma request returned HTTP 200, 90,291 bytes and event 1354459.
It contained all eleven active binary bins from 10°C or below through 20°C or
higher. The eleven descriptions and source URLs matched. They use routine
Incheon Airport observations on the station's local November 12, rounded to
whole Celsius degrees with half ties rounded upward. The descriptions also
share the secondary-source and lowest-bracket fallback. An erroneous-data
clarification remains a separate settlement contingency.
[Capture contract](contract.json) · [Retained result](result.json)

Buying one YES per outcome cost 1.097 pUSD at the supplied asks, against a
one-pUSD ordinary-resolution payout. The rejection is stronger than that
single example. Eleven YES asks and ten NO proxies admitted exact state
weights that sum to one and satisfy all 21 acquisition inequalities. Under
the ordinary one-winner model, this rules out a strictly positive gross
payout floor across arbitrary nonnegative YES and NO quantities. The
upper-tail NO proxy was unavailable. The proof excludes that instrument
and treats the available prices as linear Gamma proxies. It makes no claim
about executable prices, fills, depth or expected returns.
[Certificate method](../partition-cover-certificate/review.md)

The independent arithmetic check read the original decimals and verified
every inequality and the all-YES sum. The source attempt stopped at the
gross rejection, so no order book, fee endpoint, account or order followed.
The registry records the exact event as terminal and raises the observation
count to 206. Zero stable edges are qualified. Future work still requires
an independently eligible event or a material trigger for the family.
[Registry update](registry-update.json)
