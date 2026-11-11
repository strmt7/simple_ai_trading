# Qualified autonomous decision values

The original entry gate admitted four reproduced malformed decisions: NaN LONG
confidence, a malformed SHORT probability, NaN size and NaN price. Clamping NaN
produced full confidence/size. Converting invalid probability to zero before
inverting for SHORT produced full SHORT confidence. The position constructor
also changed invalid prices to 0.01 instead of rejecting them.

The shared boundary now accepts finite real values without boolean or string
coercion. Confidence and size must lie in `[0, 1]`; zero size remains an entry
veto. Prices must be positive and finite. Direct position construction independently
checks directional intent, explicit boolean model/AI approval, reference cash,
quantity and notional. A valid positive price is preserved without a fixed floor.
Oversized integers cannot overflow the shared mark/loss-limit validation.

The loop logs unqualified confidence without echoing its raw value and uses a
finite diagnostic signal with the rejection reason. Invalid prices terminate
the iteration before valuation or dispatch. Owned inventory and pending recovery
remain durable; remaining exposure keeps STOPPING rather than claiming flat.
Invalid entry confidence does not interrupt a qualified price-based owned exit.
No closing authority, native fee law or cash-loss limit was relaxed.

The [evidence](evidence.json) binds the original and final source phases, six
affected test modules, the additional AI consumer and final module confirmation.
The first focused phase found an oversized-integer mark overflow and one invalid
test expectation about a capped finite position. Both were corrected. A local
receipt originally referenced the previous phase's stdout hash; the actual logs,
JUnit and original receipt were preserved and reconciled without rerunning tests.
No test count is inflated by adding overlapping module confirmations.

This is an autonomous decision-boundary repair. It does not finish universal
writer/roundtrip fencing, final-intent policy/generation binding, independent
supervision, account qualification, full marked risk or native operational
acceptance. It establishes no model accuracy, financial edge, real-account
operation or whole-repository bug-hunt completion. `SAFE-02` remains partial.

## Publication check

The staged batch exposed a terminology false positive in an unchanged historical
status definition. Matching a metric phrase inside a longer word confused the
definition with a statistical claim. The scanner now checks complete phrases;
the legitimate plural forms identified by regression tests remain explicitly
prohibited. A cheap substring precheck avoids unnecessary regular-expression work.

Five obsolete whole-file exemptions, including the mutable durability ledger,
were removed. Their original bytes remain unchanged except for the separately
recorded ledger metadata. New violations in all five surfaces are checked rather
than hidden by a refreshed whole-file hash. All 37 scanner cases pass, including
the repository-wide authored-text check. The [supplement](publication-check.json)
binds this repair and the combined current source manifest. It does not establish
broader security or financial qualification.
