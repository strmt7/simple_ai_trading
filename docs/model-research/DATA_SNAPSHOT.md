# Research data snapshot

Reusable historical research data is fixed at event times strictly before
`2026-08-14 00:00:00 UTC`. The boundary is an upper limit, not a claim that
every source is complete through August 13.

Certified source rows are immutable. New derived columns or tables may be
added only from the same bounded inputs with a new identity and provenance;
missing observations must remain explicit and may not be fabricated or forward
filled.

The active Round 27, Round 28, and Round 75 campaigns are prospective evidence.
They keep their frozen schedules and internal train/evaluation rules, but their
rows cannot silently extend or merge into the reusable historical snapshot.

Run the read-only guard before historical model work:

```powershell
.\.venv\Scripts\python.exe tools\audit_research_data_snapshot.py
```

The machine-readable authority is
`research-data-snapshot-contract-v1.json`. The audit verifies exact artifact
identities and event-date bounds without opening a database, using the network,
or modifying an active capture.
