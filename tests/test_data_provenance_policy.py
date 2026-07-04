from __future__ import annotations

from tools.audit_data_provenance import audit


def test_tracked_repo_does_not_publish_synthetic_financial_evidence() -> None:
    assert audit() == []
