import json

from tools.review_collateral_stress import ROOT, review


def test_synthetic_collateral_artifact_reconstructs():
    path = ROOT / "docs/review/2026-09-05/collateral-stress-review.json"
    assert json.loads(path.read_bytes()) == json.loads(json.dumps(review(), default=str))
