from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_trading import ai_review, ai_start_gate
from simple_ai_trading.ai_uplift import _model_parameters_b


def test_ai_review_transport_and_numeric_boundaries_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> object:
            return {"status": "ok"}

    def fake_post(url: str, **kwargs: object) -> Response:
        observed["url"] = url
        observed.update(kwargs)
        return Response()

    monkeypatch.setattr(ai_review.requests, "post", fake_post)
    payload = {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "review"}],
        "stream": False,
    }

    assert ai_review._post_json("http://127.0.0.1/api/chat", payload, 0.0) == {
        "status": "ok"
    }
    assert observed["json"] == payload
    assert observed["timeout"] == pytest.approx(0.1)
    with pytest.raises(TypeError):
        ai_review._post_json(
            "http://127.0.0.1/api/chat",
            {"unsupported": object()},
            1.0,
        )

    assert ai_review._finite("1.25") == pytest.approx(1.25)
    assert ai_review._finite(True, -1.0) == -1.0
    assert ai_review._finite(object(), -1.0) == -1.0
    assert ai_review._optional_finite("2.5") == pytest.approx(2.5)
    assert ai_review._optional_finite(False) is None
    assert ai_review._optional_finite(object()) is None
    assert _model_parameters_b("", True) is None  # type: ignore[arg-type]

    compact = ai_review._compact_model_lab_report(
        {
            "requested_objectives": object(),
            "accepted_symbols": object(),
            "portfolio_risk": {"accepted_symbols": object()},
        }
    )
    assert compact["requested_objectives"] == []
    assert compact["accepted_symbols"] == []
    assert compact["portfolio_risk"]["accepted_symbols"] == []


def test_ai_json_loaders_map_json_and_encoding_failures(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="source model-lab report is unreadable"):
        ai_start_gate._strict_json_mapping(source)

    source.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="source model-lab report is unreadable"):
        ai_start_gate._strict_json_mapping(source)

    review = tmp_path / "review.json"
    review.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="artifact is unreadable"):
        ai_review.load_ai_review_report(review)
