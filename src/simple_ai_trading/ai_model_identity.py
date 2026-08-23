"""Validated local foundation-model identity shared across AI evidence gates."""

from __future__ import annotations

from dataclasses import dataclass


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_parameter_size_label(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 64
        and value == value.strip()
        and all(
            character.isascii() and (character.isalnum() or character in "._+-")
            for character in value
        )
    )


def canonical_ollama_model_name(model: str) -> str:
    """Return one explicit local Ollama model identity, including its tag."""

    if not isinstance(model, str):
        raise ValueError("Ollama model name is invalid")
    name = model.strip()
    if (
        not name
        or len(name) > 240
        or name.count(":") > 1
        or name[0] in "./:-"
        or name[-1] in "/:"
        or any(
            not character.isascii() or not (character.isalnum() or character in "._/-:")
            for character in name
        )
    ):
        raise ValueError("Ollama model name is invalid")
    return name if ":" in name else f"{name}:latest"


@dataclass(frozen=True)
class OllamaModelIdentity:
    """Exact local model identity resolved from Ollama inventory and metadata."""

    canonical_model: str
    digest: str
    metadata_sha256: str
    parameter_count: int
    parameter_size: str

    @property
    def parameters_b(self) -> float:
        return self.parameter_count / 1_000_000_000.0

    def validated(self) -> OllamaModelIdentity:
        if (
            canonical_ollama_model_name(self.canonical_model) != self.canonical_model
            or not _is_sha256(self.digest)
            or not _is_sha256(self.metadata_sha256)
            or isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, int)
            or self.parameter_count <= 0
            or not _is_parameter_size_label(self.parameter_size)
        ):
            raise ValueError("Ollama model identity is invalid")
        return self


__all__ = ["OllamaModelIdentity", "canonical_ollama_model_name"]
