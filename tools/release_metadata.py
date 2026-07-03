#!/usr/bin/env python3
"""Resolve Simple AI Trading beta release metadata for GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-((?:0|[1-9][0-9]*|[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[A-Za-z-][0-9A-Za-z-]*))*))?$"
)
PEP440_BETA_PATTERN = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)b([0-9]+)$")


@dataclass(frozen=True)
class ReleaseMetadata:
    release_version: str
    release_tag: str
    package_version: str
    package_base: str


def pep440_to_beta_semver(value: str) -> str:
    match = PEP440_BETA_PATTERN.fullmatch(value.strip())
    if match is None:
        return value.strip()
    return f"{match.group(1)}.{match.group(2)}.{match.group(3)}-beta.{match.group(4)}"


def validate_beta_version(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("release version must not be empty")
    if stripped.lower() == "latest" or stripped.lower().startswith("v"):
        raise ValueError("release version must not be latest and must not use a v prefix")
    normalized = pep440_to_beta_semver(stripped)
    match = SEMVER_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("release version must be SemVer, for example 0.1.0-beta.1")
    prerelease = match.group(4)
    if prerelease is None or not prerelease.lower().startswith("beta"):
        raise ValueError("this workflow only publishes beta prereleases such as 0.1.0-beta.1")
    return normalized


def read_project_version(repo_root: Path) -> str:
    pyproject = repo_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not isinstance(version, str):
        raise ValueError("could not resolve project.version from pyproject.toml")
    return validate_beta_version(version)


def read_existing_tags(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def resolve_metadata(
    *,
    requested_version: str,
    repo_root: Path,
    existing_tags: set[str],
    allow_existing: bool = False,
) -> ReleaseMetadata:
    release_version = (
        validate_beta_version(requested_version) if requested_version.strip() else read_project_version(repo_root)
    )
    release_tag = f"v{release_version}"
    if not allow_existing and release_tag in existing_tags:
        raise ValueError(
            f"release tag {release_tag} already exists; use replace_existing=true with an explicit release_version"
        )
    package_base = f"SimpleAITrading-{release_version}-win64-beta"
    return ReleaseMetadata(
        release_version=release_version,
        release_tag=release_tag,
        package_version=release_version,
        package_base=package_base,
    )


def write_github_env(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as env_file:
        for key, value in values.items():
            env_file.write(f"{key}={value}\n")


def write_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        for key, value in values.items():
            output_file.write(f"{key}={value}\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve Simple AI Trading beta release metadata.")
    parser.add_argument("--requested-version", default="")
    parser.add_argument("--existing-tags-file", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        metadata = resolve_metadata(
            requested_version=args.requested_version,
            repo_root=args.repo_root,
            existing_tags=read_existing_tags(args.existing_tags_file),
            allow_existing=args.allow_existing,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    values = {
        "RELEASE_VERSION": metadata.release_version,
        "RELEASE_TAG": metadata.release_tag,
        "PACKAGE_VERSION": metadata.package_version,
        "PACKAGE_BASE": metadata.package_base,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "GITHUB_ENV" in os.environ:
        write_github_env(Path(os.environ["GITHUB_ENV"]), values)
    if "GITHUB_OUTPUT" in os.environ:
        write_github_output(Path(os.environ["GITHUB_OUTPUT"]), {key.lower(): value for key, value in values.items()})

    print(f"Release version: {metadata.release_version}")
    print(f"Release tag: {metadata.release_tag}")
    print(f"Package base: {metadata.package_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
