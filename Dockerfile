# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# ----------------------------------------------------------------------------
# Stage 1: builder â€” installs the package into a throwaway venv.
# ----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_BUILD_PACKAGE=numpy

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python3

# ----------------------------------------------------------------------------
# Stage 2: runtime â€” minimal, non-root, read-only friendly.
# ----------------------------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/simple-ai-trading/numba

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 trader \
    && useradd --system --uid 10001 --gid trader --home-dir /home/trader --create-home trader \
    && mkdir -p /home/trader/.config /home/trader/work/data \
    && chown -R trader:trader /home/trader/.config /home/trader/work

COPY --from=builder /opt/venv /opt/venv

USER trader
WORKDIR /home/trader/work

# Config + data are mounted by the operator (or docker-compose). The package
# itself ships no runtime state; it persists runtime + strategy JSON under
# ``~/.config/simple_ai_trading`` and relative data files under
# ``/home/trader/work/data`` inside the container.
VOLUME ["/home/trader/.config", "/home/trader/work/data"]

# Default to the interactive shell so ``docker run -it image`` feels natural.
ENTRYPOINT ["simple-ai-trading"]
CMD ["shell"]
