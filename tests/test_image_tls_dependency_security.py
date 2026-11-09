"""Socket-free checks for the actual TLS bundle and optional image consumer."""

from importlib import metadata
from io import BytesIO
import ssl

import certifi
import pytest


def test_compromised_root_is_absent_and_tls_verification_remains_enabled() -> None:
    assert tuple(map(int, metadata.version("certifi").split("."))) >= (2026, 10, 19)
    context = ssl.create_default_context(cafile=certifi.where())
    roots = context.get_ca_certs()
    assert roots
    common_names = {
        value
        for certificate in roots
        for group in certificate["subject"]
        for key, value in group
        if key == "commonName"
    }
    assert "emSign ECC Root CA - G3" not in common_names
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "TIFF", "GIF"])
def test_optional_patched_image_codecs_preserve_ordinary_roundtrips(
    format_name,
) -> None:
    image = pytest.importorskip("PIL.Image")
    assert tuple(map(int, metadata.version("pillow").split("."))) >= (13, 0, 0)
    source = image.new("RGB", (8, 8), (30, 90, 150))
    encoded = BytesIO()
    source.save(encoded, format=format_name)
    encoded.seek(0)
    with image.open(encoded) as decoded:
        decoded.load()
        assert decoded.size == (8, 8)
        assert decoded.convert("RGB").getpixel((0, 0)) == pytest.approx(
            (30, 90, 150), abs=2 if format_name == "JPEG" else 0
        )


def test_optional_patched_multiframe_consumer_keeps_frames_independent() -> None:
    image = pytest.importorskip("PIL.Image")
    encoded = BytesIO()
    first = image.new("RGB", (8, 8), (255, 0, 0))
    second = image.new("RGB", (8, 8), (0, 255, 0))
    first.save(encoded, format="GIF", save_all=True, append_images=[second])
    encoded.seek(0)
    with image.open(encoded) as decoded:
        assert decoded.n_frames == 2
        decoded.seek(0)
        assert decoded.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        decoded.seek(1)
        assert decoded.convert("RGB").getpixel((0, 0)) == (0, 255, 0)
