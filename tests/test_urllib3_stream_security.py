"""Offline bounded regression for GHSA-gh4c-6fx4-qh6g; never opens a socket."""

import subprocess
import sys
import zlib
import pytest
from urllib3.response import DeflateDecoder


@pytest.mark.parametrize(
    "wbits", [zlib.MAX_WBITS, -zlib.MAX_WBITS], ids=["zlib", "raw-deflate"]
)
@pytest.mark.parametrize("trailing", [b"", b"trailing"], ids=["valid", "trailing"])
def test_decoder_makes_progress(wbits, trailing):
    payload = b"bounded-test" * 20
    encoder = zlib.compressobj(wbits=wbits)
    compressed = encoder.compress(payload) + encoder.flush() + trailing
    decoder = DeflateDecoder()
    chunks = [decoder.decompress(compressed, max_length=16)]
    for _ in range(len(payload) // 16 + 4):
        if not decoder.has_unconsumed_tail:
            break
        chunks.append(decoder.decompress(b"", max_length=16))
    assert not decoder.has_unconsumed_tail, (
        "decoder repeats trailing bytes without progress"
    )
    assert b"".join(chunks) + decoder.flush() == payload


@pytest.mark.parametrize(
    "wbits", [zlib.MAX_WBITS, -zlib.MAX_WBITS], ids=["zlib", "raw-deflate"]
)
@pytest.mark.parametrize("trailing", [False, True], ids=["valid", "trailing"])
def test_requests_chunked_deflate_completes_in_bounded_child(wbits, trailing):
    program = r"""
import io, http.client, zlib
import requests
from urllib3.response import HTTPResponse
payload = b"bounded-test" * 20
encoder = zlib.compressobj(wbits=WBITS)
data = encoder.compress(payload) + encoder.flush() + (b"trailing" if TRAILING else b"")
wire = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Encoding: deflate\r\n\r\n" + format(len(data),"x").encode() + b"\r\n" + data + b"\r\n0\r\n\r\n"
class Socket:
    def makefile(self, mode):
        return io.BytesIO(wire)
original = http.client.HTTPResponse(Socket(), method="GET")
original.begin()
raw = HTTPResponse(body=original, original_response=original, headers=dict(original.getheaders()), preload_content=False)
response = requests.Response()
response.raw = raw
assert b"".join(response.iter_content(chunk_size=16)) == payload
response.close()
print("decoded")
""".replace("WBITS", str(wbits)).replace("TRAILING", str(trailing))
    result = subprocess.run(
        [sys.executable, "-c", program],
        timeout=5,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "decoded"
