from __future__ import annotations

import gzip
import http.client
import io
import threading
from pathlib import Path

import pytest

from histopia.visualization._server import (
    _ViewerRequestHandler,
    create_viewer_server,
)


@pytest.mark.parametrize("error", (BrokenPipeError, ConnectionResetError))
def test_server_ignores_expected_cancelled_texture_writes(error: type[OSError]) -> None:
    class CancelledOutput:
        def write(self, _payload: bytes) -> None:
            raise error

    handler = object.__new__(_ViewerRequestHandler)

    handler.copyfile(io.BytesIO(b"cancelled texture"), CancelledOutput())


def test_server_redirects_root_to_stable_endpoint(tmp_path: Path) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable viewer")
    server = create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/")
        redirect = connection.getresponse()
        assert redirect.status == 302
        assert redirect.version == 11
        assert redirect.getheader("Location") == "/histopia/"
        assert redirect.getheader("Content-Length") == "0"
        redirect.read()

        connection.request("GET", "/histopia/")
        response = connection.getresponse()
        assert response.status == 200
        assert response.version == 11
        assert response.getheader("Cache-Control") == (
            "public, max-age=0, must-revalidate"
        )
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert response.read() == b"stable viewer"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_rejects_missing_stable_viewer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="histopia/index.html"):
        create_viewer_server(tmp_path, bind="127.0.0.1", port=0)


def test_server_compresses_and_revalidates_json(tmp_path: Path) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable viewer")
    payload = b'{"sections":[' + b'"registered",' * 200 + b"]}"
    (stable / "manifest.json").write_bytes(payload)
    server = create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request(
            "HEAD",
            "/histopia/manifest.json",
            headers={"Accept-Encoding": "gzip"},
        )
        head = connection.getresponse()
        assert head.status == 200
        assert head.getheader("Content-Encoding") == "gzip"
        assert int(head.getheader("Content-Length", "0")) < len(payload)
        assert head.read() == b""

        connection.request(
            "GET",
            "/histopia/manifest.json",
            headers={"Accept-Encoding": "gzip"},
        )
        response = connection.getresponse()
        encoded = response.read()
        etag = response.getheader("ETag")

        assert response.status == 200
        assert response.version == 11
        assert response.getheader("Content-Encoding") == "gzip"
        assert response.getheader("Vary") == "Accept-Encoding"
        assert int(response.getheader("Content-Length", "0")) == len(encoded)
        assert etag is not None
        assert gzip.decompress(encoded) == payload

        connection.request(
            "GET",
            "/histopia/manifest.json",
            headers={
                "Accept-Encoding": "gzip",
                "If-None-Match": etag,
            },
        )
        cached = connection.getresponse()
        assert cached.status == 304
        assert cached.getheader("ETag") == etag
        assert cached.read() == b""
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_respects_gzip_quality_zero(tmp_path: Path) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable viewer")
    payload = b"x" * 1_000
    (stable / "viewer.js").write_bytes(payload)
    server = create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request(
            "GET",
            "/histopia/viewer.js",
            headers={"Accept-Encoding": "gzip;q=0, identity"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Encoding") is None
        assert response.read() == payload
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "request_path",
    (
        "/histopia/.histopia-asset-cache.json",
        "/histopia/%2Ehistopia-asset-cache.json",
    ),
)
def test_server_rejects_hidden_files(tmp_path: Path, request_path: str) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable viewer")
    (stable / ".histopia-asset-cache.json").write_text("private cache metadata")
    server = create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", request_path)
        response = connection.getresponse()
        assert response.status == 404
        assert b"private cache metadata" not in response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_disables_directory_listing(tmp_path: Path) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable viewer")
    assets = stable / "assets"
    assets.mkdir()
    (assets / "section.webp").write_bytes(b"image")
    server = create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/histopia/assets/")
        response = connection.getresponse()
        assert response.status == 404
        assert b"section.webp" not in response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
