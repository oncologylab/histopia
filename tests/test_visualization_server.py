from __future__ import annotations

import gzip
import http.client
import io
import json
import threading
from pathlib import Path

import pytest

from histopia.visualization._server import (
    _ViewerRequestHandler,
    create_viewer_server,
)
from histopia.visualization._wsi_tiles import WsiTileService


@pytest.mark.parametrize("error", (BrokenPipeError, ConnectionResetError))
def test_server_ignores_expected_cancelled_texture_writes(error: type[OSError]) -> None:
    class CancelledOutput:
        def write(self, _payload: bytes) -> None:
            raise error

    handler = object.__new__(_ViewerRequestHandler)

    handler.copyfile(io.BytesIO(b"cancelled texture"), CancelledOutput())

    handler.wfile = CancelledOutput()
    handler.close_connection = False
    handler._write_body(b"cancelled API tile")
    assert handler.close_connection is True


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
        assert redirect.getheader("Location") == "histopia/"
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


def test_server_requires_and_reports_all_stable_routes(tmp_path: Path) -> None:
    for route in ("histopia", "review"):
        directory = tmp_path / route
        directory.mkdir()
        (directory / "index.html").write_text(route)
    server = create_viewer_server(
        tmp_path,
        bind="127.0.0.1",
        port=0,
        required_routes=("histopia", "review"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/review")
        redirect = connection.getresponse()
        assert redirect.status == 302
        assert redirect.getheader("Location") == "review/"
        redirect.read()

        connection.request("GET", "/review?view=cells")
        redirect = connection.getresponse()
        assert redirect.status == 302
        assert redirect.getheader("Location") == "review/?view=cells"
        redirect.read()

        connection.request("GET", "/healthz")
        health = connection.getresponse()
        assert health.status == 200
        assert health.getheader("Cache-Control") == "no-store"
        assert json.loads(health.read()) == {
            "status": "ok",
            "routes": {"/histopia/": True, "/review/": True},
            "review_api": False,
            "wsi_api": False,
        }

        connection.request("GET", "/api/wsi/mouse")
        catalog = connection.getresponse()
        assert catalog.status == 200
        assert json.loads(catalog.read()) == {
            "schema_version": 1,
            "cohort": "mouse",
            "sections": [],
        }
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_rejects_missing_required_review_route(tmp_path: Path) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable viewer")

    with pytest.raises(FileNotFoundError, match="review/index.html"):
        create_viewer_server(
            tmp_path,
            bind="127.0.0.1",
            port=0,
            required_routes=("histopia", "review"),
        )


def test_review_api_requires_key_and_same_origin(tmp_path: Path) -> None:
    for route in ("histopia", "review"):
        directory = tmp_path / route
        directory.mkdir()
        (directory / "index.html").write_text(route)
    run = tmp_path / "run"
    run.mkdir()
    (run / "mask_review.json").write_text(
        json.dumps({"slides": [{"status": "pending"}]})
    )
    config = tmp_path / "review-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {"mouse": {"registration": str(run)}},
            }
        )
    )
    token = "a-secure-test-review-token"
    server = create_viewer_server(
        tmp_path,
        bind="127.0.0.1",
        port=0,
        required_routes=("histopia", "review"),
        review_config=config,
        review_token=token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/reviews/access")
        access = connection.getresponse()
        assert access.status == 200
        assert json.loads(access.read()) == {
            "review_configured": True,
            "authentication_required": True,
        }

        connection.request("GET", "/api/reviews")
        denied = connection.getresponse()
        assert denied.status == 401
        denied.read()

        connection.request(
            "GET",
            "/api/reviews",
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "https://example.invalid",
            },
        )
        cross_origin = connection.getresponse()
        assert cross_origin.status == 403
        cross_origin.read()

        connection.request(
            "GET",
            "/api/reviews",
            headers={"Authorization": f"Bearer {token}"},
        )
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        assert payload["feedback_configured"] is False
        assert payload["cohorts"][0]["id"] == "mouse"
        assert payload["cohorts"][0]["stages"]["mask"] == {
            "available": True,
            "approved": False,
        }
        assert str(tmp_path) not in json.dumps(payload)
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_review_api_allows_explicit_public_same_origin_access(
    tmp_path: Path,
) -> None:
    for route in ("histopia", "review"):
        directory = tmp_path / route
        directory.mkdir()
        (directory / "index.html").write_text(route)
    run = tmp_path / "run"
    run.mkdir()
    (run / "mask_review.json").write_text(
        json.dumps({"slides": [{"status": "pending"}]})
    )
    config = tmp_path / "review-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {"mouse": {"registration": str(run)}},
            }
        )
    )
    server = create_viewer_server(
        tmp_path,
        bind="127.0.0.1",
        port=0,
        required_routes=("histopia", "review"),
        review_config=config,
        public_review_write=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/reviews/access")
        access = connection.getresponse()
        assert access.status == 200
        assert json.loads(access.read()) == {
            "review_configured": True,
            "authentication_required": False,
        }

        connection.request("GET", "/api/reviews")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["cohorts"][0]["id"] == "mouse"

        connection.request(
            "GET",
            "/api/reviews",
            headers={"Origin": "https://example.invalid"},
        )
        cross_origin = connection.getresponse()
        assert cross_origin.status == 403
        cross_origin.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_public_review_mode_requires_review_configuration(tmp_path: Path) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable")

    with pytest.raises(
        ValueError,
        match="public review writes require a review configuration",
    ):
        create_viewer_server(
            tmp_path,
            bind="127.0.0.1",
            port=0,
            public_review_write=True,
        )


def test_server_serves_fingerprinted_wsi_metadata_and_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable")
    run = tmp_path / "run"
    registered = tmp_path / "registered"
    run.mkdir()
    registered.mkdir()
    config = tmp_path / "review-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {
                    "mouse": {
                        "registration": str(run),
                        "registered_wsi": str(registered),
                    }
                },
            }
        )
    )
    digest = "a" * 64

    class FakeTiles:
        def catalog(self, cohort: str) -> dict[str, object]:
            if cohort != "mouse":
                raise FileNotFoundError("unknown WSI cohort")
            return {
                "schema_version": 1,
                "cohort": cohort,
                "sections": [{"section": "001"}],
            }

        def metadata(
            self,
            cohort: str,
            section: str,
            protein_model: str | None = None,
        ) -> dict[str, object]:
            assert (cohort, section) == ("mouse", "001")
            return {
                "schema_version": 1,
                "cohort": cohort,
                "section": section,
                "protein_model_id": protein_model,
            }

        def render_tile(self, *args, protein_model: str | None = None):
            if protein_model is not None:
                assert protein_model == "yap-extra-trees"
                assert args == (
                    "mouse",
                    "001",
                    "protein_predicted",
                    digest,
                    0,
                    0,
                    0,
                )
                return b"png-model", "image/png", f'"{digest}-0-0-0"'
            if args[2] == "registered":
                assert args == ("mouse", "001", "registered", digest, 0, 0, 0)
                return b"jpeg-tile", "image/jpeg", f'"{digest}-0-0-0"'
            assert args in {
                ("mouse", "001", "stain_raw", digest, 0, 0, 0),
                ("mouse", "001", "stain_adaptive", digest, 0, 0, 0),
                ("mouse", "001", "stain_adaptive_v3", digest, 0, 0, 0),
                ("mouse", "001", "stain_adaptive_v3_map", digest, 0, 0, 0),
            }
            return b"png-tile", "image/png", f'"{digest}-0-0-0"'

    monkeypatch.setattr(
        WsiTileService,
        "from_runs",
        classmethod(lambda cls, runs: FakeTiles()),
    )
    server = create_viewer_server(
        tmp_path,
        bind="127.0.0.1",
        port=0,
        review_config=config,
        public_review_write=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/wsi/mouse")
        catalog = connection.getresponse()
        assert catalog.status == 200
        assert json.loads(catalog.read())["sections"] == [{"section": "001"}]

        connection.request("GET", "/api/wsi/mouse/001")
        metadata = connection.getresponse()
        assert metadata.status == 200
        assert json.loads(metadata.read())["section"] == "001"

        connection.request(
            "GET",
            "/api/wsi/mouse/001/protein/yap-extra-trees",
        )
        model_metadata = connection.getresponse()
        assert model_metadata.status == 200
        assert json.loads(model_metadata.read())["protein_model_id"] == (
            "yap-extra-trees"
        )

        connection.request("HEAD", "/api/wsi/mouse/001")
        metadata_head = connection.getresponse()
        assert metadata_head.status == 200
        assert metadata_head.getheader("Content-Type") == "application/json"
        assert int(metadata_head.getheader("Content-Length", "0")) > 0
        assert metadata_head.read() == b""

        connection.request("GET", "/api/wsi/unknown")
        missing_catalog = connection.getresponse()
        assert missing_catalog.status == 200
        assert json.loads(missing_catalog.read())["sections"] == []

        tile_path = f"/api/wsi/mouse/001/registered/{digest}/0/0/0.jpg"
        connection.request("GET", tile_path)
        tile = connection.getresponse()
        assert tile.status == 200
        assert tile.getheader("Content-Type") == "image/jpeg"
        assert tile.getheader("Cache-Control") == (
            "public, max-age=31536000, immutable"
        )
        etag = tile.getheader("ETag")
        assert tile.read() == b"jpeg-tile"

        dzi_path = f"/api/wsi/mouse/001/registered/{digest}/dzi/8/8/0_0.jpg"
        connection.request("GET", dzi_path)
        dzi_tile = connection.getresponse()
        assert dzi_tile.status == 200
        assert dzi_tile.read() == b"jpeg-tile"

        stain_path = f"/api/wsi/mouse/001/stain_raw/{digest}/dzi/0/0/0_0.png"
        connection.request("GET", stain_path)
        stain_tile = connection.getresponse()
        assert stain_tile.status == 200
        assert stain_tile.getheader("Content-Type") == "image/png"
        assert stain_tile.getheader("ETag") == f'"{digest}-0-0-0"'
        assert stain_tile.getheader("Cache-Control") == (
            "public, max-age=31536000, immutable"
        )
        assert stain_tile.read() == b"png-tile"

        adaptive_path = f"/api/wsi/mouse/001/stain_adaptive/{digest}/dzi/0/0/0_0.png"
        connection.request("GET", adaptive_path)
        adaptive_tile = connection.getresponse()
        assert adaptive_tile.status == 200
        assert adaptive_tile.getheader("Content-Type") == "image/png"
        assert adaptive_tile.read() == b"png-tile"

        for layer in ("stain_adaptive_v3", "stain_adaptive_v3_map"):
            v3_path = f"/api/wsi/mouse/001/{layer}/{digest}/dzi/0/0/0_0.png"
            connection.request("GET", v3_path)
            v3_tile = connection.getresponse()
            assert v3_tile.status == 200
            assert v3_tile.getheader("Content-Type") == "image/png"
            assert v3_tile.read() == b"png-tile"

        model_path = (
            f"/api/wsi/mouse/001/protein/yap-extra-trees/"
            f"protein_predicted/{digest}/dzi/0/0/0_0.png"
        )
        connection.request("GET", model_path)
        model_tile = connection.getresponse()
        assert model_tile.status == 200
        assert model_tile.getheader("Content-Type") == "image/png"
        assert model_tile.read() == b"png-model"

        connection.request("HEAD", stain_path)
        stain_head = connection.getresponse()
        assert stain_head.status == 200
        assert stain_head.getheader("Content-Type") == "image/png"
        assert stain_head.getheader("Content-Length") == str(len(b"png-tile"))
        assert stain_head.getheader("ETag") == f'"{digest}-0-0-0"'
        assert stain_head.getheader("Cache-Control") == (
            "public, max-age=31536000, immutable"
        )
        assert stain_head.read() == b""

        connection.request(
            "GET",
            f"/api/wsi/mouse/001/stain_raw/{digest}/dzi/0/0/0_0.jpg",
        )
        malformed_stain = connection.getresponse()
        assert malformed_stain.status == 404
        malformed_stain.read()

        connection.request("GET", tile_path, headers={"If-None-Match": etag})
        cached = connection.getresponse()
        assert cached.status == 304
        assert cached.read() == b""

        connection.request("GET", f"{tile_path}/../../etc/passwd")
        rejected = connection.getresponse()
        assert rejected.status == 404
        rejected.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_passes_optional_stain_runs_to_tile_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("stable")
    registration = tmp_path / "registration"
    stain = tmp_path / "stain"
    registration.mkdir()
    stain.mkdir()
    config = tmp_path / "review-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {
                    "mouse": {
                        "registration": str(registration),
                        "stain": str(stain),
                    }
                },
            }
        )
    )
    captured: dict[str, object] = {}

    class FakeTiles:
        pass

    def fake_from_runs(cls, runs, **options):
        captured["runs"] = runs
        captured["options"] = options
        return FakeTiles()

    monkeypatch.setattr(WsiTileService, "from_runs", classmethod(fake_from_runs))
    server = create_viewer_server(
        tmp_path,
        bind="127.0.0.1",
        port=0,
        review_config=config,
        public_review_write=True,
    )
    try:
        assert captured == {
            "runs": {"mouse": (registration, None)},
            "options": {"stain_runs": {"mouse": stain}},
        }
    finally:
        server.server_close()


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
