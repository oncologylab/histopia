from __future__ import annotations

import http.client
import threading
from pathlib import Path

import pytest

from histopia.visualization._server import create_viewer_server


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
        assert redirect.getheader("Location") == "/histopia/"
        redirect.read()

        connection.request("GET", "/histopia/")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"stable viewer"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_rejects_missing_stable_viewer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="histopia/index.html"):
        create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
