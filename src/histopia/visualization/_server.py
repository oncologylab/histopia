"""Dependency-light HTTP serving for generated Histopia viewers."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    def _redirect_root(self) -> bool:
        if self.path.split("?", 1)[0] != "/":
            return False
        self.send_response(302)
        self.send_header("Location", "/histopia/")
        self.end_headers()
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._redirect_root():
            super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._redirect_root():
            super().do_HEAD()


def create_viewer_server(
    root: Path | str,
    *,
    bind: str = "0.0.0.0",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a server rooted above the stable ``histopia/`` endpoint."""

    root = Path(root).expanduser().resolve()
    stable_index = root / "histopia" / "index.html"
    if not stable_index.is_file():
        raise FileNotFoundError(f"viewer root is missing {stable_index}")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    handler = partial(_ViewerRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((bind, port), handler)
    server.daemon_threads = True
    return server


def serve_viewer(
    root: Path | str,
    *,
    bind: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    """Serve a generated viewer until interrupted."""

    server = create_viewer_server(root, bind=bind, port=port)
    print(f"Histopia viewer: http://{bind}:{server.server_port}/histopia/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
