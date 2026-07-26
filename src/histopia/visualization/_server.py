"""Dependency-light HTTP serving for generated Histopia viewers."""

from __future__ import annotations

import gzip
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit

_COMPRESSIBLE_SUFFIXES = frozenset(
    {
        ".css",
        ".csv",
        ".html",
        ".js",
        ".json",
        ".md",
        ".svg",
        ".txt",
        ".xml",
    }
)
_MAX_GZIP_BYTES = 16 * 1024 * 1024
_MIN_GZIP_BYTES = 512


@lru_cache(maxsize=8)
def _gzip_file(path: str, size: int, mtime_ns: int) -> bytes:
    del size, mtime_ns
    return gzip.compress(Path(path).read_bytes(), compresslevel=6, mtime=0)


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _redirect_root(self) -> bool:
        if self.path.split("?", 1)[0] != "/":
            return False
        self.send_response(302)
        self.send_header("Location", "/histopia/")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def _reject_hidden_path(self) -> bool:
        path = unquote(urlsplit(self.path).path)
        if not any(part.startswith(".") for part in Path(path).parts):
            return False
        self.send_error(404, "File not found")
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._reject_hidden_path() and not self._redirect_root():
            super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._reject_hidden_path() and not self._redirect_root():
            super().do_HEAD()

    def send_head(self):  # type: ignore[no-untyped-def]
        path = Path(self.translate_path(self.path))
        if (
            path.is_file()
            and path.suffix.lower() in _COMPRESSIBLE_SUFFIXES
            and _MIN_GZIP_BYTES <= path.stat().st_size <= _MAX_GZIP_BYTES
            and _accepts_gzip(self.headers.get("Accept-Encoding", ""))
        ):
            return self._send_gzip_head(path)
        return super().send_head()

    def copyfile(self, source, outputfile) -> None:  # type: ignore[no-untyped-def]
        """Ignore expected disconnects when a browser cancels stale textures."""

        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_gzip_head(self, path: Path) -> BytesIO | None:
        stat = path.stat()
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}-gzip"'
        if _etag_matches(self.headers.get("If-None-Match", ""), etag):
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return None
        payload = _gzip_file(str(path), stat.st_size, stat.st_mtime_ns)
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.send_header("ETag", etag)
        self.end_headers()
        return BytesIO(payload)

    def list_directory(self, path: str):  # type: ignore[no-untyped-def]
        del path
        self.send_error(404, "File not found")
        return None

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        suffix = Path(unquote(urlsplit(self.path).path)).suffix.lower()
        if suffix in _COMPRESSIBLE_SUFFIXES:
            self.send_header("Vary", "Accept-Encoding")
        super().end_headers()


def _accepts_gzip(value: str) -> bool:
    wildcard_quality: float | None = None
    for item in value.split(","):
        parts = [part.strip() for part in item.split(";")]
        encoding = parts[0].lower()
        quality = 1.0
        for parameter in parts[1:]:
            if parameter.lower().startswith("q="):
                try:
                    quality = float(parameter[2:])
                except ValueError:
                    quality = 0.0
        if encoding == "gzip":
            return quality > 0
        if encoding == "*":
            wildcard_quality = quality
    return wildcard_quality is not None and wildcard_quality > 0


def _etag_matches(value: str, expected: str) -> bool:
    return any(
        candidate.removeprefix("W/") in {expected, "*"}
        for candidate in (item.strip() for item in value.split(","))
    )


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
