"""Dependency-light HTTP serving for generated Histopia viewers."""

from __future__ import annotations

import gzip
import hmac
import json
import os
import re
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from histopia.visualization._review_api import ReviewDecisionService

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
_MAX_API_BODY_BYTES = 16 * 1024
_ROUTE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@lru_cache(maxsize=8)
def _gzip_file(path: str, size: int, mtime_ns: int) -> bytes:
    del size, mtime_ns
    return gzip.compress(Path(path).read_bytes(), compresslevel=6, mtime=0)


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def _required_routes(self) -> tuple[str, ...]:
        return self.server.required_routes  # type: ignore[attr-defined,no-any-return]

    @property
    def _review_service(self) -> ReviewDecisionService | None:
        return self.server.review_service  # type: ignore[attr-defined,no-any-return]

    @property
    def _review_token(self) -> str | None:
        return self.server.review_token  # type: ignore[attr-defined,no-any-return]

    def _redirect_root(self) -> bool:
        path = urlsplit(self.path).path
        if path == "/":
            target = "/histopia/"
        elif path.removeprefix("/") in self._required_routes:
            target = f"{path}/"
        else:
            return False
        self.send_response(302)
        self.send_header("Location", target)
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
        if self._serve_health(head=False) or self._serve_review_api():
            return
        if not self._reject_hidden_path() and not self._redirect_root():
            super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self._serve_health(head=True):
            return
        if not self._reject_hidden_path() and not self._redirect_root():
            super().do_HEAD()

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path not in {"/api/reviews/approve", "/api/reviews/feedback"}:
            self.send_error(404, "File not found")
            return
        if not self._api_authorized():
            return
        length = self.headers.get("Content-Length")
        try:
            size = int(length or "")
        except ValueError:
            size = -1
        if size < 1 or size > _MAX_API_BODY_BYTES:
            self._send_json(400, {"error": "invalid request size"})
            return
        try:
            payload = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "request body must be JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "request body must be an object"})
            return
        assert self._review_service is not None
        try:
            if path == "/api/reviews/approve":
                result = {
                    "cohort": self._review_service.approve(payload),
                }
            else:
                result = {
                    "feedback": self._review_service.save_feedback(payload),
                }
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _serve_health(self, *, head: bool) -> bool:
        if urlsplit(self.path).path != "/healthz":
            return False
        root = Path(self.directory)
        routes = {
            route: (root / route / "index.html").is_file()
            for route in self._required_routes
        }
        payload = {
            "status": "ok" if all(routes.values()) else "degraded",
            "routes": {f"/{route}/": ready for route, ready in routes.items()},
            "review_api": self._review_service is not None,
        }
        self._send_json(200 if all(routes.values()) else 503, payload, head=head)
        return True

    def _serve_review_api(self) -> bool:
        parsed = urlsplit(self.path)
        if parsed.path not in {
            "/api/reviews",
            "/api/reviews/feedback",
            "/api/reviews/feedback-summary",
        }:
            return False
        if not self._api_authorized():
            return True
        assert self._review_service is not None
        try:
            if parsed.path == "/api/reviews":
                payload = self._review_service.status()
            elif parsed.path == "/api/reviews/feedback-summary":
                payload = self._review_service.feedback_summary()
            else:
                query = parse_qs(parsed.query)
                cohort = _single_query_value(query, "cohort")
                stage = _single_query_value(query, "stage")
                payload = self._review_service.feedback(cohort, stage)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            self._send_json(409, {"error": str(exc)})
            return True
        self._send_json(200, payload)
        return True

    def _api_authorized(self) -> bool:
        if self._review_service is None or self._review_token is None:
            self._send_json(404, {"error": "review decisions are not configured"})
            return False
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if origin and (not host or urlsplit(origin).netloc != host):
            self._send_json(403, {"error": "cross-origin review requests are denied"})
            return False
        supplied = self.headers.get("Authorization", "")
        if not supplied.startswith("Bearer ") or not hmac.compare_digest(
            supplied[7:],
            self._review_token,
        ):
            self._send_json(401, {"error": "invalid review access key"})
            return False
        return True

    def _send_json(
        self,
        status: int,
        payload: dict[str, object],
        *,
        head: bool = False,
    ) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(encoded)

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
        if not self._headers_buffer_contains(b"Cache-Control:"):
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        suffix = Path(unquote(urlsplit(self.path).path)).suffix.lower()
        if suffix in _COMPRESSIBLE_SUFFIXES:
            self.send_header("Vary", "Accept-Encoding")
        super().end_headers()

    def _headers_buffer_contains(self, prefix: bytes) -> bool:
        return any(line.startswith(prefix) for line in self._headers_buffer)


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


def _single_query_value(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if not values or len(values) != 1 or not values[0]:
        raise ValueError(f"review query requires one {name}")
    return values[0]


def create_viewer_server(
    root: Path | str,
    *,
    bind: str = "0.0.0.0",
    port: int = 8765,
    required_routes: tuple[str, ...] = ("histopia",),
    review_config: Path | str | None = None,
    review_token: str | None = None,
) -> ThreadingHTTPServer:
    """Create a server rooted above stable viewer endpoints."""

    root = Path(root).expanduser().resolve()
    if not required_routes or any(
        not _ROUTE_RE.fullmatch(route) for route in required_routes
    ):
        raise ValueError("required routes must be simple non-empty path names")
    required_routes = tuple(dict.fromkeys(required_routes))
    for route in required_routes:
        stable_index = root / route / "index.html"
        if not stable_index.is_file():
            raise FileNotFoundError(f"viewer root is missing {stable_index}")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    review_service = (
        ReviewDecisionService.from_file(review_config)
        if review_config is not None
        else None
    )
    if review_service is not None:
        review_token = (
            review_token or os.environ.get("HISTOPIA_REVIEW_TOKEN", "")
        ).strip()
        if len(review_token) < 24:
            raise ValueError("review access token must contain at least 24 characters")
    else:
        review_token = None
    handler = partial(_ViewerRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((bind, port), handler)
    server.daemon_threads = True
    server.required_routes = required_routes  # type: ignore[attr-defined]
    server.review_service = review_service  # type: ignore[attr-defined]
    server.review_token = review_token  # type: ignore[attr-defined]
    return server


def serve_viewer(
    root: Path | str,
    *,
    bind: str = "0.0.0.0",
    port: int = 8765,
    required_routes: tuple[str, ...] = ("histopia",),
    review_config: Path | str | None = None,
    review_token: str | None = None,
) -> None:
    """Serve a generated viewer until interrupted."""

    server = create_viewer_server(
        root,
        bind=bind,
        port=port,
        required_routes=required_routes,
        review_config=review_config,
        review_token=review_token,
    )
    endpoints = ", ".join(
        f"http://{bind}:{server.server_port}/{route}/" for route in required_routes
    )
    print(f"Histopia viewers: {endpoints}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
