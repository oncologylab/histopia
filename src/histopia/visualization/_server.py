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
from histopia.visualization._wsi_tiles import WsiTileCapacityError, WsiTileService

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
_MAX_ANNOTATION_BODY_BYTES = 4 * 1024 * 1024
_ROUTE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_WSI_METADATA_RE = re.compile(
    r"/api/wsi/(?P<cohort>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<section>[0-9]{3,6})"
)
_WSI_MODEL_METADATA_RE = re.compile(
    r"/api/wsi/(?P<cohort>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<section>[0-9]{3,6})/protein/"
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9_.-]*)"
)
_WSI_CATALOG_RE = re.compile(r"/api/wsi/(?P<cohort>[A-Za-z0-9][A-Za-z0-9_.-]*)")
_WSI_TILE_RE = re.compile(
    r"/api/wsi/(?P<cohort>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<section>[0-9]{3,6})/"
    r"(?P<layer>raw|registered|mask|cells|stain_raw|stain_corrected|stain_adaptive(?:_v3)?(?:_map)?|"
    r"stain_output|stain_output_map|stain_counterstain|stain_residual|"
    r"stain_support|protein_predicted(?:_contrast)?|protein_dense(?:_contrast)?|protein_probability|protein_uncertainty|"
    r"protein_measured(?:_contrast)?|protein_residual|protein_observed_target|"
    r"protein_nearest_observed_target|protein_tissue_support)/"
    r"(?P<digest>[0-9a-f]{64})/"
    r"(?P<level>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+)\.(?P<format>jpg|png)"
)
_WSI_DZI_TILE_RE = re.compile(
    r"/api/wsi/(?P<cohort>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<section>[0-9]{3,6})/"
    r"(?P<layer>raw|registered|mask|cells|stain_raw|stain_corrected|stain_adaptive(?:_v3)?(?:_map)?|"
    r"stain_output|stain_output_map|stain_counterstain|stain_residual|"
    r"stain_support|protein_predicted(?:_contrast)?|protein_dense(?:_contrast)?|protein_probability|protein_uncertainty|"
    r"protein_measured(?:_contrast)?|protein_residual|protein_observed_target|"
    r"protein_nearest_observed_target|protein_tissue_support)/"
    r"(?P<digest>[0-9a-f]{64})/dzi/(?P<min_level>[0-9]+)/"
    r"(?P<level>[0-9]+)/(?P<x>[0-9]+)_(?P<y>[0-9]+)\.(?P<format>jpg|png)"
)
_WSI_MODEL_TILE_RE = re.compile(
    r"/api/wsi/(?P<cohort>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<section>[0-9]{3,6})/protein/"
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<layer>protein_predicted(?:_contrast)?|protein_dense(?:_contrast)?|"
    r"protein_probability|protein_uncertainty|protein_measured(?:_contrast)?|"
    r"protein_residual|protein_observed_target|protein_nearest_observed_target|"
    r"protein_tissue_support|stain_raw|stain_corrected|"
    r"stain_adaptive(?:_v3)?(?:_map)?|stain_output|stain_output_map|"
    r"stain_counterstain|stain_residual|stain_support|mask|raw)/"
    r"(?P<digest>[0-9a-f]{64})/"
    r"(?P<level>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+)\.(?P<format>jpg|png)"
)
_WSI_MODEL_DZI_TILE_RE = re.compile(
    r"/api/wsi/(?P<cohort>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<section>[0-9]{3,6})/protein/"
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<layer>protein_predicted(?:_contrast)?|protein_dense(?:_contrast)?|"
    r"protein_probability|protein_uncertainty|protein_measured(?:_contrast)?|"
    r"protein_residual|protein_observed_target|protein_nearest_observed_target|"
    r"protein_tissue_support|stain_raw|stain_corrected|"
    r"stain_adaptive(?:_v3)?(?:_map)?|stain_output|stain_output_map|"
    r"stain_counterstain|stain_residual|stain_support|mask|raw)/"
    r"(?P<digest>[0-9a-f]{64})/dzi/(?P<min_level>[0-9]+)/"
    r"(?P<level>[0-9]+)/(?P<x>[0-9]+)_(?P<y>[0-9]+)\.(?P<format>jpg|png)"
)


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

    @property
    def _public_review_write(self) -> bool:
        return self.server.public_review_write  # type: ignore[attr-defined,no-any-return]

    @property
    def _wsi_tiles(self) -> WsiTileService | None:
        return self.server.wsi_tiles  # type: ignore[attr-defined,no-any-return]

    def _redirect_root(self) -> bool:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/":
            target = "histopia/"
        elif path.removeprefix("/") in self._required_routes:
            target = f"{path.removeprefix('/')}/"
        else:
            return False
        # Relative redirects preserve a port-proxy prefix stripped upstream.
        if parsed.query:
            target += f"?{parsed.query}"
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
        if (
            self._serve_health(head=False)
            or self._serve_annotation_api()
            or self._serve_review_api()
            or self._serve_wsi_api(head=False)
        ):
            return
        if not self._reject_hidden_path() and not self._redirect_root():
            super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self._serve_health(head=True) or self._serve_wsi_api(head=True):
            return
        if not self._reject_hidden_path() and not self._redirect_root():
            super().do_HEAD()

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path not in {
            "/api/annotations/section",
            "/api/reviews/approve",
            "/api/reviews/feedback",
            "/api/reviews/cell-section",
            "/api/reviews/provisional",
        }:
            self.send_error(404, "File not found")
            return
        if not self._api_authorized():
            return
        length = self.headers.get("Content-Length")
        try:
            size = int(length or "")
        except ValueError:
            size = -1
        maximum_size = (
            _MAX_ANNOTATION_BODY_BYTES
            if path == "/api/annotations/section"
            else _MAX_API_BODY_BYTES
        )
        if size < 1 or size > maximum_size:
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
            if path == "/api/annotations/section":
                result = {
                    "annotations": self._review_service.save_annotation_section(
                        payload
                    ),
                }
            elif path == "/api/reviews/approve":
                result = {
                    "cohort": self._review_service.approve(payload),
                }
            elif path == "/api/reviews/cell-section":
                result = {
                    "cell_review": self._review_service.review_cell_section(payload),
                }
            elif path == "/api/reviews/provisional":
                result = {
                    "provisional_review": (
                        self._review_service.save_provisional_feedback(payload)
                    ),
                }
            else:
                result = {
                    "feedback": self._review_service.save_feedback(payload),
                }
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _serve_annotation_api(self) -> bool:
        parsed = urlsplit(self.path)
        if parsed.path not in {
            "/api/annotations",
            "/api/annotations/section",
        }:
            return False
        if not self._api_authorized():
            return True
        assert self._review_service is not None
        try:
            query = parse_qs(parsed.query)
            cohort = _single_query_value(query, "cohort")
            if parsed.path == "/api/annotations":
                payload = self._review_service.annotation_catalog(cohort)
            else:
                section = _single_query_value(query, "section")
                payload = self._review_service.annotation_section(cohort, section)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            self._send_json(409, {"error": str(exc)})
            return True
        self._send_json(200, payload)
        return True

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
            "wsi_api": self._wsi_tiles is not None,
        }
        self._send_json(200 if all(routes.values()) else 503, payload, head=head)
        return True

    def _serve_review_api(self) -> bool:
        parsed = urlsplit(self.path)
        if parsed.path not in {
            "/api/reviews/access",
            "/api/reviews",
            "/api/reviews/feedback",
            "/api/reviews/feedback-summary",
            "/api/reviews/provisional",
        }:
            return False
        if parsed.path == "/api/reviews/access":
            if self._same_origin():
                self._send_json(
                    200,
                    {
                        "review_configured": self._review_service is not None,
                        "authentication_required": (
                            self._review_service is None
                            or not self._public_review_write
                        ),
                    },
                )
            return True
        if not self._api_authorized():
            return True
        assert self._review_service is not None
        try:
            if parsed.path == "/api/reviews":
                payload = self._review_service.status()
            elif parsed.path == "/api/reviews/feedback-summary":
                payload = self._review_service.feedback_summary()
            elif parsed.path == "/api/reviews/provisional":
                query = parse_qs(parsed.query)
                payload = self._review_service.provisional_feedback(
                    _single_query_value(query, "cohort"),
                    _single_query_value(query, "stage"),
                )
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

    def _serve_wsi_api(self, *, head: bool) -> bool:
        path = urlsplit(self.path).path
        if not path.startswith("/api/wsi/"):
            return False
        catalog_match = _WSI_CATALOG_RE.fullmatch(path)
        service = self._wsi_tiles
        if service is None:
            if catalog_match is not None:
                self._send_json(
                    200,
                    {
                        "schema_version": 1,
                        "cohort": catalog_match.group("cohort"),
                        "sections": [],
                    },
                    head=head,
                )
                return True
            self._send_json(
                404,
                {"error": "WSI tiles are not configured"},
                head=head,
            )
            return True
        if catalog_match is not None:
            try:
                payload = service.catalog(catalog_match.group("cohort"))
            except FileNotFoundError:
                payload = {
                    "schema_version": 1,
                    "cohort": catalog_match.group("cohort"),
                    "sections": [],
                }
            self._send_json(200, payload, head=head)
            return True
        model_metadata_match = _WSI_MODEL_METADATA_RE.fullmatch(path)
        if model_metadata_match is not None:
            try:
                payload = service.metadata(
                    model_metadata_match.group("cohort"),
                    model_metadata_match.group("section"),
                    protein_model=model_metadata_match.group("model"),
                )
            except FileNotFoundError as error:
                self._send_json(404, {"error": str(error)}, head=head)
                return True
            self._send_json(200, payload, head=head)
            return True
        metadata_match = _WSI_METADATA_RE.fullmatch(path)
        if metadata_match is not None:
            try:
                payload = service.metadata(
                    metadata_match.group("cohort"),
                    metadata_match.group("section"),
                )
            except FileNotFoundError as error:
                self._send_json(404, {"error": str(error)}, head=head)
                return True
            self._send_json(200, payload, head=head)
            return True
        tile_match = _WSI_TILE_RE.fullmatch(path)
        protein_model: str | None = None
        if tile_match is None:
            tile_match = _WSI_MODEL_TILE_RE.fullmatch(path)
            if tile_match is not None:
                protein_model = tile_match.group("model")
        source_level: int | None = None
        if tile_match is None:
            tile_match = _WSI_DZI_TILE_RE.fullmatch(path)
            if tile_match is None:
                tile_match = _WSI_MODEL_DZI_TILE_RE.fullmatch(path)
                if tile_match is not None:
                    protein_model = tile_match.group("model")
            if tile_match is None:
                self._send_json(
                    404,
                    {"error": "unknown WSI tile"},
                    head=head,
                )
                return True
            source_level = int(tile_match.group("level")) - int(
                tile_match.group("min_level")
            )
            if source_level < 0:
                self._send_json(
                    404,
                    {"error": "invalid WSI tile level"},
                    head=head,
                )
                return True
        layer = tile_match.group("layer")
        expected_format = "png" if layer not in {"raw", "registered"} else "jpg"
        if tile_match.group("format") != expected_format:
            self._send_json(
                404,
                {"error": "invalid WSI tile format"},
                head=head,
            )
            return True
        try:
            tile_arguments = (
                tile_match.group("cohort"),
                tile_match.group("section"),
                layer,
                tile_match.group("digest"),
                (
                    source_level
                    if source_level is not None
                    else int(tile_match.group("level"))
                ),
                int(tile_match.group("x")),
                int(tile_match.group("y")),
            )
            if protein_model is None:
                payload, media_type, etag = service.render_tile(*tile_arguments)
            else:
                payload, media_type, etag = service.render_tile(
                    *tile_arguments,
                    protein_model=protein_model,
                )
        except FileNotFoundError as error:
            self._send_json(404, {"error": str(error)}, head=head)
            return True
        except WsiTileCapacityError as error:
            self._send_json(
                503,
                {"error": str(error)},
                head=head,
                extra_headers={"Retry-After": "1"},
            )
            return True
        if _etag_matches(self.headers.get("If-None-Match", ""), etag):
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            return True
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        if not head:
            self._write_body(payload)
        return True

    def _api_authorized(self) -> bool:
        if self._review_service is None:
            self._send_json(404, {"error": "review decisions are not configured"})
            return False
        if not self._same_origin():
            return False
        if self._public_review_write:
            return True
        if self._review_token is None:
            self._send_json(404, {"error": "review decisions are not configured"})
            return False
        supplied = self.headers.get("Authorization", "")
        if not supplied.startswith("Bearer ") or not hmac.compare_digest(
            supplied[7:],
            self._review_token,
        ):
            self._send_json(401, {"error": "invalid review access key"})
            return False
        return True

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if origin and (not host or urlsplit(origin).netloc != host):
            self._send_json(403, {"error": "cross-origin review requests are denied"})
            return False
        return True

    def _send_json(
        self,
        status: int,
        payload: dict[str, object],
        *,
        head: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if not head:
            self._write_body(encoded)

    def _write_body(self, payload: bytes) -> None:
        """Ignore a browser cancelling a stale tile or API response."""

        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

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
    public_review_write: bool = False,
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
    wsi_runs = review_service.wsi_runs() if review_service is not None else {}
    cell_runs = review_service.cell_runs() if review_service is not None else {}
    stain_runs = review_service.stain_runs() if review_service is not None else {}
    protein_runs = review_service.protein_runs() if review_service is not None else {}
    protein_model_runs = (
        review_service.protein_model_runs() if review_service is not None else {}
    )
    protein_model_stain_runs = (
        review_service.protein_model_stain_runs() if review_service is not None else {}
    )
    tile_options = {}
    if cell_runs:
        tile_options["cell_runs"] = cell_runs
    if review_service is not None and review_service.cell_section_scopes():
        tile_options["cell_section_scopes"] = review_service.cell_section_scopes()
    if stain_runs:
        tile_options["stain_runs"] = stain_runs
    if protein_runs:
        tile_options["protein_runs"] = protein_runs
    if protein_model_runs:
        tile_options["protein_model_runs"] = protein_model_runs
    if protein_model_stain_runs:
        tile_options["protein_model_stain_runs"] = protein_model_stain_runs
    wsi_tiles = WsiTileService.from_runs(wsi_runs, **tile_options) if wsi_runs else None
    if public_review_write and review_service is None:
        raise ValueError("public review writes require a review configuration")
    if review_service is not None and not public_review_write:
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
    server.public_review_write = public_review_write  # type: ignore[attr-defined]
    server.wsi_tiles = wsi_tiles  # type: ignore[attr-defined]
    return server


def serve_viewer(
    root: Path | str,
    *,
    bind: str = "0.0.0.0",
    port: int = 8765,
    required_routes: tuple[str, ...] = ("histopia",),
    review_config: Path | str | None = None,
    review_token: str | None = None,
    public_review_write: bool = False,
) -> None:
    """Serve a generated viewer until interrupted."""

    server = create_viewer_server(
        root,
        bind=bind,
        port=port,
        required_routes=required_routes,
        review_config=review_config,
        review_token=review_token,
        public_review_write=public_review_write,
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
