"""Bounded, provenance-bound reads of public tiled TIFFs.

Only requested byte ranges are cached. An ETag is an object version identifier,
not a claim that the entire remote slide has been SHA-256 verified. Image and
TIFF dependencies are imported only when a region is decoded.
"""

from __future__ import annotations

import hashlib
import io
import operator
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.study._manifest import fingerprint


def _open(request, timeout):
    """Retry transient transport failures, never authorization or identity errors."""
    for attempt in range(4):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        time.sleep(2**attempt)


class RangeReader(io.RawIOBase):
    """Seekable HTTP reader with a byte budget and a verified on-disk cache.

    Servers must support conditional range reads and supply a strong ETag.
    A changed object uses a different cache namespace. Cached blocks are
    checked on every read, including reads after restarting a process.
    """

    def __init__(
        self,
        url: str,
        cache_dir: Path | str,
        *,
        max_bytes: int = 512 * 1024**2,
        block_size: int = 256 * 1024,
        timeout: float = 60,
    ):
        super().__init__()
        if max_bytes <= 0 or block_size <= 0:
            raise ValueError("byte budget and block size must be positive")
        self.url, self.timeout = url, timeout
        with _open(Request(url, method="HEAD"), timeout) as response:
            self.size = int(response.headers["Content-Length"])
            etag = response.headers.get("ETag")
        if not etag or etag.startswith("W/") or self.size <= 0:
            raise ValueError("range sources require a strong ETag and known size")
        self.identity = {"url": url, "etag": etag, "content_length": self.size}
        self.name = url.rsplit("/", 1)[-1]
        self.cache = Path(cache_dir) / fingerprint(self.identity)
        self.cache.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.cache / "source.json", self.identity)
        self.max_bytes, self.block_size = max_bytes, block_size
        self.downloaded_bytes = 0
        self._position = 0
        self._used: dict[int, dict] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        offset = operator.index(offset)
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self.size}
        if whence not in origins or origins[whence] + offset < 0:
            raise ValueError("invalid seek")
        self._position = origins[whence] + offset
        return self._position

    def _block(self, start: int) -> bytes:
        length = min(self.block_size, self.size - start)
        path = self.cache / f"{start}-{length}.bin"
        digest_path = path.with_suffix(".sha256")
        if path.exists() and digest_path.exists():
            value = path.read_bytes()
            digest = hashlib.sha256(value).hexdigest()
            if len(value) != length or digest != digest_path.read_text().strip():
                raise ValueError("cached range differs from its SHA-256 binding")
        else:
            if self.downloaded_bytes + length > self.max_bytes:
                raise ValueError("remote TIFF byte budget exceeded")
            request = Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{start + length - 1}",
                    "If-Match": self.identity["etag"],
                    "Accept-Encoding": "identity",
                },
            )
            with _open(request, self.timeout) as response:
                expected = f"bytes {start}-{start + length - 1}/{self.size}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected
                    or response.headers.get("ETag") != self.identity["etag"]
                ):
                    raise ValueError("server ignored range or changed source identity")
                value = response.read(length + 1)
            if len(value) != length:
                raise ValueError("incomplete or oversized HTTP range")
            self.downloaded_bytes += len(value)
            digest = hashlib.sha256(value).hexdigest()
            # A partial block without its digest is never reusable.
            path.write_bytes(value)
            write_text_atomic(digest_path, digest + "\n")
        self._used[start] = {"offset": start, "length": length, "sha256": digest}
        return value

    def read(self, size=-1):
        if self.closed:
            raise ValueError("read of closed range reader")
        if size is None or size < 0:
            size = self.size - self._position
        size = min(operator.index(size), max(0, self.size - self._position))
        if size > self.max_bytes:
            raise ValueError("requested read exceeds bounded TIFF budget")
        end = self._position + size
        parts = []
        while self._position < end:
            start = (self._position // self.block_size) * self.block_size
            block = self._block(start)
            stop = min(end, start + len(block))
            parts.append(block[self._position - start : stop - start])
            self._position = stop
        return b"".join(parts)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[: len(value)] = value
        return len(value)

    def provenance(self) -> dict:
        """Return only actually accessed ranges; no full-slide checksum claim."""
        ranges = sorted(self._used.values(), key=lambda row: row["offset"])
        return {
            **self.identity,
            "ranges": ranges,
            "range_fingerprint": fingerprint(ranges),
            "downloaded_bytes": self.downloaded_bytes,
            "verification_scope": "accessed byte ranges; not full remote object",
        }


def read_tiled_region(page, xywh: tuple[int, int, int, int]):
    """Decode intersecting tiles at their native dtype and pixel resolution.

    Coordinates are XY in this TIFF page (not automatically base-level XY).
    Supports contiguous RGB and single-channel 2D pages; rejects planar/Z
    ambiguity. Crops must lie inside the image; missing support is not padded.
    """
    import numpy as np

    x, y, width, height = map(operator.index, xywh)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > page.imagewidth
        or y + height > page.imagelength
    ):
        raise ValueError("ROI must lie within the selected page")
    if not page.is_tiled or page.planarconfig != 1 or page.imagedepth != 1:
        raise ValueError("ROI decoder requires contiguous, tiled 2D TIFF pages")
    tw, th = page.tilewidth, page.tilelength
    columns = (page.imagewidth + tw - 1) // tw
    result = np.zeros((height, width, page.samplesperpixel), dtype=page.dtype)
    handle = page.parent.filehandle
    for row in range(y // th, (y + height - 1) // th + 1):
        for col in range(x // tw, (x + width - 1) // tw + 1):
            index = row * columns + col
            if not page.databytecounts[index]:
                raise ValueError("requested ROI includes a missing TIFF tile")
            handle.seek(page.dataoffsets[index])
            encoded = handle.read(page.databytecounts[index])
            tile, _, _ = page.decode(encoded, index, jpegtables=page.jpegtables)
            if tile is None:
                raise ValueError("requested ROI includes an undecodable TIFF tile")
            left, top = max(x, col * tw), max(y, row * th)
            right, bottom = (
                min(x + width, (col + 1) * tw),
                min(y + height, (row + 1) * th),
            )
            result[top - y : bottom - y, left - x : right - x] = tile[
                0,
                top - row * th : bottom - row * th,
                left - col * tw : right - col * tw,
            ]
    return result[..., 0] if page.samplesperpixel == 1 else result


def ome_pixels(xml: str) -> dict:
    """Extract explicit XY units and channel identities from one OME image."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    pixels = root.findall(".//{*}Pixels")
    if len(pixels) != 1:
        raise ValueError("select a single OME image before extracting metadata")
    p = pixels[0]
    return {
        "pixels": dict(p.attrib),
        "channels": [dict(c.attrib) for c in p.findall("{*}Channel")],
    }
