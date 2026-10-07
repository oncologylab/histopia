import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from histopia.study._remote_tiff import RangeReader, read_tiled_region
from histopia.study._serial import validate_serial_study


@pytest.fixture
def serial():
    subject = dict(source_id="public", species="human", subject_id="one")
    sections = [
        dict(
            **subject,
            physical_section_id=f"s{i}",
            organ="colon",
            specimen_id="specimen",
            block_id="block",
            section_order=i,
            z_um=z,
            z_reference="s0",
            z_evidence="documented cut log",
        )
        for i, z in enumerate([0, 5, 10, 14])
    ]
    return dict(
        schema_version="serial-1",
        study_id="serial-v1",
        subjects=[dict(**subject, role="development", exposed=True)],
        sections=sections,
        acquisitions=[
            dict(
                acquisition_id="he",
                physical_section_id="s0",
                modality="HE",
                source_binding="original-scan",
            ),
            dict(
                acquisition_id="restain",
                physical_section_id="s0",
                modality="IHC",
                source_binding="restained-scan",
            ),
        ],
    )


def test_nonuniform_z_and_restains(serial):
    result = validate_serial_study(serial)
    assert len(result["sections"]) == 4
    assert result["sections"][-1]["z_um"] == 14
    serial["sections"][1]["z_um"] = None
    serial["sections"][1]["z_evidence"] = None
    assert validate_serial_study(serial)["fingerprint"] != result["fingerprint"]


@pytest.mark.parametrize(
    "change", ["order", "depth", "evidence", "acquisition_z", "test_fit"]
)
def test_rejects_false_physical_identity_and_leakage(serial, change):
    if change == "order":
        serial["sections"][1]["section_order"] = 0
    elif change == "depth":
        serial["sections"][1]["z_um"] = 0
    elif change == "evidence":
        serial["sections"][1]["z_evidence"] = None
    elif change == "acquisition_z":
        serial["acquisitions"][1]["z_um"] = 5
    else:
        serial["subjects"][0].update(role="test", exposed=False)
        serial["fits"] = [{"subjects": [["public", "human", "one"]]}]
    with pytest.raises(ValueError):
        validate_serial_study(serial)


def test_subject_scope_cannot_split_by_organ(serial):
    serial["subjects"].append(dict(serial["subjects"][0], role="test", exposed=False))
    with pytest.raises(ValueError, match="unique"):
        validate_serial_study(serial)


@pytest.mark.parametrize("rgb", [False, True])
def test_native_tiff_region_matches_original_across_tiles(tmp_path, rgb):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    shape = (64, 80, 3) if rgb else (64, 80)
    original = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    path = tmp_path / "tiny.tif"
    tifffile.imwrite(
        path, original, tile=(16, 16), photometric="rgb" if rgb else "minisblack"
    )
    with tifffile.TiffFile(path) as tiff:
        np.testing.assert_array_equal(
            read_tiled_region(tiff.pages[0], (13, 9, 30, 37)), original[9:46, 13:43]
        )
        with pytest.raises(ValueError, match="within"):
            read_tiled_region(tiff.pages[0], (70, 0, 20, 20))


def test_range_reader_cache_identity_budget_and_tamper(tmp_path):
    content = bytes(range(256)) * 8
    version = ['"v1"']

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", len(content))
            self.send_header("ETag", version[0])
            self.end_headers()

        def do_GET(self):
            assert self.headers["If-Match"] == version[0]
            start, end = map(int, self.headers["Range"].split("=")[1].split("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.send_header("ETag", version[0])
            self.end_headers()
            self.wfile.write(content[start : end + 1])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/test.tif"
        with RangeReader(url, tmp_path, block_size=128, max_bytes=256) as reader:
            reader.seek(100)
            assert reader.read(70) == content[100:170]
            assert reader.downloaded_bytes == 256
            reader.seek(500)
            with pytest.raises(ValueError, match="budget"):
                reader.read(1)
            original_cache = reader.cache
        with RangeReader(url, tmp_path, block_size=128, max_bytes=256) as reader:
            assert reader.read(70) == content[:70]
            assert reader.downloaded_bytes == 0
            block = next(reader.cache.glob("0-*.bin"))
            block.write_bytes(b"x" * 128)
            reader.seek(0)
            with pytest.raises(ValueError, match="SHA-256"):
                reader.read(1)
        version[0] = '"v2"'
        with RangeReader(url, tmp_path, block_size=128, max_bytes=256) as reader:
            assert reader.cache != original_cache
            assert reader.read(1) == content[:1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("bad", ["ignored", "changed", "wrong-range"])
def test_range_reader_rejects_unbound_server_responses(tmp_path, monkeypatch, bad):
    class Response(io.BytesIO):
        status = 206

    def open_response(request, timeout):
        response = Response(b"abcd")
        response.headers = {
            "ETag": '"v1"',
            "Content-Length": "4",
            "Content-Range": "bytes 0-3/4",
        }
        if request.get_method() != "HEAD":
            if bad == "ignored":
                response.status = 200
            elif bad == "changed":
                response.headers["ETag"] = '"v2"'
            else:
                response.headers["Content-Range"] = "bytes 1-4/4"
        return response

    monkeypatch.setattr("histopia.study._remote_tiff.urlopen", open_response)
    with RangeReader(
        "https://example.test/a.tif", tmp_path, block_size=4, max_bytes=4
    ) as reader:
        with pytest.raises(ValueError, match="ignored range or changed"):
            reader.read(4)
        assert not list(reader.cache.glob("*.bin"))


def test_serial_view_retains_physical_z_and_rejects_duplicate_acquisitions(
    serial, tmp_path
):
    import json

    from PIL import Image

    from histopia.study._manifest import file_sha256
    from histopia.visualization._serial_stack import build_serial_stack_review

    image = tmp_path / "plane.png"
    Image.new("RGBA", (24, 16), (90, 80, 120, 255)).save(image)
    planes = [
        dict(
            physical_section_id=s["physical_section_id"],
            image_path=image,
            image_sha256=file_sha256(image),
            frame_id="registered-frame",
            frame_shape_yx=[16, 24],
            mpp_xy=[0.5, 0.5],
            modality="HE",
            upstream_fingerprint="a" * 64,
        )
        for s in serial["sections"]
    ]
    output = tmp_path / "review"
    build_serial_stack_review(serial, planes, output, title="Observed stack")
    result = json.loads((output / "stack.json").read_text())
    assert [p["z_um"] for p in result["planes"]] == [0, 5, 10, 14]
    assert result["physical_stack_available"]
    assert result["interpolated_planes"] == 0
    assert str(tmp_path) not in (output / "stack.json").read_text()
    with pytest.raises(ValueError, match="one observed plane"):
        build_serial_stack_review(
            serial, planes + [planes[0]], output, title="Repeated scan"
        )
    serial["sections"][1]["z_um"] = None
    build_serial_stack_review(serial, planes, output, title="Unknown gap")
    assert not json.loads((output / "stack.json").read_text())[
        "physical_stack_available"
    ]
    planes[1]["frame_id"] = "other-frame"
    with pytest.raises(ValueError, match="share one"):
        build_serial_stack_review(serial, planes, output, title="Wrong coordinates")


@pytest.mark.browser
def test_serial_navigation_hides_stale_pixels_and_ignores_late_images(serial, tmp_path):
    import json
    import mimetypes
    from urllib.parse import urlsplit

    from PIL import Image

    from histopia.study._manifest import file_sha256
    from histopia.visualization._serial_stack import build_serial_stack_review

    playwright = pytest.importorskip("playwright.sync_api")
    planes = []
    for i, section in enumerate(serial["sections"]):
        path = tmp_path / f"section-{i}.png"
        Image.new("RGB", (24, 16), (40 * i, 60, 80)).save(path)
        planes.append(
            dict(
                physical_section_id=section["physical_section_id"],
                image_path=path,
                image_sha256=file_sha256(path),
                frame_id="frame",
                frame_shape_yx=[16, 24],
                mpp_xy=[0.5, 0.5],
                modality="HE",
                upstream_fingerprint="a" * 64,
            )
        )
    output = tmp_path / "review"
    build_serial_stack_review(serial, planes, output, title="Observed sections")
    manifest = json.loads((output / "stack.json").read_text())
    delayed = manifest["planes"][1]["image"]["href"]
    missing = manifest["planes"][3]["image"]["href"]
    held, errors = [], []

    def fulfill(route, path):
        route.fulfill(
            status=200 if path.is_file() else 404,
            content_type=mimetypes.guess_type(path)[0] or "application/octet-stream",
            body=path.read_bytes() if path.is_file() else b"",
        )

    def respond(route):
        relative = urlsplit(route.request.url).path.removeprefix("/")
        if relative == delayed:
            held.append(route)
        elif relative == missing:
            route.fulfill(status=404)
        else:
            fulfill(route, output / relative)

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("http://histopia.test/**", respond)
        page.goto("http://histopia.test/index.html?mode=section")
        page.locator("#section-image:not([hidden])").wait_for()
        assert "s0" in page.locator("#section-image").get_attribute("alt")
        page.locator("#next").click()
        assert page.locator("#section").input_value() == "s1"
        assert page.locator("#section-image").is_hidden()
        assert page.locator("#viewport").get_attribute("aria-busy") == "true"
        assert page.locator("#image-status").inner_text() == "Loading section…"
        page.locator("#next").click()
        page.locator("#section-image:not([hidden])").wait_for()
        assert "s2" in page.locator("#section-image").get_attribute("alt")
        assert len(held) == 1
        with page.expect_response("http://histopia.test/" + delayed):
            fulfill(held[0], output / delayed)
        page.evaluate(
            "() => new Promise(resolve => "
            "requestAnimationFrame(() => requestAnimationFrame(resolve)))"
        )
        page.wait_for_function(
            "document.querySelector('#viewport').getAttribute('aria-busy') === 'false'"
        )
        assert "s2" in page.locator("#section-image").get_attribute("alt")
        page.locator("#next").click()
        page.get_by_text("Section image unavailable", exact=True).wait_for()
        assert page.locator("#section-image").is_hidden()
        assert page.locator("#viewport").get_attribute("aria-busy") == "false"
        assert "section=s3" in page.url
        browser.close()
    assert not errors
