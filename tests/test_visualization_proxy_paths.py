from __future__ import annotations

import pytest

from histopia.visualization._review_portal import _WORKFLOW_JS


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/proxy/8765", "/workspace/proxy/8765"])
def test_review_navigation_and_api_keep_port_proxy_prefix(prefix: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    origin = "http://histopia.test"
    seen: list[str] = []
    html = (
        """<nav></nav><iframe id="review"></iframe><script>
    globalThis.HISTOPIA_WORKFLOW_REVIEW = {tabs: [
      {id: "cells", label: "Cells", href: "cells/index.html"},
      {id: "selected-cells", label: "Selected", href: "/selected-cells/index.html"}
    ]};
    </script><script>"""
        + _WORKFLOW_JS
        + """</script><script>
    fetch(histopiaUrl('/api/wsi/4312/001')).then(r => {
      document.body.dataset.apiStatus = String(r.status);
    });
    </script>"""
    )
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page()

        def respond(route) -> None:
            seen.append(route.request.url)
            if route.request.url == origin + prefix + "/review/":
                route.fulfill(content_type="text/html", body=html)
            elif route.request.url in {
                origin + prefix + "/review/cells/index.html",
                origin + prefix + "/selected-cells/index.html",
                origin + prefix + "/api/wsi/4312/001",
            }:
                route.fulfill(content_type="text/html", body="Ready")
            else:
                route.fulfill(status=404, body="Wrong base path")

        page.route("**/*", respond)
        page.goto(origin + prefix + "/review/")
        page.locator('body[data-api-status="200"]').wait_for()
        page.locator('button[data-tab="selected-cells"]').click()
        page.frame_locator("#review").get_by_text("Ready").wait_for()
        assert origin + prefix + "/selected-cells/index.html" in seen
        assert origin + prefix + "/api/wsi/4312/001" in seen
        browser.close()
