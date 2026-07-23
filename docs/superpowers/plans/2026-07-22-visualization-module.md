# Visualization Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make viewer generation and HTTP serving first-class `histopia.visualization` capabilities with a stable `/histopia/` deployment endpoint.

**Architecture:** Move the existing static viewer implementation into a dedicated visualization package and retain thin registration compatibility exports. Add a dependency-light server API and CLI that serve a viewer root, redirect `/` to `/histopia/`, and leave all generated assets external to the repository.

**Tech Stack:** Python 3.10+, `argparse`, `http.server`, NumPy, Pillow, Three.js static assets, pytest, Playwright validation.

## Global Constraints

- Canonical imports live under `histopia.visualization`.
- Existing registration imports remain compatible during the alpha API period.
- Static generation and HTTP serving remain separate APIs.
- The stable deployed endpoint is `/histopia/`; versioned review paths are not public contracts.
- Refresh restores accepted section order; drag changes are session-only.
- Display at most 500 highest-confidence links for one selected adjacent pair.
- Keep complete uncapped correspondence arrays in semantic result artifacts.
- Do not track datasets, model weights, generated viewer assets, local paths, or secrets.

---

### Task 1: Canonical Visualization Package

**Files:**
- Create: `src/histopia/visualization/__init__.py`
- Create: `src/histopia/visualization/_viewer.py`
- Modify: `src/histopia/registration/_viewer.py`
- Modify: `src/histopia/registration/__init__.py`
- Modify: `src/histopia/registration/_cli.py`
- Create: `tests/test_visualization_api.py`
- Modify: `tests/test_registration_review_and_viewer.py`

**Interfaces:**
- Produces: `histopia.visualization.build_section_viewer(...) -> Path`
- Produces: `histopia.visualization.build_section_order_review(...) -> Path`
- Preserves: `histopia.registration.build_section_viewer` and `_viewer` imports.

- [ ] **Step 1: Write failing canonical-import and compatibility tests**

```python
from histopia import visualization
from histopia.registration import build_section_viewer as legacy_builder


def test_visualization_is_canonical_viewer_api() -> None:
    assert visualization.build_section_viewer is legacy_builder
    assert visualization.MAX_DISPLAY_LINKS == 500
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_visualization_api.py -q`

Expected: FAIL because `histopia.visualization` does not exist.

- [ ] **Step 3: Move implementation and add compatibility exports**

Move the complete viewer implementation to
`histopia.visualization._viewer`, define `MAX_DISPLAY_LINKS = 500`, use it as
the default in `_viewer_topology_pairs`, and make registration's `_viewer.py`
a re-export-only compatibility module:

```python
from histopia.visualization._viewer import (
    build_section_order_review,
    build_section_viewer,
)

__all__ = ["build_section_order_review", "build_section_viewer"]
```

Update registration's public imports and CLI lazy import to consume
`histopia.visualization`.

- [ ] **Step 4: Run visualization and registration viewer tests**

Run: `.venv/bin/python -m pytest tests/test_visualization_api.py tests/test_registration_review_and_viewer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/histopia/visualization src/histopia/registration tests/test_visualization_api.py tests/test_registration_review_and_viewer.py
git commit -m "Add canonical visualization package"
```

### Task 2: Viewer Server API And CLI

**Files:**
- Create: `src/histopia/visualization/_server.py`
- Create: `src/histopia/visualization/_cli.py`
- Modify: `src/histopia/visualization/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_visualization_server.py`
- Create: `tests/test_visualization_cli.py`

**Interfaces:**
- Produces: `serve_viewer(root: Path | str, *, bind: str = "0.0.0.0", port: int = 8765) -> None`
- Produces: `create_viewer_server(root: Path | str, *, bind: str, port: int) -> ThreadingHTTPServer`
- Produces CLI: `histopia-visualize serve ROOT [--bind HOST] [--port PORT]`

- [ ] **Step 1: Write failing route and validation tests**

```python
def test_server_redirects_root_to_stable_endpoint(tmp_path: Path) -> None:
    stable = tmp_path / "histopia"
    stable.mkdir()
    (stable / "index.html").write_text("viewer")
    server = create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
    # Start in a test thread, assert GET / is 302 to /histopia/, and
    # GET /histopia/ returns the viewer body, then shut down.


def test_server_rejects_missing_stable_viewer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="histopia/index.html"):
        create_viewer_server(tmp_path, bind="127.0.0.1", port=0)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_visualization_server.py tests/test_visualization_cli.py -q`

Expected: FAIL because server and CLI modules do not exist.

- [ ] **Step 3: Implement dependency-light serving**

Subclass `SimpleHTTPRequestHandler` only to redirect exact `/` requests to
`/histopia/`. Construct `ThreadingHTTPServer` with `functools.partial` so the
handler serves the explicit root without changing process working directory.
Validate `root/histopia/index.html` before binding and close the server when
`serve_forever()` exits.

- [ ] **Step 4: Implement CLI and package script**

Add `histopia-visualize = "histopia.visualization._cli:main"` and implement:

```text
histopia-visualize serve VIEWER_ROOT --bind 0.0.0.0 --port 8765
```

Reject unsupported commands through `argparse`; print the stable URL after
binding and before serving.

- [ ] **Step 5: Run server, CLI, and package-import tests**

Run: `.venv/bin/python -m pytest tests/test_visualization_server.py tests/test_visualization_cli.py tests/test_visualization_api.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/histopia/visualization tests/test_visualization_server.py tests/test_visualization_cli.py
git commit -m "Add visualization server and CLI"
```

### Task 3: Stable Deployment, Documentation, And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/dependency_management.md`
- Modify: `docs/semantic_atlas.md`
- Modify: `docs/kpf_registration_validation.md`
- Modify: `examples/semantic_atlas_config.toml`
- Modify: `constraints/semantic-repro.txt`

**Interfaces:**
- Consumes: `build_section_viewer` and `histopia-visualize serve` from Tasks 1-2.
- Produces: stable external `viewer_root/histopia/index.html` deployment.

- [ ] **Step 1: Update public documentation**

Document canonical visualization imports, `histopia-visualize serve`, the
500-link display cap, uncapped result artifacts, automatic K=5..15, guarded
batch diagnostics, and platform-specific PyTorch constraints. Replace tracked
machine-local paths with `/path/to/...` examples.

- [ ] **Step 2: Run full static verification**

Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
git diff --check
```

Expected: all tests pass; Ruff and diff checks report no errors. Existing
unrelated formatting findings must be reported rather than silently changed.

- [ ] **Step 3: Build stable external viewer**

Call `histopia.visualization.build_section_viewer` with the accepted 5996
registration and semantic result paths, writing directly to the external
`viewer_root/histopia` directory. Do not copy any generated assets into Git.

- [ ] **Step 4: Start package-owned server**

Stop only the existing Histopia port-8765 server, then run:

```bash
.venv/bin/histopia-visualize serve VIEWER_ROOT --bind 0.0.0.0 --port 8765
```

Verify `/` redirects to `/histopia/` and `/histopia/` returns HTTP 200.

- [ ] **Step 5: Browser acceptance test**

Use Playwright at 1920x1080 and 3840x2160. Assert no body overflow, a nonblank
canvas, no console errors, 11 K options, 22 adjacent-pair options, K switching,
pair switching, visible QC, and refresh restoring Siriusred to accepted
position 18 after a session-only reorder.

- [ ] **Step 6: Commit and push**

```bash
git add README.md docs examples constraints src tests pyproject.toml
git commit -m "Complete semantic topology visualization workflow"
git push origin feature/semantic-topology-v2
```

Confirm `AGENTS.md`, `PLANS.md`, local staging docs/scripts, datasets, model
weights, and generated viewer assets are absent from the commit.
