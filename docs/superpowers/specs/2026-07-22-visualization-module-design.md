# Histopia Visualization Module Design

## Goal

Create a first-class `histopia.visualization` module that owns generation and
serving of interactive review viewers. The module provides a stable package API
and a stable browser endpoint while allowing viewer behavior to evolve without
coupling it to registration internals.

## Architecture

The module has two independent boundaries:

- `histopia.visualization.build_section_viewer(...)` generates deterministic
  static assets from accepted registration and optional semantic results.
- `histopia.visualization.serve_viewer(...)` serves a viewer root with a
  configurable host and port. Serving does not import WSI, GPU, or semantic
  extraction dependencies.

`histopia.visualization` becomes the canonical import path. Existing imports
from `histopia.registration` remain as compatibility aliases during the alpha
API period, but registration no longer owns viewer implementation code.

## Stable Endpoint

The deployed viewer root contains a stable `histopia/` directory. The package
server exposes it at `/histopia/`, independent of dataset-specific or versioned
build directories. The server root redirects to `/histopia/`.

For the current host, the stable public URL is:

`http://149.165.171.95:8765/histopia/`

Rebuilding the current review replaces generated assets within that stable
directory. Versioned result and review directories remain external artifacts,
not user-facing endpoints.

## Viewer Behavior

- Refresh always restores the accepted section order. Dragging is session-only.
- Semantic K is selectable from every fitted candidate.
- Quantitative K and batch-correction diagnostics remain visible.
- One adjacent section pair is selected at a time.
- The topology overlay renders at most the 500 highest-confidence accepted
  links for that pair in one `THREE.LineSegments` object.
- Complete, uncapped correspondence arrays remain in semantic result artifacts.
- Texture replacement disposes prior GPU resources.
- Layout remains non-scrolling at 1920x1080 and 3840x2160.

## CLI

Add `histopia-visualize` with two commands:

- `build`: generate a stable viewer directory from named registration and
  optional semantic runs.
- `serve`: serve an existing viewer root with configurable `--bind` and
  `--port`, defaulting to `0.0.0.0:8765`.

The CLI accepts paths explicitly and does not embed local datasets, remote
storage configuration, tokens, or deployment-specific addresses in source.

## Errors And Safety

- Missing registration results, semantic labels, topology artifacts, or
  selected K values fail with actionable errors during generation.
- Serving rejects a missing viewer root before binding a socket.
- Generated review state remains external and untracked.
- Model weights, WSI files, masks, and viewer assets never enter package
  distributions.

## Testing

- Unit tests cover public imports, CLI argument handling, endpoint routing,
  dynamic K manifests, topology caps, and compatibility aliases.
- Viewer tests verify no persistent browser order storage.
- Browser validation checks initialization, K switching, pair switching,
  console errors, nonblank canvas pixels, and no overflow at 1080p and 4K.
- Existing registration and semantic tests remain unchanged in behavior.

## Migration

1. Move viewer implementation into `histopia.visualization`.
2. Add compatibility re-exports from registration.
3. Add build and serve APIs plus the `histopia-visualize` CLI.
4. Build the current 5996 review directly into the stable `histopia/` target.
5. Redirect the server root to `histopia/` and verify the public endpoint.
