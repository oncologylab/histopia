# Seven-Mouse Semantic Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce fresh, reproducible, globally clustered UNI2-H semantic atlases and comparable review artifacts for seven registration-validated KPF mice.

**Architecture:** Add strict registration preflight and feature-provenance layers before the existing semantic extractor, then add cohort QC over portable semantic result files. Extend the visualization payload with cohort metrics while preserving the stable `/histopia/` endpoint, and run all scientific artifacts outside the repository through restartable per-mouse stages.

**Tech Stack:** Python 3.10+, NumPy, Pillow, pyvips, PyTorch, timm, UNI2-H, scikit-learn, SciPy, pytest, Ruff, Three.js, Playwright.

## Global Constraints

- Phase-one mice are 3528, 4257, 4312, 4577, 4630, 5996, and 5997.
- Extract all features afresh from current Histopia registrations; never reuse legacy embeddings.
- Use one fingerprinted local UNI2-H checkpoint and one shared extraction configuration.
- Fit independent global K values from 5 through 15 for every mouse.
- Preserve uncapped accepted links in result artifacts and display at most 500 links for one selected adjacent pair.
- Do not modify raw WSI data or commit data, checkpoints, generated results, local paths, credentials, `AGENTS.md`, or `PLANS.md`.
- Every semantic result starts unapproved; only the user can approve its exact fingerprint.

---

### Task 1: Registration Preflight And Run Manifest

**Files:**
- Create: `src/histopia/semantic/_preflight.py`
- Modify: `src/histopia/semantic/__init__.py`
- Modify: `src/histopia/semantic/_cli.py`
- Create: `tests/test_semantic_preflight.py`

**Interfaces:**
- Produces: `preflight_registration(registration_run: Path | str) -> SemanticPreflight`
- Produces: `write_preflight(preflight: SemanticPreflight, output_path: Path | str) -> Path`
- Produces CLI: `histopia-semantic preflight --config CONFIG`

- [ ] **Step 1: Write failing tests for complete, approved, fingerprinted input validation**

Create synthetic registration results and assert that preflight records ordered
slides, paths, mask hashes, transform hashes, MPP, reference identity, and an
overall fingerprint. Add focused failures for missing masks, duplicate slides,
non-finite transforms, absent MPP, mismatched thumbnail/mask dimensions, and an
unapproved order manifest when one is present.

- [ ] **Step 2: Run the preflight tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_semantic_preflight.py -q`

Expected: FAIL because `histopia.semantic._preflight` does not exist.

- [ ] **Step 3: Implement typed preflight records and deterministic hashing**

Use frozen dataclasses for per-slide and run-level records. Hash file content in
chunks, canonicalize JSON before hashing structured metadata, validate every
input before writing, and exclude absolute paths from the portable fingerprint.

- [ ] **Step 4: Add the preflight CLI**

`histopia-semantic preflight --config CONFIG` writes
`OUTPUT/preflight.json`, prints the fingerprint and slide count, and exits
nonzero with a slide-specific message on failure.

- [ ] **Step 5: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_semantic_preflight.py -q
.venv/bin/ruff check src/histopia/semantic tests/test_semantic_preflight.py
.venv/bin/ruff format --check src/histopia/semantic tests/test_semantic_preflight.py
```

Commit: `Add semantic registration preflight`

### Task 2: Fingerprinted Atomic Feature Artifacts

**Files:**
- Modify: `src/histopia/semantic/_features.py`
- Modify: `src/histopia/semantic/_extract.py`
- Modify: `src/histopia/semantic/_uni2h.py`
- Modify: `src/histopia/semantic/_pipeline.py`
- Modify: `tests/test_semantic_features.py`
- Create: `tests/test_semantic_extraction_cache.py`

**Interfaces:**
- Extends: `PatchFeatures` schema version 2 with `provenance_json` and `fingerprint`
- Produces: `Uni2hEncoder.model_fingerprint: str`
- Consumes: Task 1 `SemanticPreflight`

- [ ] **Step 1: Write failing schema-v2 and cache tests**

Assert round-trip provenance, atomic save without a leftover temporary file,
exact cache reuse, and forced recomputation after any registration, mask,
checkpoint, or extraction-parameter fingerprint changes. Assert schema-v1
artifacts remain readable but are never accepted as fresh campaign caches.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_semantic_features.py tests/test_semantic_extraction_cache.py -q
```

Expected: FAIL because feature provenance and cache validation are absent.

- [ ] **Step 3: Implement portable provenance and atomic writes**

Store canonical provenance JSON in each NPZ, derive its fingerprint from input
fingerprints plus extraction settings, write to a same-directory temporary
file, validate it, and replace the destination atomically with `Path.replace`.

- [ ] **Step 4: Fingerprint the local model snapshot**

Derive a stable model fingerprint from the resolved Hugging Face snapshot
revision and model configuration without placing checkpoint paths or weights in
results. Expose it on the encoder and include it in every feature artifact.

- [ ] **Step 5: Make extraction preflight-dependent and restartable**

Reject extraction when preflight is missing or stale. Validate each existing
artifact against expected provenance; reuse only exact matches. Emit one
machine-readable extraction status row per slide.

- [ ] **Step 6: Verify and commit**

Run the semantic feature, cache, extraction, and UNI2-H tests plus Ruff.

Commit: `Add reproducible semantic feature caching`

### Task 3: Per-Mouse And Cohort Semantic QC

**Files:**
- Create: `src/histopia/semantic/_qc.py`
- Modify: `src/histopia/semantic/_result.py`
- Modify: `src/histopia/semantic/_cli.py`
- Modify: `src/histopia/semantic/__init__.py`
- Create: `tests/test_semantic_qc.py`
- Modify: `tests/test_semantic_result.py`

**Interfaces:**
- Produces: `summarize_semantic_run(run_dir: Path | str) -> SemanticRunQc`
- Produces: `write_cohort_qc(runs: Mapping[str, Path | str], output_path: Path | str) -> Path`
- Produces CLI: `histopia-semantic cohort-qc --run NAME=PATH ... --output PATH`

- [ ] **Step 1: Write failing QC tests**

Use compact synthetic results to verify slide completeness, patch counts,
tissue-fraction summaries, selected-K metrics, minimum cluster fraction,
topology link count, confidence, coverage, unsupported sections, batch decision,
and deterministic cohort outlier flags. Reject mixed or stale fingerprints.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_semantic_qc.py -q`

Expected: FAIL because semantic cohort QC does not exist.

- [ ] **Step 3: Implement deterministic result-only QC**

Read portable semantic JSON and NPZ artifacts without importing Torch, pyvips,
or model code. Keep raw measurements separate from threshold flags and record
the threshold definitions in the cohort payload.

- [ ] **Step 4: Add result provenance and CLI output**

Include preflight, extraction configuration, and model fingerprints in schema
version 3 semantic results. Write JSON and a compact TSV cohort table through
the CLI without embedding local paths.

- [ ] **Step 5: Verify and commit**

Run semantic QC and result tests, then the complete semantic test directory and
Ruff.

Commit: `Add multi-mouse semantic quality control`

### Task 4: Multi-Mouse Review Viewer

**Files:**
- Modify: `src/histopia/visualization/_viewer.py`
- Modify: `src/histopia/visualization/_cli.py`
- Modify: `tests/test_registration_review_and_viewer.py`
- Modify: `tests/test_visualization_cli.py`
- Create: `tests/test_visualization_cohort.py`

**Interfaces:**
- Extends: `build_section_viewer(..., cohort_qc: Path | str | None = None) -> Path`
- Extends CLI: `histopia-visualize build ROOT ... --cohort-qc PATH`

- [ ] **Step 1: Write failing cohort-viewer tests**

Assert seven mouse choices, per-mouse QC status, K=5 through K=15, topology
pair selection, 500-link display cap, semantic review state, and absence of
absolute source paths in generated assets.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_visualization_cohort.py tests/test_visualization_cli.py -q
```

Expected: FAIL because cohort QC is not accepted or displayed.

- [ ] **Step 3: Add compact cohort status to the existing viewer**

Keep one mouse selector and the current non-scrolling layout. Show concise QC
and review state for the selected mouse; do not add a dashboard or separate
landing page. Preserve session-only ordering and accepted-order refresh.

- [ ] **Step 4: Verify browser behavior**

Test synthetic seven-mouse output at 1920x1080 and 3840x2160 with Playwright.
Assert no overflow, console errors, failed assets, stale state across mouse
changes, or frame drops caused by loading nonselected textures.

- [ ] **Step 5: Verify and commit**

Run all visualization tests, Ruff, and browser checks.

Commit: `Add cohort semantic review visualization`

### Task 5: Seven-Mouse External Campaign

**Files:**
- Create externally: one config, preflight, feature directory, semantic result,
  and review record per mouse
- Create externally: cohort QC and stable viewer assets
- Modify publicly only if needed: `docs/semantic_atlas.md`
- Modify locally only if needed: `PLANS.md`

**Interfaces:**
- Consumes: Tasks 1-4 CLIs and public APIs
- Produces: seven fingerprinted review candidates at `/histopia/`

- [ ] **Step 1: Resolve one current registration per mouse**

Use the most recent user-accepted run, never infer acceptance from modification
time. Record the seven selected run fingerprints in a local campaign manifest.

- [ ] **Step 2: Run and inspect all preflights**

Run preflight for all seven mice. Inspect every failure and remediate the source
mask, transform, order, or metadata before extraction. Do not bypass a failed
gate.

- [ ] **Step 3: Extract fresh UNI2-H features**

Use one GPU process, one local checkpoint fingerprint, and shared settings.
Resume only exact fingerprint-valid slides. Record elapsed time, patch counts,
peak GPU memory, and failures per slide.

- [ ] **Step 4: Fit all seven global atlases**

Fit K=5 through K=15, guarded batch correction, and adjacent topology. Confirm
all registered slides occur exactly once in each result and inspect numerical
flags before viewer generation.

- [ ] **Step 5: Build cohort QC and stable viewer**

Generate the external cohort JSON/TSV and rebuild the existing stable
`viewer_root/histopia` endpoint with all seven registrations and semantic runs.
Restart only the package-owned server if needed.

- [ ] **Step 6: Perform visual acceptance testing**

Inspect masks, registered histology, every selected-K semantic stack, K
alternatives, topology pairs, and batch diagnostics for every mouse. Run
Playwright at both required viewport sizes and capture review screenshots
outside Git.

- [ ] **Step 7: Request human review**

Present the stable public URL, selected K, batch decision, key outliers, and
exact fingerprint for each mouse. Keep every review record unapproved until the
user explicitly approves it.

- [ ] **Step 8: Final verification and sanitized push**

Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
```

Confirm local-only files and all generated artifacts are untracked. Commit and
push sanitized code and documentation. Mark the goal complete only after all
seven exact result fingerprints are approved.
