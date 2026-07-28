# Semantic Topology Reconstruction

Histopia reconstructs a continuous tissue envelope and compact selected-K
semantic fields from a completed registration run and its bound semantic
atlas. The anatomical envelope comes from the approved registration masks;
semantic patches do not define the outer tissue boundary. It does not
synthesize full-resolution histology.

Install the workflow dependencies:

```bash
pip install "histopia[topology]"
```

Create a configuration from
[`examples/topology_config.toml`](../examples/topology_config.toml), then run:

```bash
histopia-topology doctor
histopia-topology preflight --config topology.toml
histopia-topology benchmark --config topology.toml
histopia-topology run --config topology.toml
histopia-topology qc --run /path/to/topology-run
```

`section_thickness_um` is required. Histopia never derives this physical value
from image pixels. Supply `z_manifest` when section positions or known missing
cuts are available. Without a manifest, the workflow evaluates morphology gap
inference with one-to-three-section endpoint holdouts. If gap-count accuracy is
insufficient, it abstains and labels z geometry as uniformly assumed rather
than adding unsupported virtual sections.

The benchmark reports tissue Dice, macro semantic Dice, boundary F1, gap-count
accuracy, and gains over zero-flow and nearest-section baselines. A topology
result additionally compares linear signed-distance, guarded correspondence
flow, and shape-preserving interpolation with leave-one-section-out approved
mask tests. The selected envelope must meet the declared tissue Dice and
boundary F1 gates.

`reconstruction_samples_per_interval` controls numerical z sampling and
defaults to 8. These samples make a continuous field; they are not inferred
histology sections. `envelope_max_xy_dim_px` bounds the reconstruction grid and
defaults to 384. The result seals the dense scientific fields, approved-mask
provenance, display meshes, quantitative class summaries, and source
fingerprints. Display-only smoothing and component filtering never change the
stored measurements.

Semantic viewer meshes use the dominant smoothed class field and retain only
components with cross-section persistence or substantial class volume, up to
eight components per class. A global morphology class can legitimately occupy
several disconnected tissue regions. The component limit prevents isolated
patches from dominating the 3D view; it does not alter the dense semantic field
or quantitative class volume.

Build a fixed-viewport review:

```bash
histopia-visualize topology-review /path/to/review \
  --run sample=/path/to/topology-run
```

The reviewer opens with a translucent connected tissue envelope and one
selectable semantic region. Physical, 12x review, and 25x strong z modes are
explicit display choices; 12x is the default. Camera presets, a cutaway plane,
uncertainty overlay, and observed-section locator support inspection without
changing reconstruction geometry. Camera fitting is anchored to the anatomical
envelope, and semantic-region visibility can be toggled independently. The same
application is used by the standalone topology route and the workflow review
hub.

For morphology-aware inspection, use a section viewer built from both the
registration and semantic runs. Its histology, semantic blend, adjacent-pair,
and topology-link modes retain source-image context:

```bash
histopia-visualize build /path/to/viewer-root \
  --run sample=/path/to/registration-run \
  --semantic-run sample=/path/to/semantic-run
```

Review the connected volume and every transition before approval. Feedback
supports `accept`, `hold`, and `reject`, structured issues, comments, and a
suggested transition interval count. Approval is fingerprint-bound:

```bash
histopia-topology approve \
  --run /path/to/topology-run \
  --reviewer "Reviewer name" \
  --review-notes "Reviewed surfaces, spacing, and every transition."
```

Estimated volumes and surface areas are not physical measurements when z
positions are inferred or assumed. Histopia records that provenance in
`topology_result.json`.
