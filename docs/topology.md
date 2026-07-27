# Semantic Topology Reconstruction

Histopia reconstructs a compact selected-K semantic volume from a completed
registration run and its bound semantic atlas. It does not synthesize
full-resolution histology. Observed semantic planes remain unchanged; optional
virtual planes contain interpolated membership weights and explicit
uncertainty.

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
result seals the benchmark, observed or virtual planes, full scientific meshes,
smaller browser meshes, quantitative class summaries, and source
fingerprints.

Build a fixed-viewport review:

```bash
histopia-visualize topology-review /path/to/review \
  --run sample=/path/to/topology-run
```

Review every transition before approval. Pair feedback supports
`accept`, `hold`, and `reject`, issue labels, comments, and a suggested interval
count. Approval is fingerprint-bound:

```bash
histopia-topology approve \
  --run /path/to/topology-run \
  --reviewer "Reviewer name" \
  --review-notes "Reviewed surfaces, spacing, and every transition."
```

Estimated volumes and surface areas are not physical measurements when z
positions are inferred or assumed. Histopia records that provenance in
`topology_result.json`.
