# Cell boundary detection

`histopia.cells` detects instance boundaries in native whole-slide pixels while
binding every result to an approved Histopia registration. Native-space
inference avoids interpolation artifacts; the recorded registration transform
supports registered and future 3D views.

The method family is derived from a benchmarked brightfield workflow. Histopia
supports direct Cellpose inference plus recursive Combined and containment
variants. The production default preserves the validated whole-image
reference's core CPSAM Combined parameters: cell-probability/diameter pairs
`-2.5/26` and `-1.75/15`, followed by containment merge at
intersection-over-smaller `0.1`. Both Combined passes use flow threshold `0.0`
and minimum size 15. The exact current reference-script SHA-256 is recorded
without exposing its external storage path.

The recorded profile is explicitly a whole-slide adaptation, not a claim that
the external runner is executed byte-for-byte. Histopia adds overlapping WSI
tiles with deterministic stitching, clips predictions to the accepted
registration tissue mask, and applies nuclear and local optical-density debris
guards. These adaptations preserve the reference segmentation constants while
making complete-slide execution resumable and preventing glass or debris from
being promoted as cells.
The accepted mask is a hard outer constraint, not sufficient evidence by
itself: hole filling or low-resolution mask geometry can include nearby glass.
The production profile therefore also requires each retained instance to
overlap a locally stain-supported native-source context bin. This conservative
spatial-breadth check rejects isolated dust that can carry nucleus-like optical
density while retaining cells embedded in actual tissue.
Second-scale additions must contain conservative hematoxylin-like nuclear
optical-density evidence. After stitching, the complete instance set receives a
second, blockwise nuclear-support check. Its validated defaults require at least
8 supported pixels and 2 percent supported area at optical density 0.15. This
removes isolated dead-cell/debris contours from either scale while retaining
the tissue-contained population. Overlapping WSI tiles are stitched by
mutual-best instance matches.

## Installation

```bash
python -m pip install 'histopia[cells]'
histopia-cells doctor --device auto
histopia-cells cache-model --model cpsam --device cpu
```

For an exact tested environment, apply `constraints/cells-repro.txt`. Model
weights are cached separately and their SHA-256 digest is recorded in every
result; weights are not distributed with Histopia. Review the Cellpose model
and training-data terms before using the workflow commercially.
Inference does not download missing weights implicitly. Use `cache-model`
explicitly on an internet-connected machine before starting a production run.
CPU selection is supported for diagnostics and constrained workloads, but CPSAM
whole-slide inference is compute intensive. Benchmark representative tiles
before a CPU-only run; the validated KPF whole-slide profile uses CUDA.

## Run

Start from `examples/cell_segmentation_config.toml` and use an approved
registration run whose native source slides remain available.

```bash
histopia-cells preflight --config cell_segmentation_config.toml
histopia-cells run --config cell_segmentation_config.toml
histopia-cells validate --run /path/to/cell-run
```

Outputs contain one pyramidal `uint32` label TIFF per selected section,
per-section QC, a registration-bound preflight, and a sealed
`cell_result.json`. Interrupted inference reuses fingerprint-matched tile masks.
Once a section finishes, its QC also seals the current preflight/profile
fingerprint and label-TIFF SHA-256. A restarted run reuses that section only
when the checkpoint, exact label geometry/type, digest, zero-outside-tissue QC,
and tile accounting all remain valid. Missing, legacy, partial, or changed
checkpoints are recomputed instead of being trusted.
Every tile prediction is clipped to the accepted registration tissue mask at
native resolution after inference. QC records the number of affected boundary
tiles and removed outside-tissue pixels; a valid final label image contains no
nonzero pixels outside that mask. The final blockwise audit records this as
`outside_tissue_pixels_final = 0` and also reports any clipped fragments below
the configured minimum area as `post_constraint_cells_below_min_size`. The
production WSI profile reapplies `min_size` after clipping and stitching,
records removed fragments separately, and requires the remaining count to be
zero before promotion.
After stitching, a fixed local optical-density evidence check removes instances
that contain no measurable brightfield stain, including regular CPSAM contours
occasionally produced by nearly uniform glass inside a deliberately hole-filled
tissue mask. Results record the evidence thresholds and removed instance/pixel
counts. The check uses local background estimation so it does not impose one
absolute intensity threshold across slides.
The local-background window is recorded in the result fingerprint alongside
the optical-density thresholds so evidence decisions remain reproducible.
The global nuclear-support settings and their removed instance/pixel counts are
also fingerprinted and written to section QC. A block-adaptive nuclear-core
gate additionally tests unusually large instances (at least 225 square
micrometres) against the local 70th-percentile hematoxylin concentration. Core
pixels must also be blue/purple (the blue channel is at least 1.08 times the
stronger red/green channel), preventing neutral-black or brown debris from
masquerading as hematoxylin. The gate requires at least eight pixels or 10% of
the instance to meet both conditions, whichever is larger. This targets
lumen/debris contours without imposing a fixed optical-density cutoff across
differently stained slides. Disable any
nuclear gate only for a separately benchmarked assay without a nuclear
counterstain.
The source-context defaults use globally aligned 256-pixel bins, require at
least 3 percent stain-bearing pixels in a bin, and require at least 1 percent of
an instance to overlap supported bins. These native-pixel thresholds are
fingerprinted and must be revalidated if scan resolution or assay preparation
changes materially.
An additional isolated-debris gate uses globally aligned 64-pixel bins to
distinguish connected tissue from material inside lumens or on glass. It finds
8-connected tissue components from bins with at least 80 percent stain
evidence and supports a component only when it contains at least eight adaptive
chromatic blue/purple nuclear pixels. At least 75 percent of an instance must
overlap a supported component; an isolated nucleus cannot validate its own
disconnected candidate. It also rejects an instance when
at least half its pixels are neutral-dark (RGB maximum at most 90 and chroma at
most 30), or when at least 45 percent are extremely dark and neutral (RGB
maximum at most 60 and chroma at most 20). The chroma constraint prevents
hematoxylin-rich viable tissue from being mistaken for black debris. Together
these checks prevent dead-cell clusters and attached black/brown folds from
borrowing support from neighboring tissue, while retaining strongly stained
non-nuclear markers embedded in ordinary tissue. The gate is applied only
after the Box-reference CPSAM Combined-containment inference and therefore does
not alter its boundary-generation constants.

A sparse-pale guard handles the complementary case where CPSAM outlines small
particulate debris together with mostly empty lumen or glass. It requires an
instance of at least 25 square micrometres to average at least 190 intensity,
no more than 25 red-minus-blue units, and to lie predominantly in a local field
with less than 40 percent predicted coverage. In source contexts containing
less than 20 percent stain, the guard also removes muted fragments averaging at
least 120 intensity and no more than 35 red-minus-blue units. Color statistics are accumulated
from every native instance pixel rather than sampled grid centers; this avoids
bias when a small debris object's center happens to land on its darkest pixel.
Dense viable cell sheets fail the sparse-context condition, while strongly
chromatic or optically dense cells fail the pale-color conditions. These
thresholds, exact statistics, and removed instance/pixel counts are sealed in
the run QC.

A separate detached-fragment topology guard addresses elongated scanning or
preparation debris that CPSAM divides into several cell-sized compartments.
It requires at least three touching muted instances, each at least 200 square
micrometres, whose combined area is at least 1,000 square micrometres and whose
native-aligned aggregate has a bounding-box aspect ratio of at least 3. The
mean source color must remain at or below 35 red-minus-blue units and 220 RGB
intensity, and the surrounding 512-pixel field must contain no more than 20%
predicted coverage. Once that bounded cluster qualifies, only immediately
touching muted model fragments are removed with it. A solitary large cell,
dense tissue, or a small fragment by itself cannot activate the rule. The
physical thresholds, context threshold, and removed label/pixel counts are
fingerprinted and recorded in section QC.

Crushed fragments need not be elongated. A stricter compact branch activates
only when at least two touching pseudo-cells are each at least 300 square
micrometres, total at least 750 square micrometres, average no more than zero
red-minus-blue units and 180 RGB intensity, and remain inside the same sparse
512-pixel context. This targets compact blue-gray preparation fragments while
preserving warm/brown large cells and densely populated viable tissue.

A second, finer 16-pixel topology check protects against a coarse-grid edge
case in which a detached luminal aggregate falls in an 8-connected 64-pixel
bin next to viable epithelium. It identifies small connected components whose
bins are at least 80 percent stain-bearing and rejects an instance only when at
least half of it lies in a component of at most 512 bins with less than 1
percent adaptive chromatic hematoxylin support. Large continuous tissue escapes
this micro-island rule, as do small nuclear-rich tissue islands. The thresholds
are native-pixel and fingerprinted; section QC records the additional removed
instance and pixel counts separately.

Strong DAB can obscure blue nuclear chromaticity even in organized epithelium.
The validated fallback therefore builds a 4-pixel CPSAM-occupancy grid inside
overlapping 1,024-pixel native windows (128-pixel overlap) and allows an
instance to escape the coarse isolated gate when at least half of it belongs
to a local 8-connected prediction component containing at least 25,000 native
predicted pixels and a bounding-box aspect ratio of at least 4. Compact
components qualify only at 50,000 predicted pixels. A bin must be at least 25
percent occupied to join the component. The bounded windows prevent unrelated
tissue and debris from changing one another's component statistics elsewhere
on a whole slide, while overlap avoids hard seams. This size-and-shape-bounded
topology rule also requires at least 350 predicted pixels per sampled CPSAM
instance, which distinguishes coherent epithelial mosaics from fragmented
dead-cell aggregates. It cannot let a small detached object
validate itself, and it does not bypass the neutral-dark, very-dark, or
weak-nuclear micro-island rejection rules. Section QC records how many
instances and pixels were preserved specifically by this organized-tissue
escape.

A complementary compact-mosaic guard catches the opposite failure mode:
foreign luminal material whose regular walls are traced as hundreds of small
pseudo-cells. On the same 4-pixel grid it finds contiguous nuclear-unsupported
prediction fields, closes only one grid-bin gaps, and rejects instances whose
majority lies in a compact component of at least 10,000 square micrometres
with no more than 250 predicted pixels per sampled instance. The component
must also remain source-color neutral: its sampled native pixels may average
no more than 21 red-minus-blue intensity units. A candidate is rejected only
when no more than 42.5 percent of the predictions in its bounding context have
nuclear support and its aspect ratio is no greater than 1.75. These coupled
gates also require the unsupported predictions to fill at least 20 percent of
their component bounding box. In addition, the complete areas of the instances
that would actually be rejected—not merely their larger connected context—must
meet the same 10,000-square-micrometre floor. A strong-chromaticity safeguard
also preserves a component when more than 10 percent of its unsupported bins
have a native red-minus-blue difference above 40. This distinguishes strongly
brown target-positive nuclei or epithelium from the validated pale honeycomb
debris pattern even when the component-wide mean remains neutral. These
constraints preserve sparse folded or annular glands. A second architecture
safeguard estimates each removable instance's covariance from native-aligned
4-pixel samples. The component is preserved when more than half of its
measurable instances have a major-to-minor covariance ratio above 2.0. This
protects organized gland-wall cells even when the folded region itself has a
compact bounding box; instances represented by fewer than three samples do
not vote. Together the gates
preserve genuine eosinophilic, DAB-positive, PAS, and red-stained tissue,
including weakly nuclear elongated epithelial ribbons, while rejecting dense
pale honeycomb-like luminal debris.
Individually nuclear-supported instances are excluded from the artifact field.
The physical-area conversion and removed instance and pixel counts are
recorded in section QC.

Finally, a physical-size guard rejects a single instance larger than 1,000
square micrometres when at least half its pixels meet the adaptive blue-core
criterion. This targets homogeneous PAS/hematoxylin precipitate, folded tissue,
and merged chromatic debris that a boundary model can otherwise return as one
giant cell. It does not reject pale large cells by size alone.

A complementary dense-fold guard handles crushed, attached tissue that appears
nuclear and therefore cannot use the aneuclear safeguard. It rejects only an
elongated cluster (aspect ratio at least 3) of at least three oversized
compartments (each at least 300 square micrometres; at least 750 square
micrometres combined). Candidate connectivity is limited to an eight-bin
dilation on the native-aligned 4-pixel grid, bridging at most approximately 64
native pixels across a preparation fold. Samples must also be very dark (mean
RGB intensity at most 115) and not strongly red/brown (mean red-minus-blue at
most 60). Compact tumor groups, isolated large cells, bright tissue, and
strongly brown target-positive tissue therefore remain outside this rule. The
exact thresholds and removed fold counts are sealed in section QC.

When upgrading an earlier result to this profile, Histopia restitches the
cached raw tile predictions and applies the complete evidence filter exactly
once. It never reapplies contextual gates to an already filtered label image,
because those neighborhood-dependent gates are intentionally not assumed to
be idempotent.

The validated v82 refilter adds one conservative shape gate for sparse objects
that survive inside hole-filled mask regions on glass. It acts only on the
cached raw CPSAM predictions, requires a low-occupancy elongated object in a
128-pixel context with limited tissue support, and preserves organized tissue
through independently nuclear-supported local prediction context. The sealed
profile uses a 500-square-micrometre maximum protected instance area, a 2.5%
minimum independent local-prediction fraction, and exact sparse-glass limits
recorded in the result. A full 25-section regression reused all 3,313 inference
tiles: 23 sections were pixel-identical, and the remaining two removed only
visually confirmed glass/debris contours with no restoration or relabeling.
This profile is eligible for targeted cache-only rescue; it does not silently
change already sealed v75 results.

For the validated mixed-size method, set `method = "containment"` and configure
`first_diameter` and `second_diameter` for the larger and smaller populations.
Each Combined mask recursively evaluates the network twice, so this costs four
network passes and must be accepted by high-resolution review for every
section. The reproducible
`multiscale_nuclear_support`, `nuclear_minimum_optical_density`,
`nuclear_minimum_pixels`, and `nuclear_minimum_fraction` controls apply only to
the smaller second pass. Disable the gate only for a separately benchmarked
assay that lacks a nuclear counterstain.

`inference_batch_size` controls the number of WSI tiles evaluated
together and is recorded in the result fingerprint. Keep the portable default
of 8 unless the target accelerator has been benchmarked for memory, runtime, and
mask equivalence; changing it invalidates cached tile predictions by design.
Fresh runs batch tiles in deterministic merge order; resumed runs may mix
cached and inferred batches without changing the final stitch order.
On an NVIDIA A100 40 GB, a four-tile H&E/CD3 check across both 26 px and 15 px
passes found batch 32 to be 1.20 times faster than batch 8, with identical cell
counts and minimum foreground Dice 0.99999757. Batch 64 improved only another
1.9 percent while increasing peak allocation, so the validation run used 32.
These measurements establish runtime equivalence for that profile, not a new
portable default.

## Scientific validation

Existing centroid annotations quantify detection and instance containment, but
they are not contour ground truth. Histopia therefore reports point-based
precision, recall, and F1 without estimating true negatives, and requires
high-resolution visual acceptance of every section before the run is approved.
Nuclei classification, stain-intensity assignment, and 3D cell rendering are
outside this module.

Compare supported profiles on an annotated tile collection with:

```bash
histopia-cells benchmark \
  --images /path/to/tiles \
  --annotations /path/to/centroid-tsv-files \
  --output /path/to/benchmark \
  --model-cache /path/to/cellpose-cache \
  --device cuda:0
```

The summary ranks profiles by protein-macro centroid-instance F1, then pooled
tile F1 and runtime. Boundary quality still requires visual inspection of the
generated overlays because centroid points are not contour annotations.

## Review and QuPath

`histopia-visualize cell-review` builds a fixed-viewport, native-resolution
OpenSeadragon reviewer. Raw histology and transparent cell boundaries remain
separate layers, and one **Accept** click records a fingerprint-bound
provisional observation. Browser observations are append-only triage: they do
not satisfy the independent section-review or final scientific-approval gate.
Every selected section must still be formally accepted before final
cell-result approval.

Before publishing a completed run to a stable reviewer, call
`validate_cell_promotion_candidate`. Unlike the general artifact validator,
this promotion gate also requires exact schema-v2 preflight coverage, algorithm
version 75 with profile v71 or separately regression-validated algorithm
version 82 with profile v78, the hashed CPSAM Combined-containment reference
profile, accepted tissue-mask clipping, both nuclear-support guards, local
debris evidence, zero final outside-tissue pixels, and complete per-section
tile accounting. The v82 branch additionally requires every exact refilter and
sparse-glass safeguard; arbitrary combinations cannot pass. Supply a fresh
`preflight_cell_run(config)` as `expected_preflight` so an old or partial
selection cannot promote itself by declaring a smaller scope.

The companion QuPath extension can launch the same registration-bound cell
workflow, open its local review application, and materialize cells intersecting
a selected ROI as editable QuPath detections. The ROI limit is deliberate so a
large WSI label image is not expanded into millions of JVM objects at once.
