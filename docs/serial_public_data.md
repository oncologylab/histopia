# Public serial-section studies

`histopia.study.validate_serial_study` validates the `serial-1` schema separately
from existing mouse studies. It keeps subjects, physical sections and image
acquisitions distinct. A restain or another fluorescence cycle refers to the same
physical section; it never adds a Z plane. Subject roles apply across organs.
An untouched test subject requires an explicit record of no previous exposure.

Sections require source, species, subject, organ, specimen, block and section
order. Known `z_um` values require both `z_reference` and `z_evidence`. Unknown
positions remain null. Acquisitions bind a physical section, modality and source
identity. Nonuniform spacing and missing acquired sections must be represented
explicitly rather than replacing physical depth with an image-list index.

`histopia.study.RangeReader` provides bounded HTTP range reads using a strong
ETag, object length and SHA-256-verified cached blocks. Its provenance describes
the bytes actually accessed; it does not claim a full-slide checksum. A changed
ETag creates a separate cache namespace. Cache and temporary directories are
caller supplied. Transient connection failures have bounded retries; changed
identities, ignored range requests and corrupted cached blocks are rejected.

`histopia.study.read_tiled_region` decodes only intersecting tiles of a contiguous
2D TIFF page. XY coordinates refer to that page, including the selected pyramid
level. The original dtype is retained. Missing tiles and out-of-image regions
are rejected rather than silently replaced by zeros. TIFF decoding requires the
optional `wsi` dependencies and the image codecs appropriate to the source.

`histopia.visualization.build_serial_stack_review` exports observed planes in
one registered coordinate frame, with a physical Z manifest and a section-image
fallback. Duplicate physical planes, mixed specimens or coordinate frames,
inconsistent image dimensions and conflicting Z references are rejected. The
display Z multiplier does not alter exported physical coordinates. Unknown Z
disables the physical stack while preserving section review. Local Three.js
assets can be supplied with `vendor_dir`; section review remains available when
WebGL or the optional renderer is unavailable.

The results catalog supports additional organs and optional `specimen_id`,
`physical_section_id`, `section_order` and `evidence_kind` presentation fields.
These fields do not change upstream fingerprints or approval records. Its URL
retains source, organ, specimen, section and analysis. Workflow tabs may declare
`scope.sources`, `scope.organs` and `scope.subjects`; an unsupported selection
cannot silently open a different dataset. Monitor services are separate from
active scientific jobs.

Tissue review uses the versioned `serial-brightfield-ihc-1` eligibility policy.
`histopia.study.assess_reconstruction_eligibility(study, assessment)` checks a
frozen serial manifest and acquisition/reconstruction evidence. Public inclusion
requires three distinct physical H&E/IHC sections from one verified block,
including chromogenic IHC, accessible images, documented order and physical Z,
and passing bounded reconstruction and native visual QC. H&E-only, IF, CyCIF
and CODEX stacks are excluded. Any organ or species is eligible. A donor match,
restain, repeated scan or similar-looking image cannot add a physical plane.

`validate_reconstruction_registry` recomputes decisions from the evidence rather
than trusting saved eligibility flags. Missing documentation or QC is pending;
contradictory modality or physical identity is excluded. Previously approved
internal studies may retain assumed Z when bound to the earlier approval. Pass
`spacing_status="assumed"` to the stack exporter to label that assumption in
the viewer and download; never relabel it as measured spacing.

Pass this registry as `reconstruction_registry` to `build_results_catalog`.
Each catalog record must bind its reconstruction ID, eligibility fingerprint,
source/organ/specimen/block, acquisition IDs and acquisition modalities. Derived
analyses must resolve to the same eligible stack. The filter runs before asset
copying, JSON, JavaScript and fallback HTML. It removes unused public asset
copies on revocation, without touching source research artifacts. Every Tissue
review export requires a registry, including a new catalog. Unknown source,
organ, block, dataset or retired view URLs do not open a substitute dataset.

Production publishers must snapshot the previous publication, prepare a clean
filtered release and switch the catalog and reconstruction bundles together.
Registry/policy changes and completed-job records invalidate the publication
cache; every completed job passes the same filter. Historical catalogs are
archive records, not automatic publication inputs. Keep candidates outside
the web root until they qualify; an empty public catalog is valid. Retire
excluded compute queues and retries while preserving checkpoints and results.

`histopia.study.validate_reconstruction_job(request, registry, policy)` applies
the same eligibility evidence before computation. A `reconstruction-job-1`
request binds the reconstruction, study and eligibility fingerprints, complete
specimen identity, acquisition IDs/modalities, method fingerprint and wall-time
limit. The `reconstruction-compute-policy-1` policy names retired public sources
and explicit qualification limits (`max_acquisitions`, `max_wall_seconds`).
Registration expansion, segmentation, features, prediction, annotation, region summaries,
neighborhoods, audits and exports require an eligible reconstruction.

Pending candidates permit only `provenance`, `native_qc` or
`bounded_reconstruction`, with an explicit `qualification_purpose` and the
policy's acquisition/time bounds. Geometry also requires accessible bound
images, verified serial identity and verified brightfield staining. Qualification
does not publish a candidate or change its evidence. An excluded reconstruction
cannot use qualification to resume processing.

`ReconstructionJobLedger(path, policy_path=...)` wraps durable single-owner
assignments. The policy file supplies `registry_path` (relative to the policy or
absolute). `add(job_id, node, request, payload)` checks current evidence before
reserving an artifact. `claim(job_id, node)` rechecks that evidence and records a
rejected claim as blocked. Executors call `authorize(job_id)` at launch and
checkpoints; `finish(..., status="complete", evidence=...)` rechecks completion.
Revoked outputs remain archived with a blocked outcome. Renaming a job or
increasing its time budget cannot repeat a reserved artifact; retries require
an explicit new assignment with new method or input evidence.

Execution adapters must verify the bound executable/input bytes, enforce the
declared acquisition scope and time budget, and preserve partial outputs when
stopped. The lightweight API does not launch processes or infer a safe stale
claim timeout. It does not replace protected mouse evaluation order, feature
identity checks, model fit scope validation or independent biological review.

Fluorescence remains outside Tissue review's reconstruction scope. Viewer
contrast is not a measurement calibration. Coarse
registration, candidate nuclear boundaries and automated feature residuals are
review evidence, not independent cell-level ground truth or validated biological
niches. Published serial examples and repository metadata must establish
physical continuity before a source is treated as a reconstruction cohort.
