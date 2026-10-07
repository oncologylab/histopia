# External brightfield validation

Histopia keeps two independent publication scopes:

- **Tissue review** requires an eligible serial reconstruction. Every displayed
  analysis resolves to its verified physical tissue block and stack.
- **External validation** accepts a versioned, donor-bound brightfield benchmark.
  It cannot grant reconstruction eligibility.

`histopia.study.validate_external_benchmark` checks acquisition identity, verified
brightfield H&E/IHC modality, documented DAB targets, and donor roles across organs.
Previously exposed donors remain in development. Duplicate image URLs or known
image checksums cannot cross donor splits. A changed manifest invalidates its jobs.

`ExternalBenchmarkLedger` adds immutable method and execution bindings to shared
storage assignments. Claims have a single owner, completion rechecks authorization,
and failures retain their provenance. Operators must investigate failures before
creating explicitly versioned repair attempts; claims are never expired by age.

`histopia.visualization.build_external_validation` publishes only image evidence
bound to that benchmark. Test images additionally require a completed evaluation,
the unchanged model-selection receipt created before test evaluation, and an
independent numerical audit. These private receipts are verified before export;
their filesystem paths are omitted from the public catalog.

The counterstain benchmark uses a fixed H/DAB matrix and white point, H-derived
support, and a fixed display range. Native nonoverlapping patches retain their
coordinates, support counts, and mean DAB optical densities. Unknown physical
scales remain unknown. Unsupported patches remain missing.

Frozen UNI2-h features and hematoxylin statistics can feed donor-weighted ridge
models. Feature normalization uses training acquisitions only. Regularization is
selected on validation donors, then model files and selection evidence are sealed
before test targets are read. Reports separate donor-level errors, within-donor
patch correlations, donor bootstrap intervals, and organ/antibody strata. Training
mean and shuffled-target controls accompany the primary model.

IHC-derived counterstain is not matched H&E. Fixed deconvolution may retain
chromogen cross-talk, and optical density is a staining proxy rather than an
absolute protein concentration. A numerical benchmark does not establish
same-cell correspondence, cell-type accuracy, mouse-model generalization, or
biological validity of a 3D reconstruction.

The portal opens an available specimen within the selected source and organ,
updates the URL, and briefly identifies automatic switches. Excluded source,
dataset, and reconstruction links stay unavailable. Methods and detailed metrics
remain collapsed so the images are visible immediately.
