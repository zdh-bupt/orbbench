# Anonymous review artifact

This package contains the OrbBench code, available inputs, recorded experimental
outputs, model checkpoints, figures, and environment specifications from the
public experimental package. The included results and scope are documented in
README.md and DATA.md. This is not a distribution of the entire raw dataset.

Personal account names in recorded paths have been replaced with neutral names.
The raw-data directory layout in DATA.md and pipeline/orbpipe/config.py uses
`trajectory_data/` and `processed_data/` beneath `ORBBENCH_DATA_ROOT`.
When rerunning experiments, arrange external inputs under those neutral directory
names. Paths under `/home/reviewer/` and `C:\Users\reviewer\` are historical
run records, not directories included in this package.

The archive contains no Git history or remote configuration. Numeric experiment
results, trained model weights, and the included binary datasets are unchanged.
Third-party data citations and license notices are retained. No additional
license for the project code is granted by this packaging step.

See README.md for the existing result checks and component-specific instructions.
Large raw sources, some staged data, and certain model inputs were not included
in the source release; DATA.md records these limitations.

## Files split for hosting

Three Parquet result files exceed the hosting service's per-file limit. Their
bytes are included as parts of at most 6,000,000 bytes under `artifact_parts/`.
After downloading the complete repository, restore the original files with:

```bash
python restore_large_files.py
```

This standard-library-only script verifies the size and SHA-256 of every part
and restored file. It does not change the dataset contents. Missing or corrupted
parts cause it to stop. The paths and checksums are recorded in
`artifact_parts_manifest.json`.
