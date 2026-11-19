# Archive verification

Verified November 18, 2026.

- All 187 archived source files were copied without edits and checked against the originals using SHA-256. Their paths and checksums are in `CODE_MANIFEST.csv`.
- Every archived Python file and the layout-restoration tool passed Python syntax parsing. The JavaScript workbook builder passed Node.js syntax checking.
- `tools/restore_layout.py` reconstructed all 187 code files in a new local verification directory. Its path-containment and existing-destination checks were also exercised.
- All 26 existing synthetic incidence tests passed in the restored layout with Python 3.12.7 in the existing `myenv2` conda environment. An audit hook blocked opening files in the original study workspace and on clinical-data drives during the test run. No clinical input files were opened by these checks.
- Tests cover tied-event competing-risk estimates and confidence intervals, event-step export, phenotype event grids, octant export validation, age-sex boundaries, sparse strata, and inference/multiplicity handling.
- The two Charlson reference files matched the pinned upstream commit after line-ending normalization. Their upstream GPL license was retained.
- The publication folder was checked for unexpected file types, prohibited paths, credential patterns, and literal patient-identifier patterns. Only code, the file manifest, repository settings, and documentation were included. Notebook files, clinical data, generated result files, and manuscript documents were excluded.

Full analyses were not rerun. The archive does not contain the private inputs, reviewed crosswalks, model-selection files, or all historical aggregate plotting assets needed to reproduce the study's numerical results and final displays. Passing the synthetic tests establishes properties of the tested estimators and exports; it does not independently validate every study finding.

## Observed Python package versions

These versions were read from the existing `myenv2` environment during archive verification. They describe that environment and are not a complete dependency lockfile.

| Package | Version |
| --- | --- |
| Python | 3.12.7 |
| DuckDB | 1.2.1 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.16.0 |
| statsmodels | 0.14.5 |
| lifelines | 0.30.3 |
| pyPheWAS | 4.2.0 |
| scikit-learn | 1.7.1 |
| stepmix | 3.0.0 |
| matplotlib | 3.10.6 |
| pyarrow | 21.0.0 |
| openpyxl | 3.1.5 |
| rapidfuzz | 3.14.5 |
| adjustText | 1.4.0 |
| joblib | 1.5.2 |
| umap-learn | 0.5.12 |
| Pillow | 11.1.0 |

The atlas JavaScript builder imports `@oai/artifact-tool`; that original authoring runtime is not distributed here.
