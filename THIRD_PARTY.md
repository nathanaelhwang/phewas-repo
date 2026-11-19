# Third-party Charlson reference code

The two files below are unchanged reference sources from Alessandro Gasparini's **comorbidity** R package. The study's `derive_baseline_charlson.py` parses their mapping and weight definitions without executing R.

- `phase_4_reporting/tmp/phewas_260923/charlson_reference/make-mapping.R`
- `phase_4_reporting/tmp/phewas_260923/charlson_reference/make-weights.R`

Upstream repository: [ellessenne/comorbidity](https://github.com/ellessenne/comorbidity), commit `8f5f79d379c3198b807e824b0f2d703688818fda`. Upstream paths are `data-raw/make-mapping.R` and `data-raw/make-weights.R`. The local sources were verified against that commit, allowing for line endings.

The package DESCRIPTION at that commit declares `GPL (>= 3)`. Its original [LICENSE.md](phase_4_reporting/tmp/phewas_260923/charlson_reference/LICENSE.md) is preserved beside the reference files. This notice identifies the upstream files; no new license was selected for the study's own code during compilation.
