# Obstructive sleep apnea phenome-wide association analyses

Analysis code supporting the manuscript **A Clinical Atlas for Phenome-Wide Association Study of Obstructive Sleep Apnea Across Electronic Health Record Domains** and its methods appendix, tables, and figures. The archive was assembled on November 18, 2026 from the study's existing analysis sources. Original source files are preserved without edits.

The scope was checked against these document versions:

- `PheWAS Manuscript 261117_clean.docx`
- `PheWAS Appendix 2601001.docx` (filename retained exactly as supplied)
- `PheWAS Tables and Figures 20261014.docx`

This repository contains code and documentation. Individual-level data, patient identifiers, saved analysis outputs, fitted models, notebooks with saved outputs, and the manuscript documents are excluded. The [OSA Atlas](https://www.osaatlas.com/) is referenced in the manuscript for aggregate findings.

| Folder | Contents |
| --- | --- |
| `phase_1_cross_sectional/` | Cohort and prevalent diagnoses; LabWAS, MedWAS, ProcWAS, CRMWAS (`procwas/measwas`), UtilWAS, BehWAS, questionnaire harmonization, and QWAS |
| `phase_2_incidence/` | Incident diagnoses, cause-specific Cox models, competing-risk curves, PAP landmark analyses, age-sex subsets, and phenotype-incidence analyses that share the incidence engine |
| `phase_3_phenotypes/` | Upstream phenotype feature builders and final physiologic, symptom, and comorbidity latent models, model selection, octants, and descriptive summaries |
| `phase_4_reporting/` | Table 1 summaries and Charlson scoring, Table 2 Welch ANOVA, manuscript figure generators and pairwise survival tests, and supplementary atlas assembly |
| `tools/` | A checksum-verified copier to reconstruct the original relative source layout |

[ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) maps the documents to code and describes execution order. [CODE_MANIFEST.csv](CODE_MANIFEST.csv) records the original relative path, archived path, SHA-256 checksum, and inclusion reason for each of the 187 archived code files. [THIRD_PARTY.md](THIRD_PARTY.md) documents the two upstream R mapping files and their license.

## Working with the archive

This is a source archive, not a standalone reproduction of the private study environment. Scripts retain their original local paths, input schemas, crosswalk dependencies, and historical output locations. The scientific code was not refactored during compilation. Numbered scripts may depend on earlier stages, private intermediate files, manual review steps, and existing aggregate exports.

To recover the original relative layout in a new directory outside this repository:

```powershell
python tools/restore_layout.py C:/path/to/new-study-code
```

The copier checks all source hashes before writing and refuses an existing destination. It does not execute analysis code. Before running analyses elsewhere, configure historical absolute paths in the restored copy and provide the required inputs in an authorized local environment. Restore the original layout before running scripts that import helpers from sibling analysis directories. Phase folders are archival groupings, not independent Python packages.

The main Python stack reported in the appendix is Python 3.12.7, DuckDB 1.2.1, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.0, statsmodels 0.14.5, and pyPheWAS 4.2.0. Additional observed dependencies and the limits of verification are listed in [VALIDATION.md](VALIDATION.md). Use an appropriate conda environment; this archive does not install packages or change an environment.

Synthetic incidence tests can be run in a restored tree with the required packages installed:

```powershell
python -m unittest discover -s analyses/incwas -p "test_*.py"
```

The atlas workbook builder also depends on the original `@oai/artifact-tool` JavaScript runtime. Its Python preparation/export code is included, but that runtime is not bundled.
