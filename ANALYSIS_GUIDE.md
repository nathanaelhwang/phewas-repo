# Document to code guide

Paths below are relative to the repository. The same mapping applies to the manuscript's Methods and Results and to the appendix's Full Methods. Figure and table numbering follows the supplied October 14, 2026 tables-and-figures document; historical generator filenames may use earlier figure numbers.

| Document analysis or display | Main code |
| --- | --- |
| Study cohort and prevalent diagnoses; Figure 1 | `phase_1_cross_sectional/analyses/icd_analysis/`: original steps `01`–`04` build inputs needed by final `v2_01`–`v2_04`; `utils.py` and `_model_worker.py` supply shared paths and model execution |
| LabWAS; Figure 2A–C | `phase_1_cross_sectional/analyses/labwas/`: extraction, LOINC matching, clinical seed/unit correction, matrices, association models, ordering propensity, and plotting |
| MedWAS; Figure 3 | `phase_1_cross_sectional/analyses/medwas/`: pharmacy fills, GPI crosswalk/categories, target construction, models, and plotting |
| ProcWAS; Figure 4 | `phase_1_cross_sectional/analyses/procwas/`: procedure extraction, windows, targets, association models, and plots |
| CRMWAS; appendix and atlas | `phase_1_cross_sectional/analyses/procwas/measwas/`: echocardiography repair, spirometry/echo construction, measurement models, and plots |
| UtilWAS; Figure 5 | `phase_1_cross_sectional/analyses/utilwas/`: windowed encounter definitions, count models, specialty and diagnosis/phecode analyses, export, and plots |
| QWAS; Figure 6 | `phase_1_cross_sectional/analyses/questionnaire/` harmonizes the three sources; `phase_1_cross_sectional/analyses/qwas/` selects pre-index observations and fits binary/continuous models |
| BehWAS; Figure 7 | `phase_1_cross_sectional/analyses/behwas/`: behavioral features, medication adherence, FMLA, models, and plots; final five-year panel generator is `phase_4_reporting/tmp/figure7_5yr_20261001/build_figure7.py` |
| Incident PheDAS; Figure 8 | `phase_2_incidence/analyses/incwas/00`–`07` scripts and `_cox_engine.py`: insurance, event store, follow-up, cause-specific Cox regression, and postprocessing |
| Severity and PAP landmark curves; Figure 9 | `phase_2_incidence/analyses/incwas/08`–`17` scripts and `_event_curves.py`; `phase_4_reporting/tmp/phewas_20261001/pairwise.py` independently checks the 64 pairwise tests shown for four outcomes |
| Age-sex-stratified subsets | `phase_2_incidence/analyses/incwas/34_age_sex_curves.py`, `35_age_sex_inference.py`, and `_age_sex_inference.py` |
| Three-domain latent models and octants; Figure 10 and Figure S1 | `phase_3_phenotypes/latent-class-analysis/analysis/`, especially `00_prepare.py` through `05_finalize.py`, `07_phenotypes.py`, and `10_octant_figures.py`; `common/` contains preparation, mixture fitting, and stability checks |
| Upstream maximal phenotype feature tables | `phase_3_phenotypes/archives/clustering/analyses/`: the three domain `01_build.py` scripts and their `common/prep.py` and `common/qc.py` helpers |
| Phenotype incidence; Figures 11–12 | `phase_2_incidence/analyses/incwas/20`–`33` scripts, `_pheno_engine.py`, and `_aj_inference.py`; octant fitting is `24_pheno_octant_cox.py`, exports/validation are `27`–`29`, and inference/annotations are `30_octant_logrank.py`, `32_octant_curve_inference.py`, and `33_figure12_annotations.py` |
| Table 1 and cohort summary | `phase_4_reporting/tmp/codex_revision_260819/build_assets.py` (`summarize_table1`), `tmp/phewas_260923/aggregate_table1_additions.py` (education and source coverage), and `tmp/phewas_260923/derive_baseline_charlson.py` (final baseline Charlson scoring) |
| Table 2 utilization summaries and P values | `phase_1_cross_sectional/analyses/utilwas/11_export_utilization_profile.py`; `phase_4_reporting/tmp/phewas_anova_20261001/calculate_anova.py` computes and independently verifies 14 Welch ANOVA comparisons from aggregate statistics |
| Final manuscript figure revisions | `phase_4_reporting/tmp/phewas_20261001/build_figures.py`, its September plotting dependencies, and the separate final Figure 7 generator |
| Supplementary association atlas | `phase_4_reporting/outputs/was_master_20260919/support/`: preparation, workbook assembly, streaming export, and verification; `v2_08_ipw_severe_vs_none.py` is retained because the atlas preparation explicitly includes its sensitivity output |

## Dependencies and execution order

Reconstruct the relative layout with `tools/restore_layout.py` before execution. Keep generated outputs and private inputs outside this public repository.

1. Build the original cohort, BMI, pre-index phenome, and complete-case cohort using ICD steps `01`–`04`. Build the final v2 cohort and model inputs with `v2_01`–`v2_02`, then run/postprocess the v2 models. The earlier v1 regression runs and interactive explorer are excluded because the final analyses use v2.
2. Build each Phase 1 domain's features and targets before fitting its models. Questionnaire extraction and harmonization precede QWAS. Domain helpers reuse ICD utilities; several pipelines also reuse the UtilWAS association engine. Reviewed crosswalks and terminology resources are inputs, not supplied datasets.
3. Build insurance, the dated phecode event store, and the follow-up spine before incident models and PAP landmark analyses. Run exports before plotting or pairwise testing. Age-sex scripts extend these same prepared incidence inputs.
4. For Phase 3, the archived clustering builders construct maximal domain feature tables. The final latent-model pipeline expects those tables in `latent-class-analysis/data/`; the archive does not establish or automate the historical transfer from the earlier clustering workspace. The symptom builder itself expects an already prepared symptom table. Its complete upstream construction is not present in these final pipeline sources.
5. The latent-model `run_all.sh` runs preparation, domain fitting, and model-selection reporting. It intentionally stops before finalization: class choices must be recorded in `results/K_CHOICES.json` after reviewing the selection report. The supplied documents report three classes per domain. Finalize the models, derive scores/octants, and produce their summaries before phenotype-incidence models and release exports.
6. Run reporting scripts only after their required aggregate files exist. The dated figure generators import earlier generators and consume historical JSON audits/manifests and aggregate exports. Those generated files are excluded, so the code archive alone is insufficient to regenerate the final document displays in one command.

## Scope and provenance limits

The archive includes final model pipelines, their identifiable upstream/shared code, manuscript-specific computations, figure generators, and associated synthetic checks. It omits exploratory notebooks, superseded PPCA/k-means clustering, raw-data profiling scripts, document formatting/comment edits, web application UI, and external-validation tooling that was not described as a completed analysis in the supplied documents.

The phase groupings do not imply that every historical branch or helper appears in a final display. Shared files contain earlier phenotype and sensitivity branches needed by downstream code or the atlas. Inclusion is based on document methods, explicit code dependencies, and named output paths; it is not a claim that every original run or every manually edited display has been independently reconstructed.
