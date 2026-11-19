# %% [markdown]
# Step 09 - Assemble the UtilWAS association results into results/utilwas_results/analysis/
# utilwas_association.xlsx. Full per-test results stay in the results_*.csv; this workbook holds
# the readable summaries + the FDR-significant adjusted (M4) findings.
# Sheets: README, specialty_summary, specialty_results, phecode_summary, phecode_sig_m4.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import utilwas_utils as U

RD = dict(keep_default_na=False, na_values=[""])
OUT = U.ANALYSIS / "utilwas_association.xlsx"


def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    spec = pd.concat([_num(pd.read_csv(U.ANALYSIS / f"results_specialty_{w}.csv", **RD),
                           ["effect", "p", "ci_low", "ci_high"]) for w, _ in U.WINDOWS], ignore_index=True)
    phe = pd.concat([_num(pd.read_csv(U.ANALYSIS / f"results_phecode_{w}.csv",
                          dtype={"feature": str, "phecode": str, "combo": str}, **RD),
                          ["effect", "p", "ci_low", "ci_high"]) for w, _ in U.WINDOWS], ignore_index=True)

    def summary(df, keys):
        g = df.groupby(keys)
        return pd.DataFrame({
            "n_tests": g.size(),
            "n_fdr_sig": g["sig_fdr"].sum(),
            "n_bon_sig": g["sig_bon"].sum(),
        }).reset_index()

    spec_sum = summary(spec[spec.model.isin(["m1", "m3", "m4"])], ["window", "family", "part", "contrast", "model"])
    phe_sum = summary(phe[phe.model.isin(["m1", "m3", "m4"])], ["window", "combo", "part", "contrast", "model"])

    # FDR-significant fully-adjusted (M4) findings (presence OR & count IRR)
    sig = phe[(phe.model == "m4") & (phe.sig_fdr) & (phe.effect_type.isin(["OR", "IRR"]))].copy()
    sig = sig[["window", "combo", "part", "feature", "phenotype", "category", "contrast",
               "effect_type", "effect", "ci_low", "ci_high", "p", "neglog10p"]]
    sig = sig.rename(columns={"feature": "phecode"}).sort_values(
        ["window", "combo", "part", "p"]).reset_index(drop=True)
    sig[["effect", "ci_low", "ci_high"]] = sig[["effect", "ci_low", "ci_high"]].round(3)

    readme = pd.DataFrame([
        ("stage", "UtilWAS association: utilization outcomes vs OSA severity"),
        ("cohort", "v2 ICD cohort, 222,109 (None 60,618 / Mild 67,040 / Moderate 44,957 / Severe 49,494)"),
        ("windows", "1yr & 5yr pre-index; contrasts: omnibus, trend, mild/moderate/severe-vs-none, AHI>=5, AHI>=15"),
        ("models", "unadjusted screen + M1 crude / M3 +age,sex,race,year,smoking,income,obs_window / M4 +BMI"),
        ("specialty (physician_level & allied_health #1)", "encounters/yr per DEPT_SPECIALTY -> Poisson-rate IRR; "
         "unadjusted KW/Spearman/Mann-Whitney; results_specialty_*.csv (full)"),
        ("phecode two-part (physician/ED/urgent/inpatient dx)", "PART1 presence -> logistic OR; "
         "PART2 count-among-present -> Poisson-rate IRR; pyPheWAS v1.2 maps; >=500-patient floor; results_phecode_*.csv"),
        ("multiplicity", "FDR (BH) + Bonferroni within each (family/combo, part, contrast, model)"),
        ("effect_type key", "OR=odds ratio (presence); IRR=incidence rate ratio (rate/count); "
         "unadjusted: epsilon2/cramers_v (omnibus), spearman_rho/ca_z (trend), rank_biserial/OR (pairwise)"),
        ("this sheet phecode_sig_m4", "FDR-sig, fully-adjusted (M4) presence-OR & count-IRR findings only"),
        ("caveat", "M4 (BMI-adjusted) is the honest dose-response; crude signals are confounded (e.g. sepsis "
         "OR 1.71 crude -> 1.10 ns M4). 73/38,303 phecode fits failed (separation), recorded NA."),
    ], columns=["field", "detail"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        readme.to_excel(xl, sheet_name="README", index=False)
        spec_sum.to_excel(xl, sheet_name="specialty_summary", index=False)
        spec.round(4).to_excel(xl, sheet_name="specialty_results", index=False)
        phe_sum.to_excel(xl, sheet_name="phecode_summary", index=False)
        sig.to_excel(xl, sheet_name="phecode_sig_m4", index=False)

    print(f"wrote {OUT}")
    print(f"  specialty tests: {len(spec):,}; phecode tests: {len(phe):,}")
    print(f"  M4 FDR-sig phecode findings (OR/IRR): {len(sig):,}")
    print("  by part:", sig.part.value_counts().to_dict())
    U.log_milestone("Step 09 association workbook: utilwas_association.xlsx "
                    "(specialty + phecode summaries + M4 FDR-sig findings).")


if __name__ == "__main__":
    main()
