# %% [markdown]
# v2 Step 6 - Build the two requested table sets into results/icd_analysis_v2_results/tables/.
#
# TASK 1 (5 tables, one per target): take that target's TOP 100 POSITIVE (LOR>0) conditions ranked
#   by m1 (crude) p-value; report each condition's p-value and LOR under m1/m2/m3/m4. Shows how each
#   condition's importance changes as covariates are added.
#
# TASK 2 (4 tables, one per model m1..m4): take the top-100-positive features for EACH of
#   none-vs-mild / -moderate / -severe; rows = the union; a contrast's pvalue/LOR cell is filled ONLY
#   if that feature is in that contrast's top-100 (blank otherwise = informative). Shows which
#   features are valuable across OSA severities.
#
# LOR = log odds ratio = the regression beta. Selection excludes unstable (separation) fits.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import utils as U

COMB = U.V2_SUMMARY / "phewas_results_combined.csv"
OUT = U.V2_RESULTS / "tables"

MODELS = ["m1_crude", "m2_demo", "m3_smk_ses_win", "m4_bmi"]
MK = {"m1_crude": "m1", "m2_demo": "m2", "m3_smk_ses_win": "m3", "m4_bmi": "m4"}
TARGETS = ["ahi_ge5", "ahi_ge15", "mild_vs_none", "moderate_vs_none", "severe_vs_none"]
SEV = ["mild_vs_none", "moderate_vs_none", "severe_vs_none"]
SEV_LABEL = {"mild_vs_none": "none_vs_mild", "moderate_vs_none": "none_vs_moderate",
             "severe_vs_none": "none_vs_severe"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(COMB, dtype={"PheWAS Code": str})
    d["unstable"] = d["unstable"].astype(bool)

    def sub(contrast, model):
        return d[(d["contrast"] == contrast) & (d["model"] == model)].set_index("PheWAS Code")

    def top100_pos(contrast, model):
        s = sub(contrast, model)
        return s[(s["beta"] > 0) & (~s["unstable"])].nsmallest(100, "p-val")

    sheets = {}

    # ---- TASK 1 ----
    for tgt in TARGETS:
        top = top100_pos(tgt, "m1_crude")
        codes = top.index
        t = pd.DataFrame({"PheCode": codes.values, "condition": top["PheWAS Name"].values})
        for m in MODELS:
            t[f"{MK[m]}_pvalue"] = sub(tgt, m).reindex(codes)["p-val"].values
        for m in MODELS:
            t[f"{MK[m]}_LOR"] = sub(tgt, m).reindex(codes)["beta"].values
        t.to_csv(OUT / f"task1_{tgt}.csv", index=False)
        sheets[f"t1_{tgt}"[:31]] = t
        print(f"task1_{tgt}: {len(t)} conditions")

    # ---- TASK 2 ----
    for m in MODELS:
        tops = {c: top100_pos(c, m) for c in SEV}
        union = sorted(set().union(*[set(t.index) for t in tops.values()]))
        name_map = {}
        for c in SEV:
            for code, r in tops[c].iterrows():
                name_map.setdefault(code, r["PheWAS Name"])
        t = pd.DataFrame({"PheCode": union})
        t["Feature"] = t["PheCode"].map(name_map)
        for c in SEV:  # pvalue cols (filled only if in that contrast's top-100)
            t[f"{SEV_LABEL[c]}_pvalue"] = t["PheCode"].map(tops[c]["p-val"])
        for c in SEV:  # LOR cols
            t[f"{SEV_LABEL[c]}_LOR"] = t["PheCode"].map(tops[c]["beta"])
        pcols = [f"{SEV_LABEL[c]}_pvalue" for c in SEV]
        t["_n"] = t[pcols].notna().sum(axis=1)
        t["_minp"] = t[pcols].min(axis=1)
        t = t.sort_values(["_n", "_minp"], ascending=[False, True]).drop(columns=["_n", "_minp"])
        t.to_csv(OUT / f"task2_{MK[m]}.csv", index=False)
        sheets[f"t2_{MK[m]}"] = t
        n3 = int((t[pcols].notna().sum(axis=1) == 3).sum())
        print(f"task2_{MK[m]}: {len(t)} features ({n3} in all 3 severities)")

    # ---- combined workbook ----
    with pd.ExcelWriter(OUT / "all_tables.xlsx", engine="openpyxl") as xl:
        for sh, df in sheets.items():
            df.to_excel(xl, sheet_name=sh, index=False)

    (OUT / "README.md").write_text(
        "# v2 result tables\n\n"
        "LOR = log odds ratio (regression beta). Selection = top-100 POSITIVE (LOR>0, non-unstable) "
        "associations by p-value.\n\n"
        "## Task 1 (per target): task1_<target>.csv\n"
        "Rows = the target's top-100 positive conditions by **m1 (crude)** p; columns give p-value and "
        "LOR under m1/m2/m3/m4. Read across a row to see a condition's importance as covariates are added "
        "(m1 crude -> m2 +age,sex,race,year -> m3 +smoking,income,obs-window -> m4 +BMI).\n\n"
        "## Task 2 (per model): task2_<model>.csv\n"
        "Rows = union of the top-100-positive features across none-vs-mild/moderate/severe (for that model). "
        "A contrast's pvalue/LOR is filled ONLY if the feature is in that contrast's top-100 (blank = not a "
        "top-100 hit there). Sorted by # severities present then significance. Shows which features matter "
        "across OSA severity.\n\n"
        "`all_tables.xlsx` has all 9 tables as sheets.\n", encoding="utf-8")
    print(f"\nWrote 5 + 4 tables + all_tables.xlsx -> {OUT}")
    U.log_milestone("v2 Step 6 result tables (Task1 x5 per-target, Task2 x4 per-model).")


if __name__ == "__main__":
    main()
