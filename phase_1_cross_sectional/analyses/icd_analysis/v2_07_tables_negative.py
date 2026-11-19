# %% [markdown]
# v2 Step 7 - Same two table sets as v2_06 but for NEGATIVE associations (LOR<0 / OR<1).
# Output: ONLY results/icd_analysis_v2_results/tables/all_tables_negative.xlsx (9 sheets).
# Task 1: per target, top-100 NEGATIVE by m1 p, with p+LOR under m1..m4.
# Task 2: per model, union of top-100-negative across none-vs-mild/mod/severe; cell filled only if
#         the feature is in that contrast's negative top-100.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import utils as U

COMB = U.V2_SUMMARY / "phewas_results_combined.csv"
OUTXLSX = U.V2_RESULTS / "tables" / "all_tables_negative.xlsx"

MODELS = ["m1_crude", "m2_demo", "m3_smk_ses_win", "m4_bmi"]
MK = {"m1_crude": "m1", "m2_demo": "m2", "m3_smk_ses_win": "m3", "m4_bmi": "m4"}
TARGETS = ["ahi_ge5", "ahi_ge15", "mild_vs_none", "moderate_vs_none", "severe_vs_none"]
SEV = ["mild_vs_none", "moderate_vs_none", "severe_vs_none"]
SEV_LABEL = {"mild_vs_none": "none_vs_mild", "moderate_vs_none": "none_vs_moderate",
             "severe_vs_none": "none_vs_severe"}


def main() -> None:
    d = pd.read_csv(COMB, dtype={"PheWAS Code": str})
    d["unstable"] = d["unstable"].astype(bool)

    def sub(contrast, model):
        return d[(d["contrast"] == contrast) & (d["model"] == model)].set_index("PheWAS Code")

    def top100_neg(contrast, model):  # NEGATIVE associations, most significant
        s = sub(contrast, model)
        return s[(s["beta"] < 0) & (~s["unstable"])].nsmallest(100, "p-val")

    sheets = {}

    # ---- TASK 1 ----
    for tgt in TARGETS:
        top = top100_neg(tgt, "m1_crude")
        codes = top.index
        t = pd.DataFrame({"PheCode": codes.values, "condition": top["PheWAS Name"].values})
        for m in MODELS:
            t[f"{MK[m]}_pvalue"] = sub(tgt, m).reindex(codes)["p-val"].values
        for m in MODELS:
            t[f"{MK[m]}_LOR"] = sub(tgt, m).reindex(codes)["beta"].values
        sheets[f"t1_{tgt}"[:31]] = t
        print(f"task1_{tgt}: {len(t)} negative conditions")

    # ---- TASK 2 ----
    for m in MODELS:
        tops = {c: top100_neg(c, m) for c in SEV}
        union = sorted(set().union(*[set(t.index) for t in tops.values()]))
        name_map = {}
        for c in SEV:
            for code, r in tops[c].iterrows():
                name_map.setdefault(code, r["PheWAS Name"])
        t = pd.DataFrame({"PheCode": union})
        t["Feature"] = t["PheCode"].map(name_map)
        for c in SEV:
            t[f"{SEV_LABEL[c]}_pvalue"] = t["PheCode"].map(tops[c]["p-val"])
        for c in SEV:
            t[f"{SEV_LABEL[c]}_LOR"] = t["PheCode"].map(tops[c]["beta"])
        pcols = [f"{SEV_LABEL[c]}_pvalue" for c in SEV]
        t["_n"] = t[pcols].notna().sum(axis=1)
        t["_minp"] = t[pcols].min(axis=1)
        t = t.sort_values(["_n", "_minp"], ascending=[False, True]).drop(columns=["_n", "_minp"])
        sheets[f"t2_{MK[m]}"] = t
        n3 = int((t[pcols].notna().sum(axis=1) == 3).sum())
        print(f"task2_{MK[m]}: {len(t)} negative features ({n3} in all 3 severities)")

    with pd.ExcelWriter(OUTXLSX, engine="openpyxl") as xl:
        for sh, df in sheets.items():
            df.to_excel(xl, sheet_name=sh, index=False)
    print(f"\nWrote {OUTXLSX}")
    U.log_milestone("v2 Step 7 negative-association tables -> all_tables_negative.xlsx.")


if __name__ == "__main__":
    main()
