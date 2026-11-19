# %% [markdown]
# v2 Step 4 - Post-process the 20 L1 runs (5 contrasts x 4 models) into analysis-ready tables.
# Writes (summary/):
#   phewas_results_combined.csv  - long: contrast x model x phecode (OR, CI, p, FDR/Bon, unstable)
#   adjustment_table.csv         - per contrast x phecode: crude OR -> adjusted ORs (m1..m4) side by side
#   dose_response.csv            - fully-adjusted (m4) OR by severity (mild/mod/severe + ge5/ge15)
#   README.md
# Crude OR = model m1; adjusted ORs = m2 (+demo), m3 (+smk/ses/window), m4 (+bmi).

# %%
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import utils as U

MODEL_ORDER = ["m1_crude", "m2_demo", "m3_smk_ses_win", "m4_bmi"]
MODEL_LABEL = {"m1_crude": "crude", "m2_demo": "+age,sex,race,year",
               "m3_smk_ses_win": "+smoking,SES,obs-window", "m4_bmi": "+BMI"}


def bon(p):
    p = np.asarray(p, float); n = np.isfinite(p).sum()
    return 0.05 / n if n else np.nan


def fdr(p, a=0.05):
    sn = np.sort(np.asarray(p, float)); sn = sn[np.isfinite(sn)]
    if not len(sn):
        return np.nan
    ok = sn <= a * (np.arange(1, len(sn) + 1) / len(sn))
    return sn[ok].max() if ok.any() else 0.0


def load(work, name):
    f = work / f"regressions_{name}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, skiprows=1, dtype={"PheWAS Code": str})
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.dropna(subset=["p-val", "beta"]).copy()
    ci = df["Conf-interval beta"].astype(str).str.strip("[]").str.split(",", expand=True)
    df["OR"] = np.exp(df["beta"].clip(-30, 30))
    df["OR_low"] = np.exp(pd.to_numeric(ci[0], errors="coerce").clip(-30, 30))
    df["OR_high"] = np.exp(pd.to_numeric(ci[1], errors="coerce").clip(-30, 30))
    df["neglog10p"] = -np.log10(df["p-val"].clip(lower=1e-300))
    bt, ft = bon(df["p-val"]), fdr(df["p-val"])
    df["sig_bon"], df["sig_fdr"] = df["p-val"] < bt, df["p-val"] < ft
    df["unstable"] = (df["beta"].abs() > 10) | (pd.to_numeric(df["std_error"], errors="coerce") > 10)
    return df


def main(work_name="pyphewas"):
    work = U.V2_RESULTS / work_name
    spec = json.loads((work / "model_spec.json").read_text())
    runs = spec["runs"]
    U.V2_SUMMARY.mkdir(parents=True, exist_ok=True)

    data = {}  # (contrast, model) -> df
    frames = []
    for r in runs:
        d = load(work, r["name"])
        if d is None:
            continue
        d["contrast"], d["model"] = r["contrast"], r["model"]
        data[(r["contrast"], r["model"])] = d
        frames.append(d)
    keep = ["contrast", "model", "PheWAS Code", "PheWAS Name", "category_string", "note",
            "beta", "std_error", "OR", "OR_low", "OR_high", "p-val", "neglog10p",
            "sig_bon", "sig_fdr", "unstable", "ICD-9", "ICD-10"]
    combined = pd.concat(frames, ignore_index=True)[lambda x: [c for c in keep if c in x.columns]]
    combined.to_csv(U.V2_SUMMARY / "phewas_results_combined.csv", index=False)

    contrasts = ["ahi_ge5", "ahi_ge15", "mild_vs_none", "moderate_vs_none", "severe_vs_none"]

    # ---- adjustment table: per contrast x phecode, OR across the 4 models ----
    adj_rows = []
    for con in contrasts:
        base = data.get((con, "m4_bmi"))
        if base is None:
            continue
        t = base[["PheWAS Code", "PheWAS Name", "category_string", "OR", "OR_low", "OR_high",
                  "p-val", "sig_fdr", "unstable"]].rename(
            columns={"OR": "OR_m4_bmi", "OR_low": "OR_low_m4", "OR_high": "OR_high_m4",
                     "p-val": "p_m4", "sig_fdr": "sig_fdr_m4", "unstable": "unstable_m4"})
        for m in ["m1_crude", "m2_demo", "m3_smk_ses_win"]:
            dm = data.get((con, m))
            if dm is not None:
                t = t.merge(dm[["PheWAS Code", "OR"]].rename(columns={"OR": f"OR_{m}"}),
                            on="PheWAS Code", how="left")
        t.insert(0, "contrast", con)
        adj_rows.append(t)
    adj = pd.concat(adj_rows, ignore_index=True)
    ordered = ["contrast", "PheWAS Code", "PheWAS Name", "category_string",
               "OR_m1_crude", "OR_m2_demo", "OR_m3_smk_ses_win", "OR_m4_bmi",
               "OR_low_m4", "OR_high_m4", "p_m4", "sig_fdr_m4", "unstable_m4"]
    adj[[c for c in ordered if c in adj.columns]].sort_values(["contrast", "p_m4"]).to_csv(
        U.V2_SUMMARY / "adjustment_table.csv", index=False)

    # ---- dose-response (fully-adjusted m4): OR by severity ----
    wide = None
    for con, suf in [("mild_vs_none", "mild"), ("moderate_vs_none", "mod"), ("severe_vs_none", "sev"),
                     ("ahi_ge5", "ge5"), ("ahi_ge15", "ge15")]:
        d = data.get((con, "m4_bmi"))
        if d is None:
            continue
        s = d[["PheWAS Code", "PheWAS Name", "category_string", "OR", "p-val", "sig_fdr"]].rename(
            columns={"OR": f"OR_{suf}", "p-val": f"p_{suf}", "sig_fdr": f"fdr_{suf}"})
        k = ["PheWAS Code", "PheWAS Name", "category_string"]
        wide = s if wide is None else wide.merge(s, on=k, how="outer")
    if wide is not None:
        wide.sort_values("p_sev", na_position="last").to_csv(U.V2_SUMMARY / "dose_response.csv", index=False)

    # ---- README ----
    L = ["# ICD-analysis PheWAS v2 results\n",
         "**Cohort:** OSA-dx-naive at index (prevalent OSA dx G47.3x/327.2x excluded), "
         ">=1 ICD encounter >=1 yr pre-index, >=2 ICD encounters. Engine: **L1** (pyPheWAS default).\n",
         "**4 sequential models** (per contrast): " +
         "; ".join(f"{k}={MODEL_LABEL[k]}" for k in MODEL_ORDER) + ".\n",
         "Crude OR = m1; adjusted ORs = m2-m4 (see `adjustment_table.csv`).\n"]
    for con in contrasts:
        d = data.get((con, "m4_bmi"))
        if d is None:
            continue
        nf = int(d["sig_fdr"].sum())
        L.append(f"\n## {con} (fully adjusted m4; tested {len(d):,} phecodes; FDR-sig {nf})")
        top = d[d["sig_fdr"] & ~d["unstable"]].sort_values("p-val").head(12)
        if len(top):
            L.append("| PheCode | Phenotype | OR (95% CI) | p |")
            L.append("|---|---|---|---|")
            for _, r in top.iterrows():
                L.append(f"| {r['PheWAS Code']} | {r['PheWAS Name']} | "
                         f"{r['OR']:.2f} ({r['OR_low']:.2f}-{r['OR_high']:.2f}) | {r['p-val']:.1e} |")
        else:
            L.append("(no FDR-significant phecodes)")
    L += ["\n## Notes / limitations",
          "- L1-regularized logit (pyPheWAS default; shrunk OR, non-classical CI) per user request.",
          "- Prevalent OSA excluded -> cohort is OSA-dx-naive at index (removes the OSA-dx positive control).",
          "- Income = h_med_income_gp categorical (Low/Mid/High, ref Mid) with Missing as its own level "
          "(no imputation; all data kept; missingness treated as informative).",
          "- obs_window = years from first pre-index ICD to index; smoking = SMOKING_STATUS (Never ref).",
          "- enc/yr computed for the cohort but NOT a model covariate (per user)."]
    (U.V2_SUMMARY / "README.md").write_text("\n".join(L), encoding="utf-8")

    print(f"combined {len(combined):,} rows ({len(data)} contrast x model sets)")
    for con in contrasts:
        bits = [f"{m.split('_')[0]}:{int(data[(con,m)]['sig_fdr'].sum()) if (con,m) in data else '-'}"
                for m in MODEL_ORDER]
        print(f"  {con} FDR-sig by model: " + " ".join(bits))
    if work_name == "pyphewas":
        U.log_milestone("v2 Step 4 post-processing complete (combined + adjustment + dose-response).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pyphewas")
