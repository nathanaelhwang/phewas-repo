# %% [markdown]
# v2 Step 2 - Build pyPheWAS group files + matched phenotype subsets + 20-run spec.
# 5 contrasts x 4 sequential L1 models. Covariates pre-encoded (continuous standardized on the
# full v2 cohort; categoricals as 0/1 dummies). Models:
#   M1 crude        : target
#   M2 demo         : + age + sex + race + index-year
#   M3 +smk/ses/win : M2 + smoking + H_MED_INCOME(+missing) + obs_window
#   M4 +bmi         : M3 + bmi
# Writes into results/icd_analysis_v2_results/pyphewas/ (or .../pyphewas_smoke for a sample).
#
# Usage: python v2_02_build_group_files.py [sample_size]

# %%
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import duckdb

import utils as U

COHORT = U.V2_COHORT / "cohort.parquet"
ICDS_ALL = (U.PHENO_DIR / "icds_all.csv").as_posix()  # v1 phenome; filtered to v2 ids below


def safe(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(s).strip().lower()).strip("_")


def main(sample_n: int | None = None) -> None:
    work = U.V2_PYPHEWAS if sample_n is None else (U.V2_RESULTS / "pyphewas_smoke")
    work.mkdir(parents=True, exist_ok=True)

    c = pd.read_parquet(COHORT)
    if sample_n is not None:
        per = max(100, sample_n // 4)
        c = pd.concat([g.sample(min(len(g), per), random_state=7)
                       for _, g in c.groupby("severity")]).reset_index(drop=True)
        print(f"[SMOKE] {len(c):,} subjects")

    cov = pd.DataFrame({"id": c["id"].astype(str)})
    for col in ["age", "bmi", "obs_window"]:
        x = c[col].astype(float)
        cov[col] = (x - x.mean()) / x.std()
    cov["sex_M"] = c["sex_M"].astype(int).values
    cov["MaxAgeAtVisit"] = c["age"].astype(float).round(2).values  # required by pyPheWAS; unused

    race_ref = c["RACE_ETH"].value_counts().idxmax()
    race_cols = []
    for lab in sorted(c["RACE_ETH"].unique()):
        if lab == race_ref:
            continue
        cov[f"race_{safe(lab)}"] = (c["RACE_ETH"] == lab).astype(int).values
        race_cols.append(f"race_{safe(lab)}")

    years = sorted(c["study_year"].unique())
    year_cols = []
    for y in years[1:]:
        cov[f"year_{int(y)}"] = (c["study_year"] == y).astype(int).values
        year_cols.append(f"year_{int(y)}")

    smoke_cols = []
    for lab in ["Former", "Current", "Unknown"]:
        cov[f"smoke_{safe(lab)}"] = (c["smoking_status"] == lab).astype(int).values
        smoke_cols.append(f"smoke_{safe(lab)}")

    # income-group dummies (ref = Mid; Missing is its own informative level, no imputation)
    income_cols = []
    for lab, name in [("Low", "income_low"), ("High", "income_high"), ("Missing", "income_missing")]:
        cov[name] = (c["income_gp"] == lab).astype(int).values
        income_cols.append(name)

    # guarantee NO covariate is NaN -> every model uses the same subject set per contrast
    assert cov.drop(columns="id").isna().sum().sum() == 0, "unexpected NaN in covariates"

    demo = ["age", "sex_M"] + race_cols + year_cols
    m3 = demo + smoke_cols + income_cols + ["obs_window"]
    m4 = m3 + ["bmi"]
    MODELS = {"m1_crude": "", "m2_demo": "+".join(demo),
              "m3_smk_ses_win": "+".join(m3), "m4_bmi": "+".join(m4)}

    full = cov.copy()
    full["ahi_ge5"] = (c["AHI"] >= 5).astype(int).values
    full["ahi_ge15"] = (c["AHI"] >= 15).astype(int).values
    full.to_csv(work / "group_full.csv", index=False)

    contrasts = {"mild_vs_none": "Mild", "moderate_vs_none": "Moderate", "severe_vs_none": "Severe"}
    for nm, sev in contrasts.items():
        mask = c["severity"].isin(["None", sev]).values
        g = cov[mask].copy()
        g.insert(1, "genotype", (c.loc[mask, "severity"] == sev).astype(int).values)
        g.to_csv(work / f"group_{nm}.csv", index=False)

    con = duckdb.connect(); con.execute("PRAGMA threads=8")

    def write_pheno(out_name, id_series):
        con.register("keep_ids", pd.DataFrame({"id": id_series.astype(str).values}))
        out = (work / out_name).as_posix()
        con.execute(f"""COPY (SELECT a.id,a.ICD_CODE,a.ICD_TYPE,a.AgeAtICD
            FROM read_csv('{ICDS_ALL}', header=true,
                 columns={{'id':'VARCHAR','ICD_CODE':'VARCHAR','ICD_TYPE':'INTEGER','AgeAtICD':'DOUBLE'}}) a
            WHERE a.id IN (SELECT id FROM keep_ids)) TO '{out}' (HEADER, DELIMITER ',')""")
        con.unregister("keep_ids")

    write_pheno("icds_full.csv", c["id"])
    for nm, sev in contrasts.items():
        write_pheno(f"icds_{nm}.csv", c.loc[c["severity"].isin(["None", sev]), "id"])

    contrast_defs = [("ahi_ge5", "ahi_ge5", "group_full.csv", "icds_full.csv"),
                     ("ahi_ge15", "ahi_ge15", "group_full.csv", "icds_full.csv"),
                     ("mild_vs_none", "genotype", "group_mild_vs_none.csv", "icds_mild_vs_none.csv"),
                     ("moderate_vs_none", "genotype", "group_moderate_vs_none.csv", "icds_moderate_vs_none.csv"),
                     ("severe_vs_none", "genotype", "group_severe_vs_none.csv", "icds_severe_vs_none.csv")]
    runs = []
    for cname, target, group, pheno in contrast_defs:
        for mkey, mcov in MODELS.items():
            runs.append({"name": f"{cname}__{mkey}", "contrast": cname, "model": mkey,
                         "target": target, "group": group, "pheno": pheno,
                         "covariates": mcov, "engine": "l1"})
    (work / "model_spec.json").write_text(json.dumps(
        {"models": MODELS, "race_ref": race_ref, "year_ref": int(years[0]), "smoke_ref": "Never",
         "runs": runs}, indent=2), encoding="utf-8")

    print(f"work {work.name}; cohort {len(c):,}; race ref {race_ref}; year ref {years[0]}")
    print(f"group_full {len(full):,}; contrasts: " +
          ", ".join(f"{nm} {len(pd.read_csv(work / f'group_{nm}.csv')):,}" for nm in contrasts))
    print(f"{len(runs)} runs (5 contrasts x 4 models) -> {work / 'model_spec.json'}")
    if sample_n is None:
        U.log_milestone(f"v2 Step 2 group files + spec: {len(runs)} L1 runs.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
