"""Stage 9 -- baseline characteristics (Table 1) for the cross-domain phenotypes.

Age, sex, BMI and race/ethnicity are NOT in `latent-class-analysis/data/`, so they are
joined from the study spine the rest of the project already uses,
`results/icd_analysis_v2_results/cohort/cohort.parquet` (one row per patient, id =
12-char MRN). That spine carries its own exclusions -- adults only, study year >= 2010,
test/invalid patients removed, and complete-case on age/sex/BMI/encounters -- so it
does not cover every phenotyped patient. Coverage is reported per phenotype and tested
for differential loss, because a Table 1 built on a differentially-covered subset is
misleading in a way that is invisible once the table is formatted.

AHI comes from this analysis's own `index_ahi` (the value used throughout), not the
spine's, so the table is internally consistent with every other figure here.

Continuous variables: mean (SD). Categorical: n (%). Gender is reported as n (%) like
any other categorical -- a mean and SD of a binary is just the proportion and its
Bernoulli SD, which says nothing extra.

Because n is large enough that any difference reaches significance, the table reports
the **maximum pairwise standardised difference** across phenotypes as the effect size
alongside the omnibus p-value; |d| > 0.1 is the usual threshold for a real imbalance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import RESULTS, WORK  # noqa: E402

SPINE = Path(r"C:\Users\natha\Projects\work\phewas\results\icd_analysis_v2_results"
             r"\cohort\cohort.parquet")
OUT = RESULTS / "phenotypes"
ORDER = ["triple-low", "symptomatic-without-hypoxemia", "comorbid-without-hypoxemia",
         "severe-but-asymptomatic", "severe-symptomatic-otherwise-well", "triple-high",
         "unclassified"]

ph = pd.read_csv(OUT / "cross_domain_phenotypes.csv", dtype={"MRN": str})
spine = pd.read_parquet(SPINE)[["id", "age", "sex_M", "bmi", "RACE_ETH"]]
spine["id"] = spine.id.astype(str)
d = ph.merge(spine, left_on="MRN", right_on="id", how="left")
d["archetype"] = pd.Categorical(d.archetype, categories=ORDER, ordered=True)
d["has_demo"] = d.age.notna()
d["sex"] = d.sex_M.map({1: "Male", 0: "Female"})

# --- coverage, and whether the loss is differential ------------------------------
cov = (d.groupby("archetype", observed=True)
       .agg(n_phenotyped=("MRN", "size"), n_with_demographics=("has_demo", "sum")))
cov["coverage"] = (cov.n_with_demographics / cov.n_phenotyped).round(4)
tab = pd.crosstab(d.archetype, d.has_demo)
chi2, p_cov, _, _ = stats.chi2_contingency(tab)
print("demographic coverage by phenotype:")
print(cov.to_string())
print(f"\ncoverage differs across phenotypes: chi2 = {chi2:,.0f}, p = {p_cov:.2e}, "
      f"range {cov.coverage.min():.1%}-{cov.coverage.max():.1%}")

an = d[d.has_demo].copy()
CONT = [("age", "Age, years"), ("bmi", "BMI, kg/m2"), ("index_ahi", "AHI, events/h")]
CAT = [("sex", "Gender"), ("RACE_ETH", "Race / ethnicity")]
groups = [g for g in ORDER if (an.archetype == g).any()]


def smd_cont(x, y):
    """Standardised mean difference, pooled SD."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return abs(x.mean() - y.mean()) / sp if sp > 0 else np.nan


def smd_prop(p1, p2):
    pb = (p1 + p2) / 2
    s = np.sqrt(pb * (1 - pb))
    return abs(p1 - p2) / s if s > 0 else np.nan


rows = []
rows.append({"Characteristic": "n", "Overall": f"{len(an):,}",
             **{g: f"{(an.archetype == g).sum():,}" for g in groups},
             "p": "", "max |SMD|": ""})

for col, label in CONT:
    vals = {g: an.loc[an.archetype == g, col].dropna() for g in groups}
    cell = {g: f"{v.mean():.1f} ({v.std():.1f})" for g, v in vals.items()}
    p = stats.f_oneway(*[v for v in vals.values() if len(v) > 1]).pvalue
    mx = max((smd_cont(vals[a], vals[b]) for i, a in enumerate(groups)
              for b in groups[i + 1:]), default=np.nan)
    rows.append({"Characteristic": f"{label}, mean (SD)",
                 "Overall": f"{an[col].mean():.1f} ({an[col].std():.1f})",
                 **cell, "p": f"{p:.2e}", "max |SMD|": f"{mx:.2f}"})

for col, label in CAT:
    rows.append({"Characteristic": f"{label}, n (%)", "Overall": "",
                 **{g: "" for g in groups}, "p": "", "max |SMD|": ""})
    sub = an[an[col].notna()]
    tabc = pd.crosstab(sub[col], sub.archetype)
    p = stats.chi2_contingency(tabc)[1]
    cats = (["Female", "Male"] if col == "sex"
            else sub[col].value_counts().index.tolist())
    for ci, c in enumerate(cats):
        cell = {}
        props = {}
        for g in groups:
            gg = sub[sub.archetype == g]
            n = int((gg[col] == c).sum())
            pr = n / len(gg) if len(gg) else np.nan
            props[g] = pr
            cell[g] = f"{n:,} ({pr:.1%})"
        n_all = int((sub[col] == c).sum())
        mx = max((smd_prop(props[a], props[b]) for i, a in enumerate(groups)
                  for b in groups[i + 1:]), default=np.nan)
        rows.append({"Characteristic": f"    {c}",
                     "Overall": f"{n_all:,} ({n_all/len(sub):.1%})", **cell,
                     "p": f"{p:.2e}" if ci == 0 else "", "max |SMD|": f"{mx:.2f}"})

t1 = pd.DataFrame(rows)[["Characteristic", "Overall"] + groups + ["p", "max |SMD|"]]
t1.to_csv(OUT / "table1_baseline_characteristics.csv", index=False)
cov.to_csv(OUT / "table1_coverage.csv")

md = ["# Table 1 — baseline characteristics by cross-domain phenotype", "",
      f"Analytic set: {len(an):,} of {len(d):,} phenotyped patients "
      f"({len(an)/len(d):.1%}) with demographics available on the study spine.", "",
      t1.to_markdown(index=False), "",
      "## Demographic coverage by phenotype", "",
      cov.assign(coverage=lambda x: (x.coverage * 100).round(1).astype(str) + "%")
        .to_markdown(), "",
      f"Coverage differs across phenotypes (chi-square {chi2:,.0f}, p = {p_cov:.2e}; "
      f"range {cov.coverage.min():.1%}–{cov.coverage.max():.1%}). The spine excludes "
      "under-18s, studies before 2010, test/invalid records, and anyone missing "
      "age, sex, BMI or encounter counts, so this is a complete-case table and the "
      "phenotypes are not equally represented in it.", "",
      "Continuous variables are mean (SD); categorical are n (%). Gender is given as "
      "n (%) rather than mean (SD) — for a binary variable the mean *is* the "
      "proportion, so a mean and SD would restate the same number twice. "
      "`max |SMD|` is the largest pairwise standardised difference across phenotypes; "
      "at this sample size p-values are uninformative (everything is significant), so "
      "read the SMD — above 0.1 is a real imbalance, above 0.5 a large one.", ""]
(OUT / "TABLE1.md").write_text("\n".join(md), encoding="utf-8")

print("\n" + t1.to_string(index=False))
print(f"\nwrote {OUT / 'TABLE1.md'} and table1_baseline_characteristics.csv")
