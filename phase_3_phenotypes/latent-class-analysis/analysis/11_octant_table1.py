"""Stage 11 -- baseline characteristics (Table 1) for the eight octant phenotypes.

Same construction as stage 9, which built Table 1 for the six corner archetypes, with two
differences that follow from what the octants are:

  * there is no `unclassified` column. The octants partition the cohort, so every
    phenotyped patient appears in exactly one group and the columns sum to n.
  * eight groups means 28 pairwise contrasts behind `max |SMD|` rather than 15, so the
    maximum is taken over a larger set and will run slightly higher for the same
    underlying imbalance. It is still the right statistic to read -- at this n every
    p-value is significant and says nothing.

Age, sex, BMI and race/ethnicity are not in `latent-class-analysis/data/`; they come from
the study spine the rest of the project uses (`icd_analysis_v2_results/cohort/
cohort.parquet`, one row per patient). That spine carries its own exclusions -- adults
only, study year >= 2010, test/invalid records removed, complete-case on
age/sex/BMI/encounters -- so it does not cover every phenotyped patient, and coverage is
reported per octant and tested for differential loss. AHI is this analysis's own
`index_ahi`, so the table is consistent with every other figure here.

PHI: this script writes a per-patient assignment file but never prints, samples or
aggregates any identifier. Every printed number is a group-level statistic that also
appears in the published table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import RESULTS  # noqa: E402

SPINE = Path(r"C:\Users\natha\Projects\work\phewas\results\icd_analysis_v2_results"
             r"\cohort\cohort.parquet")
OUT = RESULTS / "phenotypes"
# lattice order: none high, then each single axis, then each pair, then all three
BITS = {"mild-all": (0, 0, 0), "hypoxemia-predominant": (1, 0, 0),
        "symptom-predominant": (0, 1, 0), "comorbidity-predominant": (0, 0, 1),
        "hypoxemic-symptomatic": (1, 1, 0), "hypoxemic-comorbid": (1, 0, 1),
        "symptomatic-comorbid": (0, 1, 1), "high-all": (1, 1, 1)}
ORDER = list(BITS)

ph = pd.read_csv(OUT / "cross_domain_phenotypes.csv", dtype={"MRN": str})
spine = pd.read_parquet(SPINE)[["id", "age", "sex_M", "bmi", "RACE_ETH"]]
spine["id"] = spine.id.astype(str)
d = ph.merge(spine, left_on="MRN", right_on="id", how="left")
d["octant"] = pd.Categorical(d.octant, categories=ORDER, ordered=True)
d["has_demo"] = d.age.notna()
d["sex"] = d.sex_M.map({1: "Male", 0: "Female"})
assert d.octant.notna().all(), "every patient must fall in exactly one octant"

# --- coverage, and whether the loss is differential ------------------------------
cov = (d.groupby("octant", observed=True)
       .agg(n_phenotyped=("MRN", "size"), n_with_demographics=("has_demo", "sum")))
cov["coverage"] = (cov.n_with_demographics / cov.n_phenotyped).round(4)
chi2, p_cov, _, _ = stats.chi2_contingency(pd.crosstab(d.octant, d.has_demo))
print("demographic coverage by octant:")
print(cov.to_string())
print(f"\ncoverage differs across octants: chi2 = {chi2:,.1f}, p = {p_cov:.2e}, "
      f"range {cov.coverage.min():.1%}-{cov.coverage.max():.1%}")

an = d[d.has_demo].copy()
CONT = [("age", "Age, years"), ("bmi", "BMI, kg/m2"), ("index_ahi", "AHI, events/h")]
CAT = [("sex", "Gender"), ("RACE_ETH", "Race / ethnicity")]
groups = [g for g in ORDER if (an.octant == g).any()]


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


rows = [{"Characteristic": "n", "Overall": f"{len(an):,}",
         **{g: f"{(an.octant == g).sum():,}" for g in groups},
         "p": "", "max |SMD|": ""}]

for col, label in CONT:
    vals = {g: an.loc[an.octant == g, col].dropna() for g in groups}
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
    p = stats.chi2_contingency(pd.crosstab(sub[col], sub.octant))[1]
    if col == "sex":
        cats = ["Female", "Male"]
    else:                                    # largest category first, group-level counts
        cats = sub.groupby(col, observed=True).size().sort_values(
            ascending=False).index.tolist()
    for ci, c in enumerate(cats):
        cell, props = {}, {}
        for g in groups:
            gg = sub[sub.octant == g]
            n = int((gg[col] == c).sum())
            props[g] = n / len(gg) if len(gg) else np.nan
            cell[g] = f"{n:,} ({props[g]:.1%})"
        n_all = int((sub[col] == c).sum())
        mx = max((smd_prop(props[a], props[b]) for i, a in enumerate(groups)
                  for b in groups[i + 1:]), default=np.nan)
        rows.append({"Characteristic": f"    {c}",
                     "Overall": f"{n_all:,} ({n_all/len(sub):.1%})", **cell,
                     "p": f"{p:.2e}" if ci == 0 else "", "max |SMD|": f"{mx:.2f}"})

t1 = pd.DataFrame(rows)[["Characteristic", "Overall"] + groups + ["p", "max |SMD|"]]
t1.to_csv(OUT / "octant_table1_baseline_characteristics.csv", index=False)
cov.to_csv(OUT / "octant_table1_coverage.csv")

md = ["# Table 1 — baseline characteristics by octant phenotype", "",
      f"Analytic set: {len(an):,} of {len(d):,} phenotyped patients "
      f"({len(an)/len(d):.1%}) with demographics available on the study spine. "
      "The octants partition the cohort, so unlike the corner-archetype table there is "
      "no `unclassified` column and the group columns sum to n.", "",
      t1.to_markdown(index=False), "",
      "## Demographic coverage by octant", "",
      cov.assign(coverage=lambda x: (x.coverage * 100).round(1).astype(str) + "%")
        .to_markdown(), "",
      f"Coverage differs across octants (chi-square {chi2:,.1f}, p = {p_cov:.2e}; "
      f"range {cov.coverage.min():.1%}–{cov.coverage.max():.1%}). The spine excludes "
      "under-18s, studies before 2010, test/invalid records, and anyone missing age, "
      "sex, BMI or encounter counts, so this is a complete-case table.", "",
      "Continuous variables are mean (SD); categorical are n (%). Gender is given as "
      "n (%) rather than mean (SD) — for a binary variable the mean *is* the proportion, "
      "so a mean and SD would restate the same number twice. `max |SMD|` is the largest "
      "of the 28 pairwise standardised differences across the eight octants; at this "
      "sample size p-values are uninformative (everything is significant), so read the "
      "SMD — above 0.1 is a real imbalance, above 0.5 a large one. Note that eight "
      "groups give 28 pairs against the archetype table's 15, so the maximum is taken "
      "over a larger set and runs slightly higher for the same underlying imbalance.", ""]
(OUT / "OCTANT_TABLE1.md").write_text("\n".join(md), encoding="utf-8")

# --- per-patient assignment file --------------------------------------------------
# Written, never inspected: no head(), no sampling, no per-identifier aggregation.
asg = d[["MRN", "c1_physiologic", "score_c1", "c2_symptomatic", "score_c2",
         "c3_comorbidities", "score_c3", "octant", "archetype", "index_ahi"]].copy()
for i, s in enumerate(("score_c1", "score_c2", "score_c3")):
    asg[f"hi_{s.split('_')[1]}"] = [BITS[o][i] for o in asg.octant]
asg = asg[["MRN", "octant", "hi_c1", "hi_c2", "hi_c3", "c1_physiologic", "score_c1",
           "c2_symptomatic", "score_c2", "c3_comorbidities", "score_c3",
           "archetype", "index_ahi"]]
asg.to_csv(OUT / "octant_assignments.csv", index=False)

print("\n" + t1.to_string(index=False))
print(f"\nwrote OCTANT_TABLE1.md, octant_table1_baseline_characteristics.csv, "
      f"octant_table1_coverage.csv")
print(f"wrote octant_assignments.csv — {len(asg):,} rows x {asg.shape[1]} cols, "
      f"{asg.MRN.nunique():,} unique identifiers, {int(asg.MRN.isna().sum())} missing")
