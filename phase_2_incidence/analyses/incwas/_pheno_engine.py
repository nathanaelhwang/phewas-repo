"""Shared helpers for the PHENOTYPE-exposure incidence WAS.

The existing incwas pipeline estimates OSA severity -> incident phecodes. This module
swaps the exposure for the cross-domain latent phenotypes while reusing everything that
is exposure-agnostic: the dated phecode event store, the follow-up/censoring frame, the
harmonized control-exclusion washout, and the Cox fitting helpers in `_cox_engine`.
Nothing in `_cox_engine` or the numbered 00-14 scripts is modified.

Three things differ from the severity analysis, each forced by the data:

1. **The cohort is smaller, later and shorter-followed.** Only 25,380 of the 70,880
   phenotyped patients reach the incwas spine (the phenotype needs the c2 questionnaire,
   which is recent), 95% index in 2020-2023, and mean follow-up is 1.42 y against ~2.9 y
   for the full spine. Study-year dummies are therefore near-degenerate and are replaced
   by a centred continuous year.

2. **Sleep phecodes cannot be outcomes.** `327.3 Sleep apnea` is the largest "incident"
   code in this cohort (19,195 events, 76% of patients) because every patient had a
   diagnostic sleep study at index -- that is the exposure being written into the chart,
   not new disease. The whole 327.x family is excluded: it also carries insomnia and
   restless legs, which are constituents of the c2 symptom axis.

3. **Baseline encounter rate is no longer a clean covariate.** In the severity analysis
   `enc_rate_z` was a detection-bias control. Here it correlates 0.44 with score_c3,
   because the c3 axis IS coding volume (its classes order 13/46/101 distinct codes).
   Adjusting for it removes part of the exposure's own definition; not adjusting leaves
   surveillance bias. Both are run and reported side by side rather than silently
   choosing -- see ADJUSTMENT_SETS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incwas_utils as U
import _cox_engine as E

PHENO_CSV = (U.PROJECT / "latent-class-analysis" / "results" / "phenotypes"
             / "cross_domain_phenotypes.csv")
OUT = U.RESULTS / "phenotype_exposure"

# --- exposure specification ---------------------------------------------------
ARCHETYPES = ["symptomatic-without-hypoxemia", "comorbid-without-hypoxemia",
              "severe-but-asymptomatic", "severe-symptomatic-otherwise-well",
              "triple-high", "triple-low"]
ARCH_REF = "unclassified"          # n=19,149; the middle of the phenotype space
SCORES = ["score_c1", "score_c2", "score_c3"]
DOMAIN_LABELS = {"c1_physiologic": ["mild-hypoxemia", "moderate-hypoxemia",
                                    "severe-hypoxemia"],
                 "c2_symptomatic": ["minimally-symptomatic", "moderate-symptom",
                                    "high-symptom"],
                 "c3_comorbidities": ["low-burden", "moderate-burden", "high-burden"]}
DOMAIN_REF = {"c1_physiologic": "mild-hypoxemia",
              "c2_symptomatic": "minimally-symptomatic",
              "c3_comorbidities": "low-burden"}

# --- outcomes that are definitionally downstream of having had a sleep study ----
def is_circular_outcome(phecode: str) -> bool:
    """Sleep-disorder phecodes (327.x) are circular with the exposure.

    Every patient in this cohort was referred for and received a diagnostic sleep study
    at the index date, so a post-index sleep-apnea code records the index event rather
    than a new diagnosis. 327.4x insomnia and 327.7x restless legs are additionally
    constituents of the c2 symptom axis being used as the exposure.
    """
    p = str(phecode)
    return (p == "327" or p.startswith("327.")        # sleep disorders incl. apnea
            or p == "347" or p.startswith("347."))    # narcolepsy/cataplexy: also a PSG result


LCA = U.PROJECT / "latent-class-analysis" / "results"
GRADIENT = {"c1_physiologic": ["mild-hypoxemia", "moderate-hypoxemia", "severe-hypoxemia"],
            "c2_symptomatic": ["minimally-symptomatic", "moderate-symptom", "high-symptom"],
            "c3_comorbidities": ["low-burden", "moderate-burden", "high-burden"]}


def domain_scores(domains=("c1_physiologic", "c3_comorbidities")) -> pd.DataFrame:
    """Per-domain gradient scores, read from each domain's OWN assignment file.

    This is the key to the cohort size. The composite archetype label needs all three
    domains, and the c2 questionnaire is the small one, so the 3-domain intersection is
    only 25,380 patients on the incwas spine with 1.42 y mean follow-up. But the SCORES
    are per-domain quantities: c1 alone covers 239,248 patients and c3 covers 235,861,
    so c1 x c3 reaches **112,555 patients / 328,352 person-years / 2.92 y mean
    follow-up** -- 9x the person-time. Reconstructing them here rather than reading the
    3-domain composite file is what buys that (verified identical to the composite,
    max |diff| 4.4e-16).

    Class numbers are assigned by DESCENDING SIZE, not severity, so the rank map comes
    from CLASS_LABELS.json rather than from the class index.
    """
    import json
    labels = json.loads((LCA / "CLASS_LABELS.json").read_text())
    out = None
    for dom in domains:
        pre = dom.split("_")[0]
        a = pd.read_csv(LCA / dom / f"{pre}_final_assignments.csv", dtype={"MRN": str})
        K = labels[dom]["K"]
        names = [labels[dom][str(k)] for k in range(K)]
        rank = np.array([GRADIENT[dom].index(n) for n in names], float)
        P = a[[f"p_class{k}" for k in range(K)]].to_numpy()
        s = pd.DataFrame({"MRN": a.MRN, f"score_{pre}": P @ rank,
                          dom: a.class_label, f"conf_{pre}": P.max(1)})
        out = s if out is None else out.merge(s, on="MRN", how="inner")
    return out


def load_exposure(domains=("c1_physiologic", "c2_symptomatic", "c3_comorbidities"),
                  with_archetype=True) -> pd.DataFrame:
    """Phenotype exposure joined to the incwas follow-up spine, with design columns.

    ``domains`` selects which axes must be present; requiring all three restricts to the
    25,380-patient 3-domain cohort, while ("c1_physiologic", "c3_comorbidities") gives
    the 112,555-patient scan cohort. ``with_archetype`` adds the corner-phenotype label,
    which is only defined on the 3-domain intersection.

    z-scores are computed on THIS cohort, not the full spine, so a one-unit covariate
    change means the same thing as the coefficients.
    """
    f = pd.read_parquet(U.FOLLOWUP_FRAME).reset_index(drop=True)
    f["id"] = f["id"].astype(str)
    d = f.merge(domain_scores(domains), left_on="id", right_on="MRN",
                how="inner").drop(columns=["MRN"])
    if with_archetype:
        ph = pd.read_csv(PHENO_CSV, dtype={"MRN": str})
        d = d.merge(ph[["MRN", "archetype", "joint_confidence"]], left_on="id",
                    right_on="MRN", how="left").drop(columns=["MRN"])
        d["archetype"] = d["archetype"].fillna("~no3domain~")

    d["idx"] = pd.to_datetime(d["idx"])
    d["censor_date"] = pd.to_datetime(d["censor_date"])
    for c, s in (("age_z", "age"), ("bmi_z", "bmi"), ("obs_window_z", "obs_window")):
        d[c] = _z(d[s])
    d["enc_rate_z"] = _z(np.log1p(d["enc_per_year"]))
    # 95% of this cohort indexes 2020-2023, so the spine's year dummies are degenerate
    # here; a centred continuous year keeps the secular-trend adjustment identifiable.
    d["study_year_c"] = d["study_year"].astype(float) - d["study_year"].median()
    d["sex_M"] = d["sex_M"].astype(float)
    d["severity_ord"] = d["severity_ord"].astype(float)

    # age enters as a natural cubic spline: rho(age, score_c3) = 0.51 with class means
    # 43.5 / 51.7 / 63.6 years, and a single linear term cannot absorb a relationship
    # that strong without leaving residual age confounding inside the c3 estimate.
    d["age_sp1"], d["age_sp2"] = _spline(d["age"])

    if "archetype" in d.columns:
        for a in ARCHETYPES:
            d[f"arch_{_slug(a)}"] = (d["archetype"] == a).astype(float)
    for dom, levels in DOMAIN_LABELS.items():
        if dom not in d.columns:
            continue
        for lvl in levels:
            if lvl != DOMAIN_REF[dom]:
                d[f"{dom.split('_')[0]}_{_slug(lvl)}"] = (d[dom] == lvl).astype(float)

    race_ref = d["RACE_ETH"].astype("string").fillna("~NA~").value_counts().idxmax()
    race_d, race_cols = E._dummies(d["RACE_ETH"], "race", race_ref)
    smk_d, smk_cols = E._dummies(d["smoking_status"], "smk", E.SMK_REF)
    inc_d, inc_cols = E._dummies(d["income_gp"], "inc", E.INC_REF)
    d = pd.concat([d, race_d, smk_d, inc_d], axis=1)
    d.attrs["race_cols"] = race_cols
    d.attrs["smk_cols"] = smk_cols
    d.attrs["inc_cols"] = inc_cols
    return d


# --- octant phenotypes ---------------------------------------------------------
# Eight boxes formed by cutting each gradient score at its median. This replaces the
# six corner archetypes as the categorical exposure, for three reasons measured on the
# data: (1) it classifies EVERYONE -- the corner rule left 75.4% "unclassified", and the
# obvious fixes made things worse, since naming all 2-axis corners and widening the
# bands splits the cohort into 20 groups whose median arm (683) is SMALLER than the six
# corners' (855); (2) because the three axes are near-independent, median cuts give
# near-perfectly balanced octants (8,349-9,249 of 70,880; 2,737-3,527 of the
# 25,380 that reach the incwas spine) rather than the 536-9,596
# spread that collapsing the latent classes produces; (3) it removes the 19,149-patient
# "unclassified" grab-bag that was serving as the reference category.
#
# The cost is contrast: hi-vs-lo separation falls from 1.91 to 1.07 on the 0-2 scale, so
# hazard ratios shrink toward 1. sqrt(arm size) x contrast is 57.0 against 55.8 for the
# corners -- roughly a wash on power, bought with full coverage and a defensible
# reference. The cut points are cohort-derived and are recorded so the definition is
# reproducible; they are NOT transferable to another cohort unchanged.
OCTANT_NAMES = {
    (0, 0, 0): "mild-all", (1, 0, 0): "hypoxemia-predominant",
    (0, 1, 0): "symptom-predominant", (0, 0, 1): "comorbidity-predominant",
    (1, 1, 0): "hypoxemic-symptomatic", (1, 0, 1): "hypoxemic-comorbid",
    (0, 1, 1): "symptomatic-comorbid", (1, 1, 1): "high-all",
}
OCTANT_REF = "mild-all"
OCTANTS = [OCTANT_NAMES[k] for k in
           [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]]


def octant_cuts() -> dict:
    """Median cut point per axis, computed once on the FULL phenotyped cohort.

    Taken from the whole 70,880-patient phenotype table rather than from whichever
    analysis subset is in hand, so the octant a patient belongs to does not change
    depending on which model is being fitted.
    """
    ph = pd.read_csv(PHENO_CSV, dtype={"MRN": str})
    return {s: float(ph[s].median()) for s in SCORES}


def assign_octants(d: pd.DataFrame, cuts: dict | None = None) -> pd.DataFrame:
    """Add `octant` plus its 7 non-reference dummies to an exposure frame."""
    cuts = cuts or octant_cuts()
    bits = [(d[s] > cuts[s]).astype(int) for s in SCORES]
    d = d.copy()
    d["octant"] = [OCTANT_NAMES[(a, b, c)] for a, b, c in zip(*[x.to_numpy() for x in bits])]
    for o in OCTANTS:
        if o != OCTANT_REF:
            d[f"oct_{_slug(o)}"] = (d["octant"] == o).astype(float)
    return d


def exposure_blocks(d: pd.DataFrame) -> dict:
    """{name -> design columns} for each exposure specification.

    ``scores`` is the primary spec: one continuous gradient per domain, entered
    simultaneously. The axes are mutually independent (|Spearman| <= 0.04), so this
    costs one degree of freedom each instead of the five a 6-level archetype dummy set
    would, which is what makes a phecode-level scan feasible.
    """
    have = lambda cs: [c for c in cs if c in d.columns]
    return {
        "scores": have(SCORES),
        "archetype": have([f"arch_{_slug(a)}" for a in ARCHETYPES]),
        "c1": have([c for c in d.columns if c.startswith("c1_")]),
        "c2": have([c for c in d.columns if c.startswith("c2_")]),
        "c3": have([c for c in d.columns if c.startswith("c3_")]),
    }


def adjustment_sets(d: pd.DataFrame) -> dict:
    """The nested ladder, adapted to this cohort and this exposure.

    Two deliberate departures from the severity ladder:

    * **m3a / m3b split.** `enc_rate_z` correlates 0.44 with score_c3 because the c3
      axis IS coding volume (its classes order 13/46/101 distinct codes). Adjusting for
      it removes part of the exposure's own definition; omitting it leaves surveillance
      bias. The two estimands are different questions -- total effect of the burden
      phenotype including its ascertainment component (m3a) versus something closer to
      biology but conditioned on a descendant of the exposure (m3b) -- so both are
      reported at equal prominence rather than one being chosen silently.
    * **`obs_window_z` is demoted.** Baseline look-back averages 9.8 / 16.3 / 22.2 years
      across low / moderate / high burden, so it is partly a measurement artifact of c3
      rather than a confounder of it. It stays in the ladder for comparability with the
      severity analysis but is reported as part of the same conditioning-on-a-descendant
      concern.
    """
    demo = ["age_sp1", "age_sp2", "sex_M", "study_year_c"] + d.attrs["race_cols"]
    ses = d.attrs["smk_cols"] + d.attrs["inc_cols"] + ["obs_window_z"]
    return {
        "m1": [],
        "m2": demo,
        "m3a": demo + ses,                                  # no utilisation control
        "m3b": demo + ses + ["enc_rate_z"],                 # + surveillance control
        "m4": demo + ses + ["enc_rate_z", "bmi_z"],
    }


def _spline(x: pd.Series, knots=(0.1, 0.5, 0.9)) -> tuple[pd.Series, pd.Series]:
    """Natural cubic spline basis (2 columns) for a continuous covariate."""
    x = x.astype(float)
    k = np.quantile(x.dropna(), knots)
    z = (x - x.mean()) / x.std()
    kz = (k - x.mean()) / x.std()

    def d(j):
        a = np.clip(z - kz[j], 0, None) ** 3
        b = np.clip(z - kz[-1], 0, None) ** 3
        return (a - b) / (kz[-1] - kz[j])
    return z, pd.Series(d(0) - d(1), index=x.index)


# --- body-system outcomes -------------------------------------------------------
def system_events(d: pd.DataFrame, ev: pd.DataFrame, dim: pd.DataFrame,
                  system: str, rule: str = "strict") -> pd.DataFrame:
    """Survival frame for one phecode body system.

    ``strict``  at-risk = free of EVERY phecode in the system at baseline; event = first
                incident phecode in the system (rule-of-2 date). Clean "new-onset
                organ-system disease", but at-risk collapses for common systems
                (endocrine/metabolic keeps 7.9% of the cohort), so the survivors are a
                selected, unusually healthy subgroup.
    ``anynew``  everyone is at risk; event = first incident phecode in the system that
                the patient did NOT have at baseline. Keeps the whole cohort and answers
                "does this phenotype accrue new disease in this system faster", which is
                the question the burden axis actually poses.
    Circular sleep phecodes are removed from every system before either rule is applied.
    """
    cat = dict(zip(dim["phecode"].astype(str), dim["category"]))
    e = ev[ev["phecode"].map(lambda p: cat.get(str(p)) == system
                             and not is_circular_outcome(p))].copy()
    ids = set(d["id"])
    e = e[e["id"].isin(ids)]

    # "already held at baseline" uses the control-exclusion FAMILY, not the exact code.
    # The v1.2 icd9 and icd10-beta maps put the same disease on different phecodes by
    # code era, so an exact-code rule lets ICD-9-era baseline history through and
    # re-counts it as incident at system level -- the same leakage the phecode-level
    # analysis fixes with excl.
    excl = E.exclude_map()
    fam = {p: (excl.get(p, set()) | {p}) for p in e["phecode"].unique()}
    held = {}
    for i, p in zip(e.loc[e["has_pre"], "id"], e.loc[e["has_pre"], "phecode"]):
        held.setdefault(i, set()).update(fam.get(p, {p}))

    pre_ids = set(e.loc[e["has_pre"], "id"])
    post = e.dropna(subset=["second_post_date"])
    n_sys = e["phecode"].nunique()
    if rule == "strict":
        at = d[~d["id"].isin(pre_ids)].copy()
        first = (post[~post["id"].isin(pre_ids)]
                 .groupby("id")["second_post_date"].min())
    elif rule == "anynew":
        at = d.copy()
        m = [p not in held.get(i, ()) for i, p in zip(post["id"], post["phecode"])]
        first = post[m].groupby("id")["second_post_date"].min()
    else:
        raise ValueError(rule)

    # Depletion of susceptibles runs OPPOSITE to surveillance bias here: a high-burden
    # patient already holds more of the system's codes, so fewer remain available to be
    # acquired. Carrying the count of still-eligible phecodes makes the two forces
    # separable instead of netting silently.
    at["n_eligible_z"] = _z(pd.Series(
        [n_sys - len(held.get(i, ())) for i in at["id"]], index=at.index))
    at["event_date"] = pd.to_datetime(at["id"].map(first))
    is_ev = at["event_date"].notna() & (at["event_date"] <= at["censor_date"])
    at["event"] = is_ev.astype(int)
    end = at["censor_date"].where(~is_ev, at["event_date"])
    at["time"] = (end - at["idx"]).dt.days / 365.25
    return at[at["time"] > 0].copy()


# --- cumulative incidence -------------------------------------------------------
def event_time_grid(t, event, horizon=3.0) -> np.ndarray:
    """Every incident-event time within the plot horizon, plus annual checkpoints."""
    t = np.asarray(t, dtype=float)
    incident = t[(np.asarray(event) == 1) & (t > 0) & (t <= horizon)]
    return np.unique(np.r_[0.0, np.arange(1.0, horizon), horizon, incident])


def aalen_johansen(t, event, is_death, grid) -> np.ndarray:
    """Aalen-Johansen cumulative incidence of `event`, death competing, evaluated on `grid`.

    Tie-aware and permutation-invariant. Every transition sharing a time uses the risk set
    at that time, and the whole tied block is multiplied by the same S(t-). This matters
    here: event dates are daily and `time` is days/365.25, so tied times are dense (a few
    thousand distinct times carrying tens of thousands of observations). The earlier
    plotting code walked observations one at a time, decrementing the risk set and updating
    S *inside* a tied block, which made the estimate depend on the row order among ties.

    Censored observations at exactly an event time stay in that time's risk set, which is
    the standard convention and the one lifelines' Kaplan-Meier uses -- the identity
    CIF_event + CIF_death == 1 - S_KM(all-cause) holds exactly against it.
    """
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return np.zeros(len(grid))
    e1 = np.asarray(event) == 1
    e2 = (~e1) & (np.asarray(is_death) == 1)      # death only counts when not the event
    ut, inv = np.unique(t, return_inverse=True)
    at = t.size - np.concatenate([[0], np.cumsum(np.bincount(inv, minlength=ut.size))[:-1]])
    d1 = np.bincount(inv, weights=e1.astype(float), minlength=ut.size)
    d2 = np.bincount(inv, weights=e2.astype(float), minlength=ut.size)
    surv_prev = np.concatenate([[1.0], np.cumprod(1.0 - (d1 + d2) / at)[:-1]])
    cif = np.cumsum(surv_prev * d1 / at)
    pos = np.searchsorted(ut, np.asarray(grid, dtype=float), side="right") - 1
    return np.where(pos >= 0, cif[np.clip(pos, 0, None)], 0.0)


def risk_counts(t, event, times) -> tuple[np.ndarray, np.ndarray]:
    """(#still at risk, #cumulative events) at each of `times`; step 11's definition."""
    t = np.asarray(t, dtype=float)
    g = np.asarray(times, dtype=float)
    at = (t[:, None] >= g[None, :]).sum(0)
    cum = (t[np.asarray(event) == 1][:, None] <= g[None, :]).sum(0)
    return at, cum


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / s.std()


def _slug(s: str) -> str:
    return s.replace("-", "_").replace(" ", "_")
