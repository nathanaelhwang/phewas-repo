# %% [markdown]
# QWAS Step 01 - build the per-patient questionnaire target matrix.
#
# The cleaned questionnaire is ENCOUNTER-grain (228,790 encounters / 158,324 MRNs).
# QWAS needs one row per cohort patient, anchored on the sleep-study index date, so
# this step:
#   1. joins the questionnaire to the v2 cohort on MRN (the cohort supplies STUDY_DATE,
#      which sidesteps the questionnaire's own 51%-missing STUDY_DATE entirely -- that
#      field is KPHC-only),
#   2. keeps encounters with CONTACT_DATE <= STUDY_DATE (pre-treatment by construction),
#   3. takes the CLOSEST such encounter per patient (max CONTACT_DATE; ties broken by
#      source precedence kphc > somnoware > scanned, then by items answered),
#   4. recodes FOSQ 0 ("difficulty for other reasons") to <NA> per standard FOSQ scoring,
#   5. one-hots BEDTIME / WAKETIME into 12 two-hour clock bins each (NaN-preserving),
#   6. derives the four instrument scores (ESS/ISI/PHQ-2 totals, FOSQ mean),
#   7. applies the >=500-answered floor and writes the feature spec that step 02 runs.
#
# -> targets/targets.parquet (id + native-scale feature columns)
#    targets/feature_spec.csv (feature, domain, family, forms, binary cut, coverage)
#    targets/BUILD.md (de-identified build report)
# Data-safety: the parquet is keyed on the 12-char MRN and stays on local disk; stdout
# and BUILD.md carry de-identified aggregates only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import qwas_utils as U

# questionnaire_utils owns the item->family map (single source of truth for encodings)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "questionnaire"))
import questionnaire_utils as Q

SOURCE_RANK = {"kphc": 0, "somnoware": 1, "scanned": 2}
TIME_BINS = ["00:00-02:00", "02:00-04:00", "04:00-06:00", "06:00-08:00",
             "08:00-10:00", "10:00-12:00", "12:00-14:00", "14:00-16:00",
             "16:00-18:00", "18:00-20:00", "20:00-22:00", "22:00-00:00"]


def bin_col(prefix: str, label: str) -> str:
    """'BEDTIME' + '22:00-00:00' -> 'BEDTIME_2200_0000'."""
    lo, hi = label.split("-")
    return f"{prefix}_{lo.replace(':', '')}_{hi.replace(':', '')}"


def select_index_encounter(q: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One questionnaire per cohort patient: the closest one on/before the index date."""
    item_cols = [c for c in q.columns if c not in ("MRN", "STUDY_DATE", "CONTACT_DATE", "source")]
    m = q.merge(cohort[["id", "STUDY_DATE"]].rename(columns={"id": "MRN", "STUDY_DATE": "IDX"}),
                on="MRN", how="inner")
    stats = {"encounters_in_cohort": len(m), "patients_in_cohort": m["MRN"].nunique()}

    m["CONTACT_DATE"] = pd.to_datetime(m["CONTACT_DATE"], errors="coerce")
    stats["undated"] = int(m["CONTACT_DATE"].isna().sum())
    stats["post_index"] = int((m["CONTACT_DATE"] > m["IDX"]).sum())

    pre = m[m["CONTACT_DATE"].notna() & (m["CONTACT_DATE"] <= m["IDX"])].copy()
    stats["pre_index_encounters"] = len(pre)
    stats["pre_index_patients"] = pre["MRN"].nunique()

    # deterministic pick: latest date, then kphc > somnoware > scanned, then most answered
    pre["_rank"] = pre["source"].map(SOURCE_RANK).fillna(9).astype(int)
    pre["_answered"] = pre[item_cols].notna().sum(axis=1)
    pre = pre.sort_values(["MRN", "CONTACT_DATE", "_rank", "_answered"],
                          ascending=[True, False, True, False])
    one = pre.drop_duplicates("MRN", keep="first").copy()

    gap = (one["IDX"] - one["CONTACT_DATE"]).dt.days
    stats["selected_patients"] = len(one)
    stats["gap_same_day_pct"] = round(float((gap == 0).mean()) * 100, 1)
    stats["gap_median_days"] = float(gap.median())
    stats["gap_p90_days"] = float(gap.quantile(0.90))
    stats["source_mix"] = one["source"].value_counts().to_dict()
    return one, stats


def build_features(one: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Native-scale feature matrix + the spec row for each feature."""
    out = pd.DataFrame({"id": one["MRN"].to_numpy()})
    spec: list[dict] = []

    def add(name, values, family, forms, cut):
        out[name] = pd.to_numeric(pd.Series(np.asarray(values, dtype=object)),
                                  errors="coerce").astype("Float64").to_numpy()
        spec.append({"feature": name, "domain": U.domain_of(name), "family": family,
                     "forms": forms, "cut_op": cut[0] if cut else "",
                     "cut_val": cut[1] if cut else np.nan})

    # ---- raw items ----
    for code in Q.TARGET_CODES + Q.CC_COLUMNS:
        if code in U.EXCLUDED or code not in one.columns:
            continue
        family = "cc_binary" if code.startswith("CC_") else Q.FAMILY[code]
        v = pd.to_numeric(one[code], errors="coerce")
        if family == "fosq":
            v = v.mask(v == U.FOSQ_NA_VALUE)          # "other reasons" is not-applicable
        if family in U.BINARY_ONLY:
            add(code, v, family, "binary", None)
        elif family in U.CONT_ONLY:
            add(code, v, family, "continuous", None)
        elif family in U.BINARY_CUT:
            add(code, v, family, "both", U.BINARY_CUT[family])
        # any other family (text/empty/timebin) is excluded above

    # ---- BEDTIME / WAKETIME -> 12 two-hour bin binaries each (NaN-preserving) ----
    for prefix in ("BEDTIME", "WAKETIME"):
        lab = one[prefix].astype("string")
        known = lab.notna()
        for b in TIME_BINS:
            add(bin_col(prefix, b), np.where(known, (lab == b).astype(float), np.nan),
                "timebin_onehot", "binary", None)

    # ---- derived instrument scores ----
    for name, items, require_all, _rng, cut in U.COMPOSITES:
        vals = one[items].apply(pd.to_numeric, errors="coerce")
        if name == "FOSQ_MEAN":
            vals = vals.mask(vals == U.FOSQ_NA_VALUE)
            score = vals.mean(axis=1).where(vals.notna().sum(axis=1) >= U.FOSQ_MIN_ITEMS)
        else:
            score = vals.sum(axis=1).where(vals.notna().all(axis=1)) if require_all else vals.sum(axis=1)
        forms = "both" if cut else "continuous"
        add(name, score, "composite", forms, cut)

    return out, spec


def main() -> None:
    U.ensure_dirs()
    cohort = pd.read_parquet(U.COHORT, columns=["id", "STUDY_DATE", "severity"])
    q = pd.read_parquet(U.QUESTIONNAIRE)
    print(f"questionnaire {len(q):,} encounters / {q['MRN'].nunique():,} MRNs; cohort {len(cohort):,}")

    one, stats = select_index_encounter(q, cohort)
    print(f"selected {stats['selected_patients']:,} patients "
          f"({stats['gap_same_day_pct']}% same-day as the sleep study)")

    tg, spec = build_features(one)
    sp = pd.DataFrame(spec)

    # ---- coverage + floors ----
    sp["n_answered"] = [int(tg[f].notna().sum()) for f in sp["feature"]]
    sp["n_positive"] = [int((tg[f] >= 1).sum()) if fm == "binary" else pd.NA
                        for f, fm in zip(sp["feature"], sp["forms"])]
    sp["referral_item"] = sp["feature"].isin(U.REFERRAL_ITEMS)

    sp["drop_reason"] = ""
    sp.loc[sp["n_answered"] < U.MIN_PATIENTS, "drop_reason"] = f"n_answered<{U.MIN_PATIENTS}"
    # a natively-binary feature also needs both classes represented
    minority = sp.apply(lambda r: min(r["n_positive"], r["n_answered"] - r["n_positive"])
                        if r["forms"] == "binary" and pd.notna(r["n_positive"]) else np.nan, axis=1)
    sp.loc[(sp["forms"] == "binary") & (minority < U.MIN_EVENTS), "drop_reason"] = \
        f"minority_class<{U.MIN_EVENTS}"

    dropped = sp[sp["drop_reason"] != ""]
    sp = sp[sp["drop_reason"] == ""].drop(columns="drop_reason").reset_index(drop=True)
    tg = tg[["id"] + sp["feature"].tolist()]

    tg.to_parquet(U.TARGETS_DIR / "targets.parquet", index=False)
    sp.to_csv(U.TARGETS_DIR / "feature_spec.csv", index=False)

    # ---- de-identified build report ----
    sev = cohort.merge(tg[["id"]], on="id", how="inner")["severity"].value_counts()
    lines = [
        "# QWAS targets — build report\n",
        f"- questionnaire encounters: **{len(q):,}** ({q['MRN'].nunique():,} MRNs)",
        f"- encounters for cohort patients: {stats['encounters_in_cohort']:,} "
        f"({stats['patients_in_cohort']:,} patients)",
        f"- dropped: {stats['undated']:,} undated, {stats['post_index']:,} post-index",
        f"- on/before index: {stats['pre_index_encounters']:,} encounters "
        f"({stats['pre_index_patients']:,} patients)",
        f"- **analytic sample: {stats['selected_patients']:,} patients** "
        f"(closest encounter on/before index)",
        f"- index gap: {stats['gap_same_day_pct']}% same day, median "
        f"{stats['gap_median_days']:.0f} d, p90 {stats['gap_p90_days']:.0f} d before the study",
        f"- source of the selected encounter: {stats['source_mix']}",
        f"- severity mix: {sev.reindex(U.SEVERITIES).to_dict()}\n",
        f"## Phenome: {len(sp)} features",
        f"- by form: {sp['forms'].value_counts().to_dict()}",
        f"- by domain: {sp['domain'].value_counts().reindex(U.DOMAIN_ORDER).dropna().astype(int).to_dict()}",
        f"- flagged as referral-driving (ascertainment caveat): "
        f"{sorted(sp.loc[sp['referral_item'], 'feature'])}",
        f"- dropped below floor ({len(dropped)}): "
        f"{dict(zip(dropped['feature'], dropped['drop_reason'])) if len(dropped) else 'none'}\n",
        "## Encoding decisions",
        f"- FOSQ value 0 (\"difficulty for other reasons\") recoded to <NA> "
        f"(standard FOSQ scoring treats it as not-applicable), so FOSQ items run 1-4, higher = better.",
        f"- FOSQ_MEAN requires >={U.FOSQ_MIN_ITEMS}/10 answered items; ESS/ISI/PHQ-2 totals require all items.",
        "- BEDTIME/WAKETIME enter as 12 two-hour clock-bin binaries each (the continuous "
        "minute versions wrap at midnight).",
        "- Binary form of an ordinal item = the uniform \"moderate or worse\" scale midpoint "
        "(>=2; FOSQ <=2). Composites use their published cuts (ESS>=11, ISI>=15, PHQ-2>=3).",
    ]
    (U.TARGETS_DIR / "BUILD.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"phenome {len(sp)} features ({sp['forms'].value_counts().to_dict()}); "
          f"{len(dropped)} below floor")
    print("wrote", U.TARGETS_DIR / "targets.parquet", "+ feature_spec.csv + BUILD.md")
    U.log_milestone(f"Step 01 targets: {stats['selected_patients']:,}-patient index-anchored "
                    f"questionnaire matrix, {len(sp)}-feature phenome.")


if __name__ == "__main__":
    main()
