"""Validate the octant website export and write its manifest (G + VALIDATION).

Runs the release checks over the files scripts 27 and 28 wrote, then emits
`phenotype_website_export_manifest.json` with the estimator decision, the definitions a
reader needs to interpret the numbers, and a SHA-256 over every file in the directory.

The export FAILS if any check fails: the manifest is still written so the failure is on the
record, `validation.status` reads FAIL, and the process exits non-zero.

Check 14 is the disclosure sweep. It is deliberately blunt -- it re-reads every exported
file as text and looks for identifier column names, ISO dates in any cell, absolute paths,
and the names of the patient-level files that live next door -- because the cost of a false
alarm is a minute of triage and the cost of a miss is a PHI release.

Usage:  python 29_octant_export_validate.py
"""
from __future__ import annotations

import datetime as dt
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incwas_utils as U  # noqa: E402
import _pheno_engine as P  # noqa: E402

OUT = P.OUT / "website_exports"
EXPORTER = "analyses/incwas/27_octant_website_export.py + 28_octant_cluster_export.py"
VERSION = "1.2.0"
N_PANELS, HORIZON = 168, 3.0
N_OUTCOMES, N_OVERALL_RISK_ROWS = 21, 84
SAMPLED_EQUIVALENCE_TOLERANCE_PP = 0.00000051
N_COHORT, DISCLOSURE_MIN = 70_880, 11
COUNT_COLS = ["n_focal", "events_focal", "n_rest", "events_rest", "n_at_risk",
              "n_events_cum", "n_at_risk_baseline", "total_events",
              "n_at_risk_3yr", "n_nonmissing", "numerator", "denominator", "n"]
ID_COLS = {"mrn", "id", "ids", "person_id", "patient_id", "patientid", "subject_id",
           "ptid", "medrec", "record_number"}
FORBIDDEN_NAMES = ["octant_assignments", "cross_domain_phenotypes", "_identified",
                   "patient_records", "phecode_events", "followup_frame", "cohort.parquet"]
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
PATH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/Users/|\\Users\\")
OVERALL_AUDIT_NAME = "octant_cif_overall_audit.json"
OVERALL_CURVE_FILE = "octant_cif_overall_curves.csv"
OVERALL_RISK_FILE = "octant_cif_overall_risk_table.csv"
OVERALL_META_FILE = "octant_cif_overall_metadata.csv"
RISK_TIMES = np.array([0.0, 1.0, 2.0, 3.0])
MANIFEST_EXTENSION_FILES = {
    "race_ethnicity": (
        "octant_race_ethnicity_distributions.csv", "octant_race_ethnicity_audit.json"),
    "cpap_treatment_summary": ("octant_cpap_summary.csv", "octant_cpap_summary_audit.json"),
}
OVERALL_SCHEMAS = {
    OVERALL_CURVE_FILE: [
        "level", "outcome", "outcome_name", "category", "group", "time_years",
        "cif_pct", "curve_status", "suppressed",
    ],
    OVERALL_RISK_FILE: [
        "level", "outcome", "outcome_name", "category", "group", "time_years",
        "n_at_risk", "n_events_cum", "suppressed",
    ],
    OVERALL_META_FILE: [
        "level", "outcome", "outcome_name", "category", "n_at_risk_baseline",
        "total_events", "cif3_pct", "mean_followup_years", "n_at_risk_3yr",
        "pct_baseline_at_risk_3yr", "curve_status", "suppressed",
        "event_definition", "estimator", "competing_event", "time_origin",
    ],
}
KEYS = {
    "octant_cif_curves.csv": ["level", "outcome", "octant", "group", "time_years"],
    "octant_cif_metadata.csv": ["level", "outcome", "octant"],
    "octant_cif_risk_table.csv": ["level", "outcome", "octant", "group", "time_years"],
    "octant_cluster_distributions.csv": ["octant", "metric_id"],
    "octant_cluster_summary.csv": ["octant"],
    "octant_score_correlations.csv": ["score_x", "score_y"],
    OVERALL_CURVE_FILE: ["level", "outcome", "group", "time_years"],
    OVERALL_RISK_FILE: ["level", "outcome", "group", "time_years"],
    OVERALL_META_FILE: ["level", "outcome"],
}
REQUIRED_17 = ["odi4pct", "min_spo2", "mean_spo2", "t90_fraction", "ess_total",
               "isi_total", "fosq_impair", "phq2_total", "stop_total",
               "n_phecodes_distinct", "cond_obesity", "cond_hyperlipidemia",
               "cond_hypertension", "cond_impaired_fasting_glucose", "cond_anxiety",
               "cond_gerd", "index_ahi"]


def load(name, **kw):
    return pd.read_csv(OUT / name, **kw)


def _booleans(s: pd.Series) -> pd.Series:
    """Parse exported True/False values without treating arbitrary strings as truthy."""
    return s.astype("string").str.strip().str.lower().map({"true": True, "false": False})


def _event_times_valid(values: pd.Series) -> bool:
    """Accept variable event timelines with exact annual reporting checkpoints."""
    time = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return bool(len(time) >= len(RISK_TIMES) and np.isfinite(time).all()
                and time[0] == 0.0 and time[-1] == HORIZON
                and (np.diff(time) > 0).all()
                and np.isin(RISK_TIMES, time).all())


def _sampling_evidence_valid(evidence: dict, prefix: str) -> bool:
    diff = evidence[f"{prefix}_max_abs_diff_vs_previous_sampled_pp"]
    return bool(evidence[f"{prefix}_sampled_equivalent"] is True
                and isinstance(diff, (float, int)) and np.isfinite(diff)
                and 0 <= diff <= SAMPLED_EQUIVALENCE_TOLERANCE_PP
                and evidence[f"{prefix}_risk_table_unchanged"] is True
                and evidence[f"{prefix}_metadata_unchanged"] is True)


def _overall_evidence(aud: dict) -> dict:
    def boolean(key):
        return {"True": True, "False": False}.get(aud.get(key))

    def number(key, cast=float):
        try:
            return cast(aud[key])
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "direct_estimation_confirmed": boolean("overall_direct_estimation"),
        "raw_partition_reconciled": boolean("overall_raw_partition_reconciled"),
        "curves_audited": number("overall_curves_audited", int),
        "permutation_max_abs_diff_pp": number(
            "overall_max_abs_diff_under_row_permutation_pp"),
        "deterministic": boolean("overall_deterministic"),
        "risk_counts_permutation_invariant": boolean(
            "overall_risk_counts_permutation_invariant"),
        "focal_sha256_before": aud.get("focal_curves_sha256_before"),
        "focal_sha256_after": aud.get("focal_curves_sha256_after"),
        "curve_granularity": aud.get("curve_granularity"),
        "horizon_years": number("horizon_years"),
        **{f"{prefix}_{key}": boolean(f"{prefix}_{key}")
           for prefix in ("focal", "overall")
           for key in ("sampled_equivalent", "risk_table_unchanged", "metadata_unchanged")},
        **{f"{prefix}_max_abs_diff_vs_previous_sampled_pp": number(
            f"{prefix}_max_abs_diff_vs_previous_sampled_pp")
           for prefix in ("focal", "overall")},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preserve_manifest_extensions(manifest: dict, previous: dict) -> None:
    """Keep later exporters' decisions only for their byte-identical artifacts."""
    old_files = {entry["name"]: entry for entry in previous.get("files", [])}
    new_files = {entry["name"]: entry for entry in manifest["files"]}
    retained = {}
    for section, names in MANIFEST_EXTENSION_FILES.items():
        if section not in previous:
            if any(name in new_files for name in names):
                raise RuntimeError(f"cannot preserve {section}: prior extension is missing")
            continue
        block = previous[section]
        if set(block.get("files", {}).values()) != set(names):
            raise RuntimeError(f"cannot preserve {section}: artifact registry is invalid")
        for name in names:
            old, new = old_files.get(name), new_files.get(name)
            if (old is None or new is None
                    or not re.fullmatch(r"[0-9a-f]{64}", str(old.get("sha256", "")))
                    or old["sha256"] != new["sha256"]
                    or old.get("bytes") != new["bytes"]):
                raise RuntimeError(f"cannot preserve {section}: artifact changed or missing: {name}")
        retained[section] = names

    for section, names in retained.items():
        manifest[section] = copy.deepcopy(previous[section])
        for name in names:
            # Fresh byte/hash/CSV-row observations take precedence; retain the later
            # exporter's schema, approval, JSON-record counts, and audit status.
            merged = copy.deepcopy(old_files[name])
            merged.update(new_files[name])
            new_files[name].clear()
            new_files[name].update(merged)
        for field in ("suppressed_cells_by_file", "suppressed_rows_by_file"):
            previous_counts = previous.get("disclosure", {}).get(field, {})
            current_counts = manifest.setdefault("disclosure", {}).setdefault(field, {})
            for name in names:
                if name in previous_counts:
                    current_counts[name] = previous_counts[name]

    if "race_ethnicity" in retained:
        manifest["disclosure"]["note"] = previous["disclosure"]["note"]
        manifest["unavailable_or_excluded"] = [
            item for item in manifest.get("unavailable_or_excluded", [])
            if str(item.get("field", "")).strip().casefold() != "race/ethnicity breakdown"]
    if "cpap_treatment_summary" in retained:
        manifest["unavailable_or_excluded"] = [
            item for item in manifest.get("unavailable_or_excluded", [])
            if "cpap" not in str(item.get("field", "")).casefold()]
        manifest["unavailable_or_excluded"].extend(copy.deepcopy([
            item for item in previous.get("unavailable_or_excluded", [])
            if "cpap" in str(item.get("field", "")).casefold()]))


def _suppression_stats(cur: pd.DataFrame, risk: pd.DataFrame,
                       meta: pd.DataFrame) -> dict:
    def flags(df):
        parsed = _booleans(df["suppressed"])
        return int(parsed.fillna(False).sum())

    return {
        OVERALL_CURVE_FILE: {
            "suppressed_rows": flags(cur),
            "suppressed_cells": int(cur.cif_pct.isna().sum()),
        },
        OVERALL_RISK_FILE: {
            "suppressed_rows": flags(risk),
            "suppressed_cells": int(
                risk[["n_at_risk", "n_events_cum"]].isna().sum().sum()),
        },
        OVERALL_META_FILE: {
            "suppressed_rows": flags(meta),
            "suppressed_cells": int(meta[
                ["n_at_risk_baseline", "total_events", "n_at_risk_3yr", "cif3_pct",
                 "pct_baseline_at_risk_3yr"]
            ].isna().sum().sum()),
        },
    }


def _overall_audit(cur: pd.DataFrame, risk: pd.DataFrame, meta: pd.DataFrame,
                   aud: dict, evidence: dict, status="PENDING",
                   results: list[dict] | None = None) -> dict:
    relevant = [] if results is None else [
        {"check": c["check"], "description": c["description"], "status": c["status"]}
        for c in results if int(c["check"]) >= 18
    ]
    failed = sum(c["status"] == "FAIL" for c in relevant)
    return {
        "files": [
            {"name": OVERALL_CURVE_FILE, "rows": len(cur)},
            {"name": OVERALL_RISK_FILE, "rows": len(risk)},
            {"name": OVERALL_META_FILE, "rows": len(meta)},
        ],
        "validation": {
            "status": status, "n_checks": len(relevant), "n_failed": failed,
            "checks": relevant,
        },
        "suppression": {
            "threshold": DISCLOSURE_MIN,
            "by_file": _suppression_stats(cur, risk, meta),
        },
        "estimator": {
            "name": aud.get("estimator", "Aalen-Johansen, tie-aware"),
            "tie_handling": aud.get("tie_handling", "grouped by unique time"),
            "competing_event": "death",
            "curves_are_adjusted": False,
        },
        "direct_estimation": {
            "group": "all",
            "cohort": "complete outcome-specific at-risk cohort",
            "confirmed": evidence["direct_estimation_confirmed"],
            "raw_partition_reconciled": evidence["raw_partition_reconciled"],
        },
        "determinism": {
            "curves_audited": evidence["curves_audited"],
            "max_abs_diff_under_row_permutation_pp":
                evidence["permutation_max_abs_diff_pp"],
            "deterministic": evidence["deterministic"],
            "risk_counts_permutation_invariant":
                evidence["risk_counts_permutation_invariant"],
        },
        "existing_focal_curves": {
            "file": "octant_cif_curves.csv",
            "sha256_before": evidence["focal_sha256_before"],
            "sha256_after": evidence["focal_sha256_after"],
            "sampled_equivalent": evidence["focal_sampled_equivalent"],
            "max_abs_diff_vs_previous_sampled_pp":
                evidence["focal_max_abs_diff_vs_previous_sampled_pp"],
            "risk_table_unchanged": evidence["focal_risk_table_unchanged"],
            "metadata_unchanged": evidence["focal_metadata_unchanged"],
        },
        "event_time_export": {
            "granularity": evidence["curve_granularity"],
            "horizon_years": evidence["horizon_years"],
            "sampled_equivalent": evidence["overall_sampled_equivalent"],
            "max_abs_diff_vs_previous_sampled_pp":
                evidence["overall_max_abs_diff_vs_previous_sampled_pp"],
            "risk_table_unchanged": evidence["overall_risk_table_unchanged"],
            "metadata_unchanged": evidence["overall_metadata_unchanged"],
            "sampling_equivalence_tolerance_pp": SAMPLED_EQUIVALENCE_TOLERANCE_PP,
        },
        "confidence_intervals": {
            "available": False,
            "reason": "The canonical tie-aware estimator has no validated variance implementation.",
        },
    }


def checks(cur, meta, risk, dist, summ, corr, overall_cur, overall_meta,
           overall_risk, overall_evidence) -> list[dict]:
    out, panel = [], ["level", "outcome", "octant"]

    def add(n, desc, ok, detail=""):
        out.append({"check": n, "description": desc,
                    "status": "PASS" if ok else "FAIL", "detail": str(detail)})

    add(1, "exactly 168 metadata rows", len(meta) == N_PANELS, len(meta))
    add(2, "variable curve rows include annual checkpoints for all 336 curves",
        len(cur) >= N_PANELS * 2 * len(RISK_TIMES), len(cur))

    g = cur.groupby(panel)["group"].agg(lambda s: sorted(set(s)))
    add(3, "each panel has exactly the two groups focal/other_seven",
        len(g) == N_PANELS and all(v == ["focal", "other_seven"] for v in g),
        f"{len(g)} panels")

    per = cur.groupby(panel + ["group"])
    ordered = per["time_years"].apply(_event_times_valid).all()
    sizes = per["time_years"].size()
    add(4, "each event timeline is finite, strictly increasing, spans 0 to 3 years, "
           "and includes annual checkpoints", bool(ordered),
        f"{len(sizes)} curves; points per curve min={sizes.min()}, max={sizes.max()}")
    add(5, "every curve begins at time zero",
        bool((per["time_years"].min() == 0.0).all()))
    add(6, "cif_pct within [0, 100]",
        bool(cur.cif_pct.between(0, 100).all()),
        f"range [{cur.cif_pct.min():.4f}, {cur.cif_pct.max():.4f}]")
    worst = per["cif_pct"].apply(lambda s: float(np.diff(s.to_numpy()).min())
                                 if len(s) > 1 else 0.0).min()
    add(7, "cif_pct nondecreasing within tolerance 1e-9", worst >= -1e-9,
        f"most negative step {worst:.2e} pp")

    end = (cur[cur.time_years == HORIZON]
           .pivot_table(index=panel, columns="group", values="cif_pct"))
    j = meta.set_index(panel).join(end.reindex(columns=["focal", "other_seven"]))
    d8 = pd.concat([(j.cif3_focal_pct - j.focal).abs(),
                    (j.cif3_rest_pct - j.other_seven).abs()]).max()
    endpoint_ok = bool(len(end) == N_PANELS
                       and j[["cif3_focal_pct", "cif3_rest_pct", "focal", "other_seven"]]
                       .notna().all().all() and d8 <= 0.02)
    add(8, "exact 3-year curve endpoint matches metadata within 0.02 pp", endpoint_ok,
        f"max |diff| {d8:.6f} pp")

    ci = meta.dropna(subset=["hr_m4", "ci_low", "ci_high"])
    add(9, "ci_low <= hr_m4 <= ci_high",
        bool((ci.ci_low <= ci.hr_m4).all() and (ci.hr_m4 <= ci.ci_high).all()),
        f"{len(ci)} of {len(meta)} rows with a fitted HR")
    ps = pd.concat([meta.p, meta.omnibus_p_m4]).dropna()
    add(10, "p-values within [0, 1]", bool(ps.between(0, 1).all()),
        f"{len(ps)} p-values, max {ps.max():.4f}")

    add(11, f"cluster counts reconcile to {N_COHORT:,}",
        int(summ.n.sum()) == N_COHORT and len(summ) == 8,
        f"{int(summ.n.sum()):,} over {len(summ)} octants, "
        f"pct sum {summ.pct.sum():.4f}")

    have = dist[dist.octant != "overall"].groupby("octant")["metric_id"].apply(set)
    add(12, "every octant carries all 17 required metrics",
        len(have) == 8 and all(set(REQUIRED_17) <= s for s in have),
        f"{len(have)} octants, {len(set(dist.metric_id))} metrics each")

    dups = {f: int(load(f, dtype=str).duplicated(k).sum()) for f, k in KEYS.items()}
    add(13, "no duplicate logical keys in any file", all(v == 0 for v in dups.values()),
        dups)

    hits = []
    for f in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json")):
        txt = f.read_text(encoding="utf-8")
        cols = [c.strip().strip('"').lower()
                for c in txt.split("\n", 1)[0].split(",")] if f.suffix == ".csv" else []
        hits += [f"{f.name}: identifier column '{c}'" for c in cols if c in ID_COLS]
        if f.suffix == ".csv" and DATE_RE.search(txt):
            hits.append(f"{f.name}: ISO date in a cell")
        if PATH_RE.search(txt):
            hits.append(f"{f.name}: absolute path")
        hits += [f"{f.name}: references {n}" for n in FORBIDDEN_NAMES if n in txt]
    time_ok = all(pd.to_numeric(frame.time_years, errors="coerce")
                  .between(0.0, HORIZON).all() for frame in (cur, overall_cur))
    risk_ok = all(np.array_equal(pd.to_numeric(grp.time_years, errors="coerce"),
                                 RISK_TIMES)
                  for _, grp in risk.groupby(panel + ["group"], dropna=False))
    risk_ok &= set(pd.to_numeric(overall_risk.time_years, errors="coerce")) == set(RISK_TIMES)
    if not time_ok or not risk_ok:
        hits.append("relative curve times or annual risk-table times violate the export contract")
    add(14, "no identifiers, calendar dates, absolute paths or "
            "forbidden filenames", not hits, hits or "clean")

    bad = []
    for f in KEYS:
        t = load(f)
        for c in [c for c in COUNT_COLS if c in t.columns]:
            v = pd.to_numeric(t[c], errors="coerce").dropna()
            if (v < DISCLOSURE_MIN).any():
                bad.append(f"{f}:{c} min {int(v.min())}")
    add(15, f"no unsuppressed count below {DISCLOSURE_MIN}", not bad, bad or "clean")

    # the section E / F structural expectations, checked rather than assumed
    m = corr.pivot(index="score_x", columns="score_y", values="spearman_rho")
    add(16, "score correlations are a complete symmetric 3x3 with unit diagonal",
        bool(len(corr) == 9 and m.shape == (3, 3)
             and np.allclose(m.to_numpy(), m.to_numpy().T)
             and np.allclose(np.diag(m.to_numpy()), 1.0)),
        f"off-diagonal range [{m.to_numpy()[~np.eye(3, dtype=bool)].min():+.4f}, "
        f"{m.to_numpy()[~np.eye(3, dtype=bool)].max():+.4f}]")

    gl = summ.glyph.map(lambda g: tuple(c == "■" for c in g))
    bits = list(zip(summ.physiology_high, summ.symptom_high, summ.comorbidity_high))
    add(17, "glyph reads physiology/symptom/comorbidity and coverage is 100%",
        bool(list(gl) == bits and len(set(gl)) == 8
             and (summ.classification_coverage_pct == 100.0).all()),
        f"{len(set(gl))} distinct glyphs")

    # Overall ("all") curves: a separate 21-outcome release contract. Reindexing lets a
    # malformed schema be recorded as FAIL rather than causing the later checks to crash.
    frames = {
        OVERALL_CURVE_FILE: overall_cur,
        OVERALL_RISK_FILE: overall_risk,
        OVERALL_META_FILE: overall_meta,
    }
    schema_ok = all(list(frames[name].columns) == columns
                    for name, columns in OVERALL_SCHEMAS.items())
    add(18, "overall exports have the exact approved schemas", schema_ok,
        {name: list(df.columns) for name, df in frames.items()})
    oc = overall_cur.reindex(columns=OVERALL_SCHEMAS[OVERALL_CURVE_FILE])
    ora = overall_risk.reindex(columns=OVERALL_SCHEMAS[OVERALL_RISK_FILE])
    om = overall_meta.reindex(columns=OVERALL_SCHEMAS[OVERALL_META_FILE])

    rows_ok = (len(oc) >= N_OUTCOMES * len(RISK_TIMES) and
               len(ora) == N_OVERALL_RISK_ROWS and len(om) == N_OUTCOMES)
    groups_ok = bool((oc.group == "all").all() and (ora.group == "all").all())
    add(19, "overall exports have variable curve rows, 84 risk rows, 21 metadata rows, "
            "and group is always all",
        rows_ok and groups_ok,
        {OVERALL_CURVE_FILE: len(oc), OVERALL_RISK_FILE: len(ora),
         OVERALL_META_FILE: len(om)})

    key_cols = ["level", "outcome"]
    legacy_keys = meta[key_cols].fillna("").astype(str).drop_duplicates()
    level_counts = legacy_keys.level.value_counts().to_dict()
    expected_keys = set(legacy_keys.itertuples(index=False, name=None))
    keys_ok = len(expected_keys) == N_OUTCOMES and level_counts == {
        "phecode": 15, "system": 6}
    outcome_counts = {}
    for name, df in frames.items():
        keys = df.reindex(columns=key_cols).fillna("").astype(str).drop_duplicates()
        outcome_counts[name] = len(keys)
        keys_ok &= (len(keys) == N_OUTCOMES
                    and set(keys.itertuples(index=False, name=None)) == expected_keys)
    add(20, "overall files match the exact 21 legacy outcome keys (15 phecode/6 system)",
        bool(keys_ok), {"legacy_by_level": level_counts, "files": outcome_counts})

    curve_groups = oc.groupby(key_cols, sort=False, dropna=False)
    event_timelines = []
    for _, grp in curve_groups:
        event_timelines.append(_event_times_valid(grp.time_years))
    add(21, "each overall event timeline is finite, strictly increasing, spans 0 to 3 "
            "years, and includes annual checkpoints",
        len(event_timelines) == N_OUTCOMES and all(event_timelines),
        f"{sum(event_timelines)}/{len(event_timelines)} valid timelines")

    allowed_status = {"available", "withheld"}
    oc_flag, om_flag = _booleans(oc.suppressed), _booleans(om.suppressed)
    curve_status_ok = bool(
        oc.curve_status.isin(allowed_status).all() and oc_flag.notna().all()
        and (((oc.curve_status == "available") & oc_flag.eq(False))
             | ((oc.curve_status == "withheld")
                & oc_flag.eq(True))).all()
        and om.curve_status.isin(allowed_status).all() and om_flag.notna().all()
        and om.loc[om.curve_status == "withheld", "suppressed"].pipe(
            _booleans).eq(True).all())
    curve_state = oc[key_cols + ["curve_status"]].copy()
    curve_state["suppressed"] = oc_flag
    curve_state = curve_state.drop_duplicates()
    meta_state = om[key_cols + ["curve_status"]].copy()
    meta_state["suppressed"] = om_flag
    meta_state = meta_state.drop_duplicates()
    state_match = curve_state.merge(meta_state, on=key_cols, how="outer",
                                    suffixes=("_curve", "_meta"), indicator=True)
    curve_status_ok &= bool(
        len(curve_state) == N_OUTCOMES and len(meta_state) == N_OUTCOMES
        and (state_match._merge == "both").all()
        and (state_match.curve_status_curve == state_match.curve_status_meta).all())
    add(22, "overall curve_status and suppression flags are valid and metadata-consistent",
        curve_status_ok, f"{len(curve_state)} curve states, {len(meta_state)} metadata states")

    cif_ok, worst_overall, available_curves, withheld_curves = True, 0.0, 0, 0
    for _, grp in curve_groups:
        status = set(grp.curve_status.dropna())
        if status == {"available"}:
            available_curves += 1
            cif = pd.to_numeric(grp.cif_pct, errors="coerce")
            steps = np.diff(cif.to_numpy()) if len(cif) > 1 else np.array([0.0])
            finite_steps = steps[np.isfinite(steps)]
            if finite_steps.size:
                worst_overall = min(worst_overall, float(finite_steps.min()))
            cif_ok &= bool(cif.notna().all() and cif.between(0, 100).all()
                           and np.isfinite(cif).all() and (steps >= -1e-9).all())
        elif status == {"withheld"}:
            withheld_curves += 1
            cif_ok &= bool(grp.cif_pct.isna().all())
        else:
            cif_ok = False
    add(23, "available overall CIFs are complete/bounded/monotone and withheld CIFs null",
        bool(cif_ok and available_curves + withheld_curves == N_OUTCOMES),
        f"available {available_curves}, withheld {withheld_curves}, "
        f"most negative step {worst_overall:.2e} pp")

    endpoints = oc[pd.to_numeric(oc.time_years, errors="coerce") == HORIZON][
        key_cols + ["cif_pct", "curve_status"]
    ].merge(om[key_cols + ["cif3_pct", "curve_status"]], on=key_cols, how="outer",
            suffixes=("_curve", "_meta"), indicator=True)
    endpoint_diff = (pd.to_numeric(endpoints.cif_pct, errors="coerce")
                     - pd.to_numeric(endpoints.cif3_pct, errors="coerce")).abs()
    endpoint_available = (endpoints.curve_status_curve == "available")
    endpoint_withheld = endpoints.curve_status_curve == "withheld"
    endpoint_ok = bool(
        len(endpoints) == N_OUTCOMES and (endpoints._merge == "both").all()
        and (endpoints.curve_status_curve == endpoints.curve_status_meta).all()
        and endpoint_diff[endpoint_available].notna().all()
        and (endpoint_diff[endpoint_available] <= 0.0001).all()
        and endpoints.loc[endpoint_withheld, ["cif_pct", "cif3_pct"]].isna().all().all())
    max_endpoint_diff = float(endpoint_diff[endpoint_available].max()) \
        if endpoint_available.any() else 0.0
    add(24, "overall 3-year curve endpoints reconcile to cif3_pct within 0.0001 pp",
        endpoint_ok, f"max |diff| {max_endpoint_diff:.6f} pp")

    risk_groups = ora.groupby(key_cols, sort=False, dropna=False)
    exact_risk_grids = []
    for _, grp in risk_groups:
        time = pd.to_numeric(grp.time_years, errors="coerce").to_numpy()
        exact_risk_grids.append(len(time) == len(RISK_TIMES)
                                and np.array_equal(time, RISK_TIMES))
    add(25, "every overall risk table uses the exact ordered 0/1/2/3-year grid",
        len(exact_risk_grids) == N_OUTCOMES and all(exact_risk_grids),
        f"{sum(exact_risk_grids)}/{len(exact_risk_grids)} exact grids")

    count_ok, direction_ok = True, True
    for frame, columns in ((ora, ["n_at_risk", "n_events_cum"]),
                           (om, ["n_at_risk_baseline", "total_events", "n_at_risk_3yr"])):
        for column in columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            known = numeric.dropna()
            count_ok &= bool((frame[column].isna() == numeric.isna()).all()
                             and (known >= DISCLOSURE_MIN).all()
                             and np.equal(known, np.floor(known)).all())
    for _, grp in risk_groups:
        at = pd.to_numeric(grp.n_at_risk, errors="coerce").dropna().to_numpy()
        events = pd.to_numeric(grp.n_events_cum, errors="coerce").dropna().to_numpy()
        direction_ok &= bool((np.diff(at) <= 0).all() and (np.diff(events) >= 0).all())
    add(26, f"overall exact counts are integral, disclosed only at {DISCLOSURE_MIN}+, "
            "and known risk/event counts are directional", count_ok and direction_ok)

    risk_flag = _booleans(ora.suppressed)
    risk_missing = ora[["n_at_risk", "n_events_cum"]].isna().any(axis=1)
    meta_missing = om[["n_at_risk_baseline", "total_events", "n_at_risk_3yr"]].isna().any(axis=1)
    meta_suppressed = meta_missing | om.curve_status.eq("withheld")
    pct_inputs_suppressed = om[["n_at_risk_baseline", "n_at_risk_3yr"]].isna().any(axis=1)
    suppression_ok = bool(
        risk_flag.notna().all() and (risk_flag == risk_missing).all()
        and om_flag.notna().all() and (om_flag == meta_suppressed).all()
        and om.loc[pct_inputs_suppressed, "pct_baseline_at_risk_3yr"].isna().all())
    add(27, "overall null exact counts and suppression flags agree; suppressed 3-year "
            "counts have null percentages", suppression_ok,
        _suppression_stats(oc, ora, om))

    estimator_text = om.estimator.fillna("").astype(str)
    methods_ok = bool(
        om.event_definition.fillna("").astype(str).str.len().gt(0).all()
        and estimator_text.str.contains("Aalen-Johansen", case=False, regex=False).all()
        and estimator_text.str.contains("tie-aware", case=False, regex=False).all()
        and om.competing_event.fillna("").astype(str).str.lower().eq("death").all()
        and om.time_origin.fillna("").astype(str).str.len().gt(0).all()
        and pd.to_numeric(om.mean_followup_years, errors="coerce").ge(0).all())
    audit_direct_ok = bool(
        overall_evidence["direct_estimation_confirmed"] is True
        and overall_evidence["raw_partition_reconciled"] is True
        and overall_evidence["curves_audited"] == N_OUTCOMES)
    add(28, "overall audit confirms direct complete-cohort estimation and raw partition "
            "reconciliation", methods_ok and audit_direct_ok,
        {"direct": overall_evidence["direct_estimation_confirmed"],
         "partition": overall_evidence["raw_partition_reconciled"],
         "curves_audited": overall_evidence["curves_audited"]})

    permutation_ok = bool(
        overall_evidence["deterministic"] is True
        and overall_evidence["permutation_max_abs_diff_pp"] == 0.0
        and overall_evidence["risk_counts_permutation_invariant"] is True)
    add(29, "overall curves and risk counts are deterministic under row-order permutation",
        permutation_ok,
        {"deterministic": overall_evidence["deterministic"],
         "max_abs_diff_pp": overall_evidence["permutation_max_abs_diff_pp"],
         "risk_counts": overall_evidence["risk_counts_permutation_invariant"]})

    before = overall_evidence["focal_sha256_before"]
    after = overall_evidence["focal_sha256_after"]
    current_focal_sha = _sha256(OUT / "octant_cif_curves.csv")
    focal_hash_ok = bool(
        isinstance(before, str)
        and isinstance(after, str) and re.fullmatch(r"[0-9a-f]{64}", before)
        and re.fullmatch(r"[0-9a-f]{64}", after)
        and after == current_focal_sha)
    preservation_ok = all(_sampling_evidence_valid(overall_evidence, prefix)
                          for prefix in ("focal", "overall"))
    event_time_ok = (overall_evidence["curve_granularity"] == "event_time"
                     and overall_evidence["horizon_years"] == HORIZON)
    add(30, "event-time curves reproduce previous sampled estimates, preserve annual "
            "risk and metadata tables, and match the audited focal artifact",
        focal_hash_ok and preservation_ok and event_time_ok,
        {"focal_sampled_equivalent": overall_evidence["focal_sampled_equivalent"],
         "overall_sampled_equivalent": overall_evidence["overall_sampled_equivalent"],
         "annual_and_metadata_unchanged": preservation_ok,
         "curve_granularity": overall_evidence["curve_granularity"],
         "after_equals_current": after == current_focal_sha})
    return out


def main() -> int:
    manifest_path = OUT / "phenotype_website_export_manifest.json"
    previous_manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                         if manifest_path.exists() else {})
    cur = load("octant_cif_curves.csv", dtype={"outcome": str})
    meta = load("octant_cif_metadata.csv", dtype={"outcome": str})
    risk = load("octant_cif_risk_table.csv", dtype={"outcome": str})
    overall_cur = load(OVERALL_CURVE_FILE, dtype={"outcome": str})
    overall_meta = load(OVERALL_META_FILE, dtype={"outcome": str})
    overall_risk = load(OVERALL_RISK_FILE, dtype={"outcome": str})
    dist = load("octant_cluster_distributions.csv")
    summ = load("octant_cluster_summary.csv")
    corr = load("octant_score_correlations.csv")
    aud = dict(zip(*load("_survival_export_audit.csv").values.T))
    caud = dict(zip(*load("_cluster_export_audit.csv").values.T))

    audit_path = OUT / OVERALL_AUDIT_NAME
    overall_evidence = _overall_evidence(aud)

    # The disclosure sweep runs inside checks(). Put a path-free provisional audit on disk
    # first so even a first release validates the audit that the final manifest will hash.
    audit_path.write_text(json.dumps(
        _overall_audit(overall_cur, overall_risk, overall_meta, aud, overall_evidence),
        indent=2, ensure_ascii=False), encoding="utf-8")

    res = checks(cur, meta, risk, dist, summ, corr, overall_cur, overall_meta,
                 overall_risk, overall_evidence)
    failed = [c for c in res if c["status"] == "FAIL"]
    overall_failed = [c for c in failed if int(c["check"]) >= 18]
    audit_path.write_text(json.dumps(
        _overall_audit(overall_cur, overall_risk, overall_meta, aud, overall_evidence,
                       status="FAIL" if overall_failed else "PASS", results=res),
        indent=2, ensure_ascii=False), encoding="utf-8")
    overall_suppression = _suppression_stats(overall_cur, overall_risk, overall_meta)

    man = {
        "release": {
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "exporter": EXPORTER, "exporter_version": VERSION,
            "analysis": "octant phenotype incidence WAS (phenotype-exposure incwas)",
        },
        "estimator": {
            "estimator": aud["estimator"], "tie_handling": aud["tie_handling"],
            "competing_event": "death (cause-specific; a patient who dies without the "
                               "outcome contributes to the competing cause)",
            "curves_are_adjusted": False,
            "curves_audited": int(aud["curves_audited"]),
            "tied_observations_pct": float(aud["tied_observations_pct"]),
            "max_abs_diff_vs_previous_implementation_pp": float(
                aud["max_abs_diff_vs_legacy_sequential_pp"]),
            "mean_abs_diff_vs_previous_implementation_pp": float(
                aud["mean_abs_diff_vs_legacy_sequential_pp"]),
            "identity_residual_vs_lifelines_km_pp": float(
                aud["max_abs_identity_residual_vs_lifelines_km_pp"]),
            "permutation_invariant": aud["deterministic"] == "True",
            "deterministic": aud["deterministic"] == "True",
            "decision": "Adopted the tie-aware estimator as canonical and regenerated the "
                        "publication figures and 3-year summaries (script 25) from it. "
                        "87% of observations sit on a tied time because event dates are "
                        "daily, and the previous implementation walked observations one at "
                        "a time, so it depended on the row order among ties. All 21 "
                        "published endpoints moved by <=0.04 pp, always downward.",
            "reference_check": aud["note"],
        },
        "design": {
            "follow_up_horizon_years": HORIZON,
            "time_grid": {"start": 0.0, "end": HORIZON, "spacing": "event_time",
                          "checkpoints_years": RISK_TIMES.tolist(),
                          "definition": "every unique incident-event time plus annual checkpoints"},
            "risk_table_times_years": [0.0, 1.0, 2.0, 3.0],
            "cohort_on_incwas_spine": 25_380,
            "shared_phenotype_cohort": N_COHORT,
            "tail_warning": "mean follow-up on this cohort is 1.42 y; only ~8.7% of the "
                            "baseline risk set is still under observation at 3 y, so the "
                            "right-hand end of every curve is thin -- read the risk table "
                            "before quoting a 3-year number.",
        },
        "curves": {
            "rows": len(cur), "panels": int(len(cur.groupby(
                ["level", "outcome", "octant"]))),
            "panels_by_level": {k: int(v) for k, v in cur.groupby("level").apply(
                lambda g: len(g.groupby(["outcome", "octant"]))).items()},
            "groups_per_panel": 2, "granularity": "event_time",
            "points_per_curve_min": int(cur.groupby(
                ["level", "outcome", "octant", "group"]).size().min()),
            "points_per_curve_max": int(cur.groupby(
                ["level", "outcome", "octant", "group"]).size().max()),
        },
        "metadata_table": {
            "rows": len(meta), "model": "m4",
            "currently_published_panels": int(meta.currently_published.sum()),
            "by_level": {k: int(v) for k, v in meta.level.value_counts().items()},
        },
        "overall_exports": {
            "definition": "group=all is estimated directly from each complete "
                          "outcome-specific at-risk cohort; it is not an octant average "
                          "or a reconstruction from focal curves",
            "group": "all", "outcomes": int(overall_meta[["level", "outcome"]]
                                              .drop_duplicates().shape[0]),
            "curves": {"file": OVERALL_CURVE_FILE, "rows": len(overall_cur),
                       "granularity": "event_time",
                       "points_per_curve_min": int(overall_cur.groupby(
                           ["level", "outcome", "group"]).size().min()),
                       "points_per_curve_max": int(overall_cur.groupby(
                           ["level", "outcome", "group"]).size().max())},
            "risk_table": {"file": OVERALL_RISK_FILE, "rows": len(overall_risk),
                           "times_years": RISK_TIMES.tolist()},
            "metadata": {"file": OVERALL_META_FILE, "rows": len(overall_meta)},
            "audit": OVERALL_AUDIT_NAME,
            "estimator": aud["estimator"], "competing_event": "death",
            "curves_are_adjusted": False,
            "direct_estimation_confirmed":
                overall_evidence["direct_estimation_confirmed"],
            "raw_partition_reconciled": overall_evidence["raw_partition_reconciled"],
            "permutation_invariant": overall_evidence["deterministic"],
            "risk_counts_permutation_invariant":
                overall_evidence["risk_counts_permutation_invariant"],
            "max_abs_diff_under_row_permutation_pp":
                overall_evidence["permutation_max_abs_diff_pp"],
            "existing_focal_curves_sampled_equivalent":
                overall_evidence["focal_sampled_equivalent"],
            "existing_focal_curves_max_abs_diff_pp":
                overall_evidence["focal_max_abs_diff_vs_previous_sampled_pp"],
            "previous_overall_curves_sampled_equivalent":
                overall_evidence["overall_sampled_equivalent"],
            "previous_overall_curves_max_abs_diff_pp":
                overall_evidence["overall_max_abs_diff_vs_previous_sampled_pp"],
            "annual_risk_tables_unchanged": all(
                overall_evidence[f"{prefix}_risk_table_unchanged"] is True
                for prefix in ("focal", "overall")),
            "metadata_tables_unchanged": all(
                overall_evidence[f"{prefix}_metadata_unchanged"] is True
                for prefix in ("focal", "overall")),
            "withholding": "if the complete cohort fails the disclosure threshold, "
                           "curve coordinates and cif3_pct are null and curve_status is "
                           "withheld; suppressed exact counts are never written as zero",
            "confidence_intervals": {
                "available": False,
                "reason": "The canonical tie-aware estimator has no validated variance "
                          "implementation; no intervals were approximated.",
            },
        },
        "cluster_tables": {
            "distributions_rows": len(dist),
            "distributions_metrics": int(dist.metric_id.nunique()),
            "distributions_groups": "8 octants + 'overall' cohort reference row",
            "summary_rows": len(summ), "correlations_rows": len(corr),
            "standardized_difference_definition": caud["standardized_difference_def"],
            "signature_reproduction_max_dev": float(caud["signature_max_dev"]),
        },
        "disclosure": {
            "threshold": DISCLOSURE_MIN,
            "rule": "exact counts below the threshold are blanked (null) and the row is "
                    "flagged suppressed=true; a suppressed value is never replaced by 0",
            "suppressed_cells_by_file": {
                "octant_cif_metadata.csv": int(aud["suppressed_cells_metadata"]),
                "octant_cif_risk_table.csv": int(aud["suppressed_cells_risk_table"]),
                "octant_cluster_distributions.csv": int(
                    caud["suppressed_cells_distributions"]),
                "octant_cif_curves.csv": 0,
                "octant_cluster_summary.csv": 0,
                "octant_score_correlations.csv": 0,
                OVERALL_CURVE_FILE: overall_suppression[OVERALL_CURVE_FILE][
                    "suppressed_cells"],
                OVERALL_RISK_FILE: overall_suppression[OVERALL_RISK_FILE][
                    "suppressed_cells"],
                OVERALL_META_FILE: overall_suppression[OVERALL_META_FILE][
                    "suppressed_cells"],
            },
            "suppressed_rows_by_file": {
                OVERALL_CURVE_FILE: overall_suppression[OVERALL_CURVE_FILE][
                    "suppressed_rows"],
                OVERALL_RISK_FILE: overall_suppression[OVERALL_RISK_FILE][
                    "suppressed_rows"],
                OVERALL_META_FILE: overall_suppression[OVERALL_META_FILE][
                    "suppressed_rows"],
            },
            "note": "cumulative-incidence percentages are retained even where the event "
                    "count behind them is suppressed: they are not exact counts, the "
                    "smallest group behind any curve holds over a thousand patients, and "
                    "the metadata carries a suppressed flag so a thin curve can be "
                    "greyed out. Race/ethnicity is excluded entirely.",
        },
        "definitions": {
            "phecode_outcome": (
                "incident phecode: at-risk = free of the phecode AND of every phecode in "
                "its pyPheWAS control-exclusion family at index (this harmonises the "
                "ICD-9/ICD-10 phecode split); event = second distinct post-index code "
                "(rule of 2), dated at that second code."),
            "system_outcome": (
                "first NEW phecode in the body system, with the whole cohort at risk "
                "(`anynew`). The estimand is a rate of DIAGNOSTIC ACCRUAL, not first-ever "
                "disease onset. The strict system-wide washout was rejected because it "
                "deletes the exposure: it retains 0.2% of high-comorbidity patients for "
                "respiratory and collapses mean score_c3 from 0.83 to 0.18."),
            "body_system": "the `category` field of pyPheWAS v1.2's phecode dimension (18 "
                           "categories over 1,866 phecodes). Two of the 18 -- `symptoms` "
                           "and `other` -- are not organ systems.",
            "octant": "each of the three latent gradient scores cut at its own cohort "
                      "median, giving eight cells that classify every patient. The cut "
                      "points are cohort-derived medians, NOT clinical thresholds, and do "
                      "not transfer to another sample.",
            "excluded_outcomes": "phecodes 327.x (sleep disorders, incl. apnea) and 347.x "
                                 "(narcolepsy/cataplexy) are excluded throughout as "
                                 "circular with the sleep-study index event.",
        },
        "model_m4": {
            "estimator": "Cox proportional hazards (lifelines), ridge-penalised",
            "penalizer": "0.01, falling back to 0.1 only if that fit fails; the value "
                         "used is recorded per row. A scale-free L2 penalty shrinks a "
                         "rare binary dummy far harder than a continuous covariate, which "
                         "is what attenuated the corner-archetype models ~65%.",
            "covariates": ["age (natural cubic spline, 2 df)", "sex",
                           "study year (centred continuous)", "race/ethnicity",
                           "smoking status", "income group",
                           "baseline observation window (z)",
                           "baseline encounter rate (z)", "BMI (z)"],
            "not_adjusted": "CPAP is deliberately not adjusted (it is a mediator on the "
                            "OSA -> treatment -> outcome path).",
            "caveat": "the octants are strongly structured by age and sex, almost entirely "
                      "through the comorbidity axis (c3 high vs low: 56.6 vs 44.8 years, "
                      "56.6% vs 31.1% female). Every model here adjusts for an age spline "
                      "and sex, but no UNADJUSTED cross-octant comparison -- including the "
                      "curves in this export -- should be read as clinical.",
        },
        "testing": {
            "scheme": "hierarchical. Stage 1 is one 7-df joint Wald test per outcome (do "
                      "the eight octants differ at all?). Stage 2 runs one-vs-rest for "
                      "each octant against the pooled other seven, and ONLY for outcomes "
                      "whose stage-1 omnibus reached p < 0.05.",
            "omnibus_gate_alpha": 0.05,
            "multiplicity": "BH-FDR and Bonferroni applied within each (contrast, model) "
                            "group, so the omnibus is corrected across outcomes on its "
                            "own and each octant's one-vs-rest tests are corrected across "
                            "the gated outcomes (6 at system level, 15 at phecode level). "
                            "Bonferroni therefore means p < 0.05/6 and p < 0.05/15 "
                            "respectively -- NOT 0.05/168.",
            "q_values": ("reconstructed with monotone BH inside the same groups the "
                         "pipeline corrected in, and accepted only after verifying that "
                         "q<=0.05 reproduces the stored sig_fdr flag for every row"
                         if aud["q_fdr_verified"].count("True") == 2 else
                         "NOT verifiable against the stored flags; column left null"),
            "outcomes_gated": {"phecode": "15 of 368", "system": "6 of 18"},
        },
        "unavailable_or_excluded": [
            {"field": "ph_p", "reason": "the proportional-hazards (Schoenfeld) screen is "
                                        "not computed for the octant models; the column "
                                        "is present and null throughout."},
            {"field": "CPAP initiation and adherence distribution",
             "reason": "cpap_usage.parquet covers 92,373 of 157,025 spine patients and "
                       "the data dictionary does not establish whether an absent row "
                       "means never-started or not-captured; mean_pap_usage units are "
                       "flagged UNCONFIRMED at source. No defensible denominator."},
            {"field": "race/ethnicity breakdown",
             "reason": "excluded by the disclosure rules; one cell in the octant Table 1 "
                       "falls below the threshold and the breakdown is not approved."},
            {"field": "adjusted (model-based) curves",
             "reason": "out of scope by design -- the curves are descriptive and "
                        "unadjusted; the adjusted estimates are the m4 hazard ratios."},
            {"field": "overall cumulative-incidence confidence intervals",
             "reason": "the canonical tie-aware Aalen-Johansen implementation does not "
                       "provide a scientifically validated competing-risk variance; no "
                       "intervals were invented or approximated."},
        ],
        "files": [],
        "validation": {"status": "FAIL" if failed else "PASS",
                       "n_checks": len(res), "n_failed": len(failed), "checks": res},
    }

    for f in sorted(OUT.iterdir()):
        if f.name.startswith("phenotype_website_export_manifest"):
            continue
        b = f.read_bytes()
        e = {"name": f.name, "bytes": len(b),
             "sha256": hashlib.sha256(b).hexdigest(),
             "internal": f.name.startswith("_")}
        if f.suffix == ".csv":
            e["rows"] = sum(1 for _ in f.open(encoding="utf-8")) - 1
        man["files"].append(e)

    _preserve_manifest_extensions(man, previous_manifest)
    manifest_path.write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'CHECK':<6}{'STATUS':<8}DESCRIPTION")
    for c in res:
        print(f"{c['check']:<6}{c['status']:<8}{c['description']}")
        if c["status"] == "FAIL" or c["detail"] not in ("clean", ""):
            print(f"{'':<14}-> {c['detail']}")
    print(f"\nvalidation: {man['validation']['status']} "
          f"({len(res) - len(failed)}/{len(res)} checks)")
    print(f"{len(man['files'])} files hashed, "
          f"{sum(f['bytes'] for f in man['files'])/1e6:.2f} MB total")
    if failed:
        return 1
    U.log_milestone(
        f"Octant website export validated ({len(res)}/{len(res)} checks) and manifested: "
        f"{len(man['files'])} files in phenotype_exposure/website_exports/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
