"""Aggregate curve/metadata export for the website's octant-phenotype section (A, B, C).

Script 25 draws the cumulative-incidence panels but keeps the coordinates in memory and
saves only the figures plus the 3-year endpoints, so the site cannot redraw them
interactively. This writes the same aggregate coordinates out.

Scope is every *omnibus-gated* one-vs-rest contrast at m4, not just the ones that survive
Bonferroni: for each outcome whose 7-df omnibus clears p<0.05 the site can show all eight
octants, with `currently_published` marking the 21 that are drawn in the paper figures.

ESTIMATOR. The Aalen-Johansen implementation moved into `_pheno_engine.aalen_johansen`,
which is tie-aware; script 25's local version walked observations one at a time and so
depended on the row order among tied times. This script audits the difference rather than
changing it silently -- it recomputes every curve under both, checks the tie-aware version
against lifelines' Kaplan-Meier through the identity CIF_event + CIF_death = 1 - S(t), and
tests permutation invariance. The audit numbers land in the manifest.

PHI: reads patient-level frames in memory only. Every file written and every number
printed is a group-level aggregate; exact counts below DISCLOSURE_MIN are suppressed to
null with a `suppressed` flag rather than replaced by zero. No identifier, no date, no
per-patient row leaves this process.

Usage:  python 27_octant_website_export.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incwas_utils as U  # noqa: E402
import _cox_engine as E  # noqa: E402
import _pheno_engine as P  # noqa: E402

OUT = P.OUT / "website_exports"
HORIZON = 3.0
LEGACY_GRID = np.linspace(0.0, HORIZON, 120)  # only for validating the previous export
RISK_TIMES = [0.0, 1.0, 2.0, 3.0]
OMNI_ALPHA = 0.05                 # the gate script 24 applied
DISCLOSURE_MIN = 11
EVENT_DEF = {
    "phecode": "first incident phecode (rule-of-2 date) among patients free of the "
               "phecode and its control-exclusion family at index",
    "system": "first NEW phecode in the body system with the whole cohort at risk "
              "(`anynew`) -- a diagnostic-accrual hazard, not first-ever disease onset",
}
SELECTION = ("all eight octants for every outcome whose m4 omnibus (7-df joint Wald) "
             "reached p < 0.05; one-vs-rest = focal octant against the pooled other seven")
OVERALL_ESTIMATOR = "Aalen-Johansen, tie-aware"
OVERALL_COMPETING_EVENT = "death"
OVERALL_TIME_ORIGIN = "index sleep study"


# ---------------------------------------------------------------- estimator audit
def _legacy_sequential_aj(t, ev, death, grid):
    """Script 25's original estimator, kept ONLY to measure what changed.

    Walks observations in `argsort` order, so within a tied time it decrements the risk
    set and updates S between individuals. Not called by the export.
    """
    o = np.argsort(t)
    t, ev, death = np.asarray(t)[o], np.asarray(ev)[o], np.asarray(death)[o]
    n = len(t)
    S, cif, out, at = 1.0, 0.0, [], n
    j = 0
    for g in grid:
        while j < n and t[j] <= g:
            d_e, d_d = int(ev[j] == 1), int(death[j] == 1 and ev[j] == 0)
            if at > 0 and (d_e or d_d):
                cif += S * d_e / at
                S *= (1 - (d_e + d_d) / at)
            at -= 1
            j += 1
        out.append(cif)
    return np.array(out)


def audit_estimator(curves: list[dict], rng: np.random.Generator) -> dict:
    """Tie-aware vs legacy-sequential vs lifelines, over every exported curve."""
    from lifelines import KaplanMeierFitter

    d_legacy, d_km, d_perm, n_tied, n_obs = [], [], [], 0, 0
    for c in curves:
        t, ev, dth = c["t"], c["event"], c["is_death"]
        grid = c["grid"]
        new = c["cif"]
        d_legacy.append(np.abs(new - _legacy_sequential_aj(t, ev, dth, grid)).max())

        # identity check against a tie-aware reference: the two cause-specific CIFs must
        # sum to the all-cause Kaplan-Meier failure function at every grid point. The
        # death CIF has to be computed with the phecode event as ITS competing risk, so
        # both curves share the same all-cause S(t-) -- hence the swapped arguments.
        cif_d = P.aalen_johansen(t, (ev != 1) & (dth == 1), (ev == 1), grid)
        km = KaplanMeierFitter().fit(t, (ev == 1) | (dth == 1))
        s = km.survival_function_at_times(grid).to_numpy()
        d_km.append(np.abs((new + cif_d) - (1.0 - s)).max())

        o = rng.permutation(len(t))
        d_perm.append(np.abs(new - P.aalen_johansen(t[o], ev[o], dth[o], grid)).max())
        n_tied += len(t) - len(np.unique(t))
        n_obs += len(t)

    return {
        "estimator": "Aalen-Johansen, tie-aware (all transitions at a time share that "
                     "time's risk set and one S(t-) multiplier); death competing",
        "tie_handling": "grouped by unique time; censoring at an event time stays in that "
                        "time's risk set",
        "curves_audited": len(curves),
        "tied_observations_pct": round(100 * n_tied / max(n_obs, 1), 2),
        "max_abs_diff_vs_legacy_sequential_pp": float(np.round(100 * max(d_legacy), 6)),
        "mean_abs_diff_vs_legacy_sequential_pp": float(
            np.round(100 * float(np.mean(d_legacy)), 6)),
        "max_abs_identity_residual_vs_lifelines_km_pp": float(
            np.round(100 * max(d_km), 12)),
        "max_abs_diff_under_row_permutation_pp": float(np.round(100 * max(d_perm), 12)),
        "deterministic": bool(max(d_perm) == 0.0),
        "note": "lifelines' AalenJohansenFitter jitters tied durations to break them, so "
                "it is a randomised tie-breaker rather than a tie-aware reference; the "
                "check used instead is the exact identity against lifelines' tie-aware "
                "Kaplan-Meier.",
    }


# ------------------------------------------------------------------------ helpers
def bh_q(p: np.ndarray) -> np.ndarray:
    """Monotone Benjamini-Hochberg q-values; q<=0.05 reproduces `add_fdr_bonferroni`."""
    p = np.asarray(p, dtype=float)
    n = p.size
    o = np.argsort(p, kind="mergesort")
    raw = n * p[o] / np.arange(1, n + 1)
    q = np.empty(n)
    q[o] = np.minimum.accumulate(raw[::-1])[::-1]
    return np.clip(q, 0.0, 1.0)


def add_q(r: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Reconstruct q within the same (contrast, model) groups script 24 corrected in.

    Returned flag says whether q<=0.05 reproduces the stored `sig_fdr` for every row; if
    it does not, the caller nulls the column rather than publishing a q it cannot verify.
    """
    r = r.copy()
    r["q_fdr"] = np.nan
    for _, ix in r[r.p.notna()].groupby(["contrast", "model"]).groups.items():
        r.loc[ix, "q_fdr"] = bh_q(r.loc[ix, "p"].to_numpy())
    ok = r[r.p.notna()]
    return r, bool(((ok.q_fdr <= 0.05) == ok.sig_fdr).all())


def suppress(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int, int]:
    """Blank exact counts below threshold; return suppressed row and cell counts."""
    df = df.copy()
    mask = pd.Series(False, index=df.index)
    cells = 0
    for c in cols:
        m = df[c].notna() & (df[c] < DISCLOSURE_MIN)
        df.loc[m, c] = np.nan
        mask |= m
        cells += int(m.sum())
    existing = (df["suppressed"].fillna(False).astype(bool)
                if "suppressed" in df else pd.Series(False, index=df.index))
    df["suppressed"] = existing | mask
    for c in cols:
        df[c] = df[c].astype("Int64")
    return df, int(df["suppressed"].sum()), cells


def file_sha256(path: Path) -> str | None:
    """SHA-256 for an aggregate export, or None when no baseline file exists."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_previous_curves(path, updated, keys):
    """Require the same CIF at every previous sample before replacing its grid."""
    old = pd.read_csv(path, dtype={"outcome": str}, float_precision="round_trip")
    previous = dict(list(old.groupby(keys, sort=False)))
    current = dict(list(updated.groupby(keys, sort=False)))
    if previous.keys() != current.keys():
        raise RuntimeError(f"{path.name}: curve membership changed")
    maximum_error = 0.0
    for key, prior in previous.items():
        prior = prior.sort_values("time_years")
        new = current[key].sort_values("time_years")
        times = prior.time_years.to_numpy()
        if len(times) == len(LEGACY_GRID) and np.array_equal(times, np.round(LEGACY_GRID, 6)):
            times = LEGACY_GRID
        pos = np.searchsorted(new.time_years.to_numpy(), times, side="right") - 1
        values = new.cif_pct.to_numpy()[pos]
        if not np.array_equal(np.isnan(values), prior.cif_pct.isna().to_numpy()):
            raise RuntimeError(f"{path.name}: withheld status changed")
        if np.isfinite(values).any():
            maximum_error = max(maximum_error, float(np.nanmax(np.abs(values - prior.cif_pct))))
    if maximum_error > 0.00000051:
        raise RuntimeError(f"{path.name}: previous sampled estimates changed")
    return maximum_error


def verify_unchanged_table(path, updated):
    """Keep existing aggregate tables byte-for-byte after checking regenerated values."""
    previous = pd.read_csv(path, dtype={"outcome": str}, keep_default_na=False)
    import io
    current = pd.read_csv(io.StringIO(updated.to_csv(index=False)),
                          dtype={"outcome": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(previous, current, check_dtype=False, check_exact=False,
                                  atol=1e-12, rtol=0)
    return True


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    overall_rng = np.random.default_rng(1)
    focal_curves_path = OUT / "octant_cif_curves.csv"
    focal_sha_before = file_sha256(focal_curves_path)

    d = P.load_exposure(("c1_physiologic", "c2_symptomatic", "c3_comorbidities"),
                        with_archetype=False)
    d = P.assign_octants(d)
    E._PREP, E._ID2IX = (d, {}), {v: i for i, v in enumerate(d["id"].to_numpy())}
    prev, excl = E.prev_index(), E.exclude_map()
    ev_all = pd.read_parquet(U.EVENTS)
    ev_all["phecode"] = ev_all["phecode"].astype(str)
    ev_all = ev_all[ev_all["id"].isin(set(d["id"]))]
    ev_all = ev_all[~ev_all["phecode"].map(P.is_circular_outcome)]
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {x.phecode: (x.phenotype, x.category) for x in dim.itertuples()}
    print(f"cohort {len(d):,} patients on the 3-domain incwas spine", flush=True)

    curve_rows, meta_rows, risk_rows, audit_curves, gate = [], [], [], [], {}
    overall_curve_rows, overall_meta_rows, overall_risk_rows = [], [], []
    overall_audit_curves, overall_direct_keys = [], []
    overall_perm_curve_diff, overall_perm_risk_equal = [], []
    q_verified = {}
    for level in ("phecode", "system"):
        r = pd.read_csv(P.OUT / f"results_octant_{level}.csv", dtype={"outcome": str})
        r, q_ok = add_q(r)
        q_verified[level] = q_ok
        om = r[(r.contrast == "omnibus") & (r.model == "m4")].set_index("outcome")
        gated = om[om.p < OMNI_ALPHA]
        ovr = r[(r.model == "m4") & r.contrast.str.startswith("one_vs_rest")].copy()
        ovr["octant"] = ovr.contrast.str.split(":").str[1]
        assert set(ovr.outcome) == set(gated.index), f"{level}: gate/stage-2 mismatch"
        gate[level] = {"outcomes_tested": len(om), "outcomes_gated": len(gated),
                       "panels": len(ovr), "published": int(ovr.sig_bon.sum())}
        print(f"{level}: {len(gated)} of {len(om)} outcomes gated -> {len(ovr)} panels "
              f"({int(ovr.sig_bon.sum())} Bonferroni-significant)", flush=True)

        for outcome in gated.index:
            if level == "system":
                surv = P.system_events(d, ev_all, dim, outcome, rule="anynew")
                name, cat = outcome, outcome
            else:
                surv = E.build_survival(outcome, ev_all[ev_all.phecode == outcome], d,
                                        prev, excl)
                name, cat = lab.get(outcome, (outcome, ""))
            oct_arr = surv["octant"].to_numpy()
            t = surv["time"].to_numpy(float)
            evn = surv["event"].to_numpy()
            dth = surv["is_death"].to_numpy()

            # Direct full outcome-specific cohort estimate. This deliberately precedes
            # every octant split and is never reconstructed from the eight subgroup CIFs.
            all_grid = P.event_time_grid(t, evn, HORIZON)
            cif_all = P.aalen_johansen(t, evn, dth, all_grid)
            at_all, cum_all = P.risk_counts(t, evn, RISK_TIMES)
            n_baseline = int(at_all[0])
            events_3yr = int(cum_all[-1])
            curve_withheld = n_baseline < DISCLOSURE_MIN or events_3yr < DISCLOSURE_MIN
            curve_status = "withheld" if curve_withheld else "available"
            overall_curve_rows.append(pd.DataFrame({
                "level": level, "outcome": outcome, "outcome_name": name,
                "category": cat, "group": "all",
                "time_years": RISK_TIMES if curve_withheld else all_grid,
                "cif_pct": (np.full(len(RISK_TIMES), np.nan) if curve_withheld else
                            100 * cif_all),
                "curve_status": curve_status, "suppressed": curve_withheld}))
            overall_risk_rows += [
                {"level": level, "outcome": outcome, "outcome_name": name,
                 "category": cat, "group": "all", "time_years": tt,
                 "n_at_risk": int(a), "n_events_cum": int(c)}
                for tt, a, c in zip(RISK_TIMES, at_all, cum_all)]
            overall_meta_rows.append({
                "level": level, "outcome": outcome, "outcome_name": name,
                "category": cat, "n_at_risk_baseline": n_baseline,
                "total_events": int(evn.sum()),
                "cif3_pct": round(100 * float(cif_all[-1]), 4),
                "mean_followup_years": round(float(np.mean(t)), 4),
                "n_at_risk_3yr": int(at_all[-1]),
                "pct_baseline_at_risk_3yr": round(
                    100 * float(at_all[-1]) / n_baseline, 4) if n_baseline else np.nan,
                "curve_status": curve_status, "suppressed": curve_withheld,
                "event_definition": EVENT_DEF[level],
                "estimator": OVERALL_ESTIMATOR,
                "competing_event": OVERALL_COMPETING_EVENT,
                "time_origin": OVERALL_TIME_ORIGIN})
            overall_direct_keys.append((level, outcome))
            overall_audit_curves.append(
                {"t": t, "event": evn, "is_death": dth, "cif": cif_all, "grid": all_grid})

            perm = overall_rng.permutation(len(t))
            cif_perm = P.aalen_johansen(t[perm], evn[perm], dth[perm], all_grid)
            at_perm, cum_perm = P.risk_counts(t[perm], evn[perm], RISK_TIMES)
            overall_perm_curve_diff.append(float(np.max(np.abs(cif_all - cif_perm))))
            overall_perm_risk_equal.append(
                bool(np.array_equal(at_all, at_perm) and np.array_equal(cum_all, cum_perm)))

            for row in ovr[ovr.outcome == outcome].itertuples():
                for group, m in (("focal", oct_arr == row.octant),
                                 ("other_seven", oct_arr != row.octant)):
                    grid = P.event_time_grid(t[m], evn[m], HORIZON)
                    cif = P.aalen_johansen(t[m], evn[m], dth[m], grid)
                    curve_rows.append(pd.DataFrame({
                        "level": level, "outcome": outcome, "outcome_name": name,
                        "category": cat, "octant": row.octant, "group": group,
                        "time_years": grid, "cif_pct": 100 * cif}))
                    at, cum = P.risk_counts(t[m], evn[m], RISK_TIMES)
                    risk_rows += [{"level": level, "outcome": outcome,
                                   "octant": row.octant, "group": group,
                                   "time_years": tt, "n_at_risk": int(a),
                                   "n_events_cum": int(c)}
                                  for tt, a, c in zip(RISK_TIMES, at, cum)]
                    audit_curves.append({"t": t[m], "event": evn[m], "is_death": dth[m],
                                         "cif": cif, "grid": grid})
                    if group == "focal":
                        cif_f, n_f, ev_f = cif, int(m.sum()), int(evn[m].sum())
                    else:
                        cif_r, n_r, ev_r = cif, int(m.sum()), int(evn[m].sum())

                o = om.loc[outcome]
                meta_rows.append({
                    "level": level, "outcome": outcome, "outcome_name": name,
                    "category": cat, "octant": row.octant, "contrast": row.contrast,
                    "model": "m4", "hr_m4": row.effect, "se": row.se,
                    "ci_low": row.ci_low, "ci_high": row.ci_high, "p": row.p,
                    "q_fdr": row.q_fdr, "sig_fdr": bool(row.sig_fdr),
                    "sig_bon": bool(row.sig_bon),
                    "omnibus_p_m4": o.p, "omnibus_q_m4": o.q_fdr,
                    "omnibus_sig_fdr": bool(o.sig_fdr),
                    "n_focal": n_f, "events_focal": ev_f,
                    "n_rest": n_r, "events_rest": ev_r,
                    "cif3_focal_pct": round(100 * float(cif_f[-1]), 4),
                    "cif3_rest_pct": round(100 * float(cif_r[-1]), 4),
                    "penalizer": row.penalizer,
                    "unstable": "" if pd.isna(row.unstable) else str(row.unstable),
                    "ph_p": row.ph_p,
                    "currently_published": bool(row.sig_bon),
                    "selection_scope": SELECTION,
                    "event_definition": EVENT_DEF[level]})

    curves = pd.concat(curve_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    risk = pd.DataFrame(risk_rows)
    overall_curves = pd.concat(overall_curve_rows, ignore_index=True)
    overall_meta_raw = pd.DataFrame(overall_meta_rows)
    overall_risk_raw = pd.DataFrame(overall_risk_rows)
    for level, ok in q_verified.items():
        if not ok:
            meta.loc[meta.level == level, ["q_fdr", "omnibus_q_m4"]] = np.nan
            print(f"WARNING {level}: reconstructed q did not reproduce sig_fdr; nulled")

    audit = audit_estimator(audit_curves + overall_audit_curves, rng)
    print("\nestimator audit: max |tie-aware - legacy sequential| = "
          f"{audit['max_abs_diff_vs_legacy_sequential_pp']:.6f} pp; "
          f"identity residual vs lifelines KM = "
          f"{audit['max_abs_identity_residual_vs_lifelines_km_pp']:.2e} pp; "
          f"permutation-invariant = {audit['deterministic']}; "
          f"tied observations = {audit['tied_observations_pct']}%")

    # Raw focal/rest partitions must reconcile to each directly estimated full cohort.
    panel_key = ["level", "outcome", "octant", "time_years"]
    outcome_time_key = ["level", "outcome", "time_years"]
    group_sets = risk.groupby(panel_key)["group"].agg(lambda x: set(x))
    part_risk = (risk.groupby(panel_key, as_index=False)
                 [["n_at_risk", "n_events_cum"]].sum())
    all_risk_for_join = overall_risk_raw[outcome_time_key +
                                         ["n_at_risk", "n_events_cum"]].rename(
        columns={"n_at_risk": "n_at_risk_all",
                 "n_events_cum": "n_events_cum_all"})
    risk_join = part_risk.merge(all_risk_for_join, on=outcome_time_key, how="left",
                                validate="many_to_one")
    risk_count_diff = pd.concat([
        (risk_join.n_at_risk - risk_join.n_at_risk_all).abs(),
        (risk_join.n_events_cum - risk_join.n_events_cum_all).abs()])
    risk_partition_ok = bool(
        len(risk_join) == len(part_risk)
        and all(x == {"focal", "other_seven"} for x in group_sets)
        and (risk_count_diff == 0).all())

    part_meta = meta[["level", "outcome", "octant", "n_focal", "n_rest",
                      "events_focal", "events_rest"]].copy()
    part_meta["n_all_partition"] = part_meta.n_focal + part_meta.n_rest
    part_meta["events_all_partition"] = part_meta.events_focal + part_meta.events_rest
    meta_join = part_meta.merge(
        overall_meta_raw[["level", "outcome", "n_at_risk_baseline", "total_events"]],
        on=["level", "outcome"], how="left", validate="many_to_one")
    meta_count_diff = pd.concat([
        (meta_join.n_all_partition - meta_join.n_at_risk_baseline).abs(),
        (meta_join.events_all_partition - meta_join.total_events).abs()])
    meta_partition_ok = bool(len(meta_join) == len(part_meta)
                             and (meta_count_diff == 0).all())
    overall_raw_partition_reconciled = risk_partition_ok and meta_partition_ok
    overall_raw_partition_max_abs_count_diff = int(max(
        risk_count_diff.max(), meta_count_diff.max()))

    overall_risk_ordered = overall_risk_raw.sort_values("time_years")
    overall_raw_risk_nonincreasing = bool(overall_risk_ordered.groupby(
        ["level", "outcome", "group"])["n_at_risk"].apply(
            lambda s: s.is_monotonic_decreasing).all())
    overall_raw_events_nondecreasing = bool(overall_risk_ordered.groupby(
        ["level", "outcome", "group"])["n_events_cum"].apply(
            lambda s: s.is_monotonic_increasing).all())

    octant_keys = set(map(tuple, meta[["level", "outcome"]].drop_duplicates().to_numpy()))
    overall_keys = set(overall_direct_keys)
    overall_keys_match_octant_metadata = overall_keys == octant_keys
    overall_direct_estimation = bool(
        len(overall_direct_keys) == len(overall_keys) == len(overall_meta_raw))
    overall_max_perm = max(overall_perm_curve_diff, default=0.0)
    overall_max_perm_pp = float(np.round(100 * overall_max_perm, 12))
    overall_risk_permutation_deterministic = bool(all(overall_perm_risk_equal))
    overall_deterministic = bool(overall_max_perm == 0.0
                                 and overall_risk_permutation_deterministic)

    meta, n_sup_meta_rows, n_sup_meta_cells = suppress(
        meta, ["n_focal", "events_focal", "n_rest", "events_rest"])
    risk_raw = risk.copy()
    risk, n_sup_risk_rows, n_sup_risk_cells = suppress(
        risk, ["n_at_risk", "n_events_cum"])

    overall_risk, n_sup_overall_risk_rows, n_sup_overall_risk_cells = suppress(
        overall_risk_raw, ["n_at_risk", "n_events_cum"])
    overall_meta = overall_meta_raw.copy()
    baseline_count_suppressed = (
        overall_meta_raw.n_at_risk_baseline < DISCLOSURE_MIN)
    at_risk_3yr_suppressed = overall_meta_raw.n_at_risk_3yr < DISCLOSURE_MIN
    overall_meta, _, n_sup_overall_meta_count_cells = suppress(
        overall_meta, ["n_at_risk_baseline", "total_events", "n_at_risk_3yr"])
    pct_null = baseline_count_suppressed | at_risk_3yr_suppressed
    cif_null = overall_meta.curve_status == "withheld"
    n_sup_overall_meta_derived_cells = int(
        (pct_null & overall_meta.pct_baseline_at_risk_3yr.notna()).sum()
        + (cif_null & overall_meta.cif3_pct.notna()).sum())
    overall_meta.loc[pct_null, "pct_baseline_at_risk_3yr"] = np.nan
    overall_meta.loc[cif_null, "cif3_pct"] = np.nan
    overall_meta["suppressed"] = overall_meta.suppressed | pct_null | cif_null
    n_sup_overall_meta_rows = int(overall_meta.suppressed.sum())
    n_sup_overall_meta_cells = (n_sup_overall_meta_count_cells
                                + n_sup_overall_meta_derived_cells)
    n_sup_overall_curve_rows = int(overall_curves.suppressed.sum())
    n_sup_overall_curve_cells = int(overall_curves.cif_pct.isna().sum())

    expected_curve_cols = ["level", "outcome", "outcome_name", "category", "group",
                           "time_years", "cif_pct", "curve_status", "suppressed"]
    expected_risk_cols = ["level", "outcome", "outcome_name", "category", "group",
                          "time_years", "n_at_risk", "n_events_cum", "suppressed"]
    expected_meta_cols = [
        "level", "outcome", "outcome_name", "category", "n_at_risk_baseline",
        "total_events", "cif3_pct", "mean_followup_years", "n_at_risk_3yr",
        "pct_baseline_at_risk_3yr", "curve_status", "suppressed",
        "event_definition", "estimator", "competing_event", "time_origin"]
    overall_per_curve = overall_curves.groupby(["level", "outcome"], sort=False)
    available = overall_curves[overall_curves.curve_status == "available"]
    withheld = overall_curves[overall_curves.curve_status == "withheld"]
    endpoint = overall_curves[overall_curves.time_years == HORIZON][
        ["level", "outcome", "cif_pct", "curve_status"]].merge(
            overall_meta[["level", "outcome", "cif3_pct"]],
            on=["level", "outcome"], how="left", validate="one_to_one")
    endpoint_available = endpoint[endpoint.curve_status == "available"]
    endpoint_diff = ((endpoint_available.cif_pct - endpoint_available.cif3_pct).abs().max()
                     if len(endpoint_available) else 0.0)
    overall_checks = {
        "schemas_exact": (list(overall_curves.columns) == expected_curve_cols
                          and list(overall_risk.columns) == expected_risk_cols
                          and list(overall_meta.columns) == expected_meta_cols),
        "outcomes_exact": len(overall_keys) == 21 and overall_keys_match_octant_metadata,
        "row_counts_exact": (len(overall_per_curve) == 21
                             and len(overall_risk) == 84 and len(overall_meta) == 21),
        "group_all": (set(overall_curves.group) == {"all"}
                      and set(overall_risk.group) == {"all"}),
        "event_time_grid_valid": bool(all(
            points.time_years.iloc[0] == 0.0 and points.time_years.iloc[-1] == HORIZON
            and (np.diff(points.time_years) > 0).all()
            and set(RISK_TIMES).issubset(points.time_years)
            for _, points in overall_per_curve)),
        "curves_start_at_zero": bool(
            (overall_per_curve.time_years.min() == 0.0).all()
            and (overall_per_curve.cif_pct.first().dropna() == 0.0).all()),
        "cif_in_bounds": bool(available.cif_pct.between(0.0, 100.0).all()),
        "cif_nondecreasing": bool(all(
            np.diff(c["cif"]).min() >= -1e-12 for c in overall_audit_curves)),
        "withheld_cif_null": bool(withheld.cif_pct.isna().all()),
        "cif3_reconciles": bool(endpoint_diff <= 0.0001),
        "raw_n_at_risk_nonincreasing": overall_raw_risk_nonincreasing,
        "raw_n_events_cum_nondecreasing": overall_raw_events_nondecreasing,
        "raw_partition_reconciled": overall_raw_partition_reconciled,
        "row_permutation_deterministic": overall_deterministic,
    }
    failed_overall_checks = [k for k, ok in overall_checks.items() if not ok]
    if failed_overall_checks:
        raise RuntimeError("Overall aggregate validation failed: "
                           + ", ".join(failed_overall_checks))

    # Validate the former sampled estimates and unchanged scientific summaries before
    # publishing denser coordinates. All patient-level frames remain inside this process.
    focal_difference = compare_previous_curves(
        focal_curves_path, curves, ["level", "outcome", "octant", "group"])
    overall_curves_path = OUT / "octant_cif_overall_curves.csv"
    overall_difference = compare_previous_curves(
        overall_curves_path, overall_curves, ["level", "outcome", "group"])
    unchanged_tables = {
        "focal_metadata_unchanged": verify_unchanged_table(OUT / "octant_cif_metadata.csv", meta),
        "focal_risk_table_unchanged": verify_unchanged_table(OUT / "octant_cif_risk_table.csv", risk),
        "overall_metadata_unchanged": verify_unchanged_table(OUT / "octant_cif_overall_metadata.csv", overall_meta),
        "overall_risk_table_unchanged": verify_unchanged_table(OUT / "octant_cif_overall_risk_table.csv", overall_risk),
    }
    focal_candidate = curves.to_csv(index=False).encode("utf-8")
    focal_candidate_sha = hashlib.sha256(focal_candidate).hexdigest()
    focal_curves_path.write_bytes(focal_candidate)

    overall_curves_path = OUT / "octant_cif_overall_curves.csv"
    overall_risk_path = OUT / "octant_cif_overall_risk_table.csv"
    overall_meta_path = OUT / "octant_cif_overall_metadata.csv"
    overall_curves.to_csv(overall_curves_path, index=False)

    focal_sha_after = file_sha256(focal_curves_path)
    focal_curves_unchanged = bool(focal_sha_before is not None
                                  and focal_sha_before == focal_sha_after)
    overall_files = [overall_curves_path.name, overall_risk_path.name,
                     overall_meta_path.name]
    overall_file_hashes = {p.name: file_sha256(p) for p in
                           (overall_curves_path, overall_risk_path, overall_meta_path)}

    pd.DataFrame([{"key": k, "value": str(v)} for k, v in
                  ({**audit, "curve_granularity": "event_time",
                    "horizon_years": HORIZON,
                    "focal_curve_rows": len(curves),
                    "focal_sampled_equivalent": True,
                    "focal_max_abs_diff_vs_previous_sampled_pp": focal_difference,
                    "overall_sampled_equivalent": True,
                    "overall_max_abs_diff_vs_previous_sampled_pp": overall_difference,
                    **unchanged_tables,
                    "disclosure_min": DISCLOSURE_MIN,
                    "suppressed_rows_metadata": n_sup_meta_rows,
                    "suppressed_cells_metadata": n_sup_meta_cells,
                    "suppressed_rows_risk_table": n_sup_risk_rows,
                    "suppressed_cells_risk_table": n_sup_risk_cells,
                    "q_fdr_verified": q_verified,
                    "gate": gate,
                    "overall_files_produced": overall_files,
                    "overall_outcomes": len(overall_keys),
                    "overall_curve_rows": len(overall_curves),
                    "overall_risk_rows": len(overall_risk),
                    "overall_metadata_rows": len(overall_meta),
                    "overall_direct_estimation": overall_direct_estimation,
                    "overall_direct_estimation_method":
                        "canonical estimator called on complete outcome-specific "
                        "survival frame before any octant split",
                    "overall_keys_match_octant_metadata":
                        overall_keys_match_octant_metadata,
                    "overall_raw_partition_reconciled":
                        overall_raw_partition_reconciled,
                    "overall_raw_partition_max_abs_count_diff":
                        overall_raw_partition_max_abs_count_diff,
                    "overall_raw_n_at_risk_nonincreasing":
                        overall_raw_risk_nonincreasing,
                    "overall_raw_n_events_cum_nondecreasing":
                        overall_raw_events_nondecreasing,
                    "overall_curves_audited": len(overall_audit_curves),
                    "overall_max_abs_diff_under_row_permutation_pp":
                        overall_max_perm_pp,
                    "overall_risk_permutation_deterministic":
                        overall_risk_permutation_deterministic,
                    "overall_risk_counts_permutation_invariant":
                        overall_risk_permutation_deterministic,
                    "overall_deterministic": overall_deterministic,
                    "overall_suppressed_rows_curves": n_sup_overall_curve_rows,
                    "overall_suppressed_cells_curves": n_sup_overall_curve_cells,
                    "overall_suppressed_rows_risk_table":
                        n_sup_overall_risk_rows,
                    "overall_suppressed_cells_risk_table":
                        n_sup_overall_risk_cells,
                    "overall_suppressed_rows_metadata": n_sup_overall_meta_rows,
                    "overall_suppressed_cells_metadata": n_sup_overall_meta_cells,
                    "overall_confidence_intervals_available": False,
                    "overall_confidence_intervals_note":
                        "canonical tie-aware estimator has no validated variance output; "
                        "confidence interval fields omitted",
                    "overall_validation_status": "PASS",
                    "overall_validation_checks": overall_checks,
                    "focal_curves_sha256_before": focal_sha_before,
                    "focal_curves_candidate_sha256": focal_candidate_sha,
                    "focal_curves_sha256_after": focal_sha_after,
                    "focal_curves_unchanged": focal_curves_unchanged,
                    "octant_cif_overall_curves_sha256":
                        overall_file_hashes[overall_curves_path.name],
                    "octant_cif_overall_risk_table_sha256":
                        overall_file_hashes[overall_risk_path.name],
                    "octant_cif_overall_metadata_sha256":
                        overall_file_hashes[overall_meta_path.name]}).items()]).to_csv(
        OUT / "_survival_export_audit.csv", index=False)

    print(f"\noctant_cif_curves.csv     {len(curves):,} rows")
    print(f"octant_cif_metadata.csv   {len(meta):,} rows "
          f"({n_sup_meta_cells} suppressed count cells)")
    print(f"octant_cif_risk_table.csv {len(risk):,} rows "
          f"({n_sup_risk_cells} suppressed count cells)")
    print(f"octant_cif_overall_curves.csv     {len(overall_curves):,} rows "
          f"({n_sup_overall_curve_rows} suppressed rows)")
    print(f"octant_cif_overall_risk_table.csv {len(overall_risk):,} rows "
          f"({n_sup_overall_risk_cells} suppressed count cells)")
    print(f"octant_cif_overall_metadata.csv   {len(overall_meta):,} rows "
          f"({n_sup_overall_meta_cells} suppressed cells)")
    mono = all((risk_raw.sort_values("time_years").groupby(
        ["level", "outcome", "octant", "group"])["n_at_risk"].apply(
        lambda s: s.is_monotonic_decreasing)))
    print(f"n_at_risk nonincreasing in every panel-group: {mono}")
    print(f"3-year at-risk share of baseline: "
          f"{risk_raw[risk_raw.time_years == 3].n_at_risk.sum() / risk_raw[risk_raw.time_years == 0].n_at_risk.sum():.1%}")
    U.log_milestone(
        f"Octant website export (A/B/C): {len(curves):,} CIF coordinate rows over "
        f"{len(meta)} omnibus-gated one-vs-rest panels plus {len(overall_meta)} direct "
        f"full-cohort curves, tie-aware Aalen-Johansen -> "
        f"phenotype_exposure/website_exports/.")


if __name__ == "__main__":
    main()
