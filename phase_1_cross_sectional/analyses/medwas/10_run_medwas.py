# %% [markdown]
# Step 10 - MedWAS association engine (binary "filled" ~ OSA severity, per GPI-4 class).
#
# Mirrors labwas/10_run_labwas.py but with a BINARY whole-cohort outcome and LOGISTIC models
# (not Poisson): every one of the 222,109 cohort patients is 1 (filled the class in the
# window) or 0 (not). Per (gpi4 x window):
#   - unadjusted screen: chi2 omnibus (4 severity bands), Cochran-Armitage trend,
#     2x2 crude ORs for mild/moderate/severe_vs_none, ahi_ge5, ahi_ge15.
#   - adjusted nested logistic M1 crude / M3 (+demo+smoking+SES+obs_window) / M4 (+BMI);
#     factor-severity gives a Wald omnibus + 3 pairwise ORs; ordinal severity gives a trend
#     OR; ahi_ge5 / ahi_ge15 give dichotomized ORs. Effect = OR = exp(beta).
# BH-FDR + Bonferroni across classes within each (window, contrast, model) family.
#
# ENGINE = "mle" (default; classical logit, clean CIs -- appropriate because the >=500-patient
# floor makes every tested class common, so the separation that motivated ICD's L1 is absent).
# Set ENGINE = "l1" for penalized logit to exactly match the ICD PheDAS.
#
# Usage: python 10_run_medwas.py [smoke_n]   (smoke_n = test on the N most-prevalent classes)

# %%
import os
# L1 fit_regularized is GIL-bound -> parallelize across PROCESSES, one BLAS thread each
# (must be set before numpy imports so spawned workers inherit it).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import medwas_utils as U

ENGINE = "l1"         # "l1" (pyPheWAS default, matches ICD PheDAS) | "mle"
L1_ALPHA = 0.1        # pyPheWAS default penalty weight if ENGINE == "l1"
PHE_THRESH = 5        # min positives to attempt a fit
MAX_PAR = 20          # processes (host has 24 cores; L1 fit_regularized is CPU-bound)
UNSTABLE = 10.0       # |beta| or se above this -> flag unstable

CONTRASTS_PAIR = [("mild_vs_none", "Mild"), ("moderate_vs_none", "Moderate"),
                  ("severe_vs_none", "Severe")]


# ---- cohort + covariate design (built once) --------------------------------
def prep_cohort() -> pd.DataFrame:
    c = pd.read_parquet(U.COHORT)
    c["id"] = c["id"].astype(str)
    for col in ["age", "bmi", "obs_window"]:
        x = c[col].astype(float)
        c[f"{col}_z"] = (x - x.mean()) / x.std()
    c["sex_M"] = c["sex_M"].astype(int)
    c["ahi_ge5"] = (c["AHI"] >= 5).astype(int)
    c["ahi_ge15"] = (c["AHI"] >= 15).astype(int)
    c["severity"] = pd.Categorical(c["severity"], ["None", "Mild", "Moderate", "Severe"])
    c["severity_ord"] = c["severity_ord"].astype(int)
    keep = ["id", "severity", "severity_ord", "ahi_ge5", "ahi_ge15",
            "age_z", "sex_M", "bmi_z", "obs_window_z",
            "RACE_ETH", "study_year", "smoking_status", "income_gp"]
    c = c[keep].dropna().reset_index(drop=True)
    return c


def _dummies(s: pd.Series, ref: str, prefix: str) -> pd.DataFrame:
    d = pd.get_dummies(s.astype(str), prefix=prefix)
    ref_col = f"{prefix}_{ref}"
    if ref_col in d.columns:
        d = d.drop(columns=[ref_col])
    return d.astype(float)


def build_design(c: pd.DataFrame) -> dict:
    """Precompute exposure blocks + covariate blocks (numpy) shared across all classes."""
    n = len(c)
    intercept = np.ones((n, 1))

    # exposure blocks
    sev = c["severity"].astype(str)
    factor = np.column_stack([(sev == lv).astype(float) for lv in ["Mild", "Moderate", "Severe"]])
    exp = {
        "factor": (factor, ["Mild", "Moderate", "Severe"]),
        "trend": (c["severity_ord"].to_numpy(float).reshape(-1, 1), ["severity_ord"]),
        "ahi_ge5": (c["ahi_ge5"].to_numpy(float).reshape(-1, 1), ["ahi_ge5"]),
        "ahi_ge15": (c["ahi_ge15"].to_numpy(float).reshape(-1, 1), ["ahi_ge15"]),
    }

    # covariate blocks (each already includes the intercept as its last column)
    race_ref = c["RACE_ETH"].astype(str).value_counts().idxmax()
    year_ref = str(sorted(c["study_year"].unique())[0])
    demo = np.column_stack([
        c["age_z"].to_numpy(float), c["sex_M"].to_numpy(float),
        _dummies(c["RACE_ETH"], race_ref, "race").to_numpy(),
        _dummies(c["study_year"], year_ref, "yr").to_numpy(),
    ])
    ses = np.column_stack([
        _dummies(c["smoking_status"], "Never", "smk").to_numpy(),
        _dummies(c["income_gp"], "Mid", "inc").to_numpy(),
        c["obs_window_z"].to_numpy(float),
    ])
    m3 = np.column_stack([demo, ses, intercept])
    m4 = np.column_stack([demo, ses, c["bmi_z"].to_numpy(float), intercept])
    cov = {"m1": intercept, "m3": m3, "m4": m4}
    return {"exp": exp, "cov": cov, "n": n,
            "sev_ord": c["severity_ord"].to_numpy(int)}


# ---- fitting ---------------------------------------------------------------
def _fit(y, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y, X)
        if ENGINE == "l1":  # exact pyPheWAS fit (matches the ICD PheDAS)
            return model.fit_regularized(method="l1", alpha=L1_ALPHA, disp=0,
                                         trim_mode="size", qc_verbose=0)
        return model.fit(disp=0, maxiter=100)


def _or_row(res, idx):
    beta = float(res.params[idx]); se = float(res.bse[idx])
    ci = res.conf_int()
    lo, hi = float(ci[idx][0]), float(ci[idx][1])
    p = float(res.pvalues[idx])
    unstable = "" if (abs(beta) < UNSTABLE and se < UNSTABLE) else "unstable"
    return dict(effect_type="OR", effect=np.exp(beta), se=se,
                ci_low=np.exp(lo), ci_high=np.exp(hi), p=p, unstable=unstable)


def cochran_armitage(y, scores):
    """Trend of a binary y across ordinal `scores`; returns (z, two-sided p)."""
    lv = np.unique(scores)
    n_i = np.array([(scores == k).sum() for k in lv], float)
    x_i = np.array([y[scores == k].sum() for k in lv], float)
    N = n_i.sum(); X = x_i.sum(); pbar = X / N
    t = lv.astype(float)
    num = np.sum(t * (x_i - n_i * pbar))
    var = pbar * (1 - pbar) * (np.sum(n_i * t ** 2) - (np.sum(n_i * t) ** 2) / N)
    if var <= 0:
        return (np.nan, np.nan)
    z = num / np.sqrt(var)
    return (z, 2 * stats.norm.sf(abs(z)))


_D = None  # per-worker shared design (set by _init_worker)


def _init_worker(D):
    global _D
    _D = D


def run_class(gpi4, member_idx):
    """All contrasts x models for one GPI-4 class. Returns list of result dicts."""
    D = _D
    n = D["n"]
    y = np.zeros(n, dtype=float)
    if len(member_idx):
        y[member_idx] = 1.0
    n_filled = int(y.sum())
    rows = []

    def base(contrast, model, **kw):
        return dict(target="filled", code=gpi4, contrast=contrast, model=model,
                    n=n, n_filled=n_filled, **kw)

    if n_filled <= PHE_THRESH or n_filled >= n:
        return [base("omnibus", "unadjusted", effect_type="skip", effect=np.nan,
                     se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, unstable="low_count")]

    sev = D["sev_ord"]
    # ---- unadjusted screen ----
    tab = np.array([[ (y[sev == k] == 0).sum() for k in range(4)],
                    [ (y[sev == k] == 1).sum() for k in range(4)]], float)
    try:
        chi2, pc, _, _ = stats.chi2_contingency(tab)
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        rows.append(base("omnibus", "unadjusted", effect_type="cramers_v", effect=v,
                         se=np.nan, ci_low=np.nan, ci_high=np.nan, p=pc, unstable=""))
    except Exception as e:
        rows.append(base("omnibus", "unadjusted", effect_type="cramers_v", effect=np.nan,
                         se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, unstable=str(e)[:40]))
    z, pt = cochran_armitage(y, sev)
    rows.append(base("trend", "unadjusted", effect_type="CA_z", effect=z,
                     se=np.nan, ci_low=np.nan, ci_high=np.nan, p=pt, unstable=""))
    for nm, lv in CONTRASTS_PAIR:
        k = ["None", "Mild", "Moderate", "Severe"].index(lv)
        mask = np.isin(sev, [0, k])
        rows.append(_crude_or(base, nm, y[mask], sev[mask] == k))
    rows.append(_crude_or(base, "ahi_ge5", y, D["exp"]["ahi_ge5"][0][:, 0]))
    rows.append(_crude_or(base, "ahi_ge15", y, D["exp"]["ahi_ge15"][0][:, 0]))

    # ---- adjusted nested logistic ----
    for model in ("m1", "m3", "m4"):
        cov = D["cov"][model]
        # factor severity -> omnibus + 3 pairwise
        Xf = np.column_stack([D["exp"]["factor"][0], cov])
        try:
            res = _fit(y, Xf)
            R = np.zeros((3, Xf.shape[1])); R[0, 0] = R[1, 1] = R[2, 2] = 1
            if ENGINE == "mle":
                w = res.wald_test(R, scalar=True)
                rows.append(base("omnibus", model, effect_type="wald_chi2",
                                 effect=float(np.ravel(w.statistic)[0]), se=np.nan,
                                 ci_low=np.nan, ci_high=np.nan, p=float(w.pvalue), unstable=""))
            for j, (nm, _lv) in enumerate(CONTRASTS_PAIR):
                rows.append(base(nm, model, **_or_row(res, j)))
        except Exception as e:
            rows.append(base("omnibus", model, effect_type="wald_chi2", effect=np.nan,
                             se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, unstable=str(e)[:40]))
        # ordinal trend
        _fit_single(rows, base, "trend", model, y, D["exp"]["trend"][0], cov)
        _fit_single(rows, base, "ahi_ge5", model, y, D["exp"]["ahi_ge5"][0], cov)
        _fit_single(rows, base, "ahi_ge15", model, y, D["exp"]["ahi_ge15"][0], cov)
    return rows


def _crude_or(base, nm, y, exposure):
    """2x2 crude OR + chi2 p for a binary exposure."""
    e = exposure.astype(bool)
    a = float((y[e] == 1).sum()); b = float((y[e] == 0).sum())
    cc = float((y[~e] == 1).sum()); d = float((y[~e] == 0).sum())
    if min(a, b, cc, d) == 0:
        orr = np.nan; p = np.nan; unstable = "zero_cell"
    else:
        orr = (a * d) / (b * cc)
        _, p, _, _ = stats.chi2_contingency(np.array([[a, b], [cc, d]]))
        unstable = ""
    return base(nm, "unadjusted", effect_type="OR", effect=orr, se=np.nan,
                ci_low=np.nan, ci_high=np.nan, p=p, unstable=unstable)


def _fit_single(rows, base, contrast, model, y, exp_col, cov):
    X = np.column_stack([exp_col, cov])
    try:
        res = _fit(y, X)
        rows.append(base(contrast, model, **_or_row(res, 0)))
    except Exception as e:
        rows.append(base(contrast, model, effect_type="OR", effect=np.nan, se=np.nan,
                         ci_low=np.nan, ci_high=np.nan, p=np.nan, unstable=str(e)[:40]))


# ---- multiple testing ------------------------------------------------------
def add_multiplicity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sig_bon"] = False; df["sig_fdr"] = False
    for _, idx in df.groupby(["contrast", "model"]).groups.items():
        g = df.loc[idx]
        p = g["p"].to_numpy(float)
        ok = np.isfinite(p)
        m = ok.sum()
        if m == 0:
            continue
        df.loc[g.index[ok], "sig_bon"] = p[ok] < 0.05 / m
        order = np.argsort(p[ok]); ps = p[ok][order]
        crit = 0.05 * (np.arange(1, m + 1)) / m
        passed = ps <= crit
        thr = ps[passed].max() if passed.any() else -1
        df.loc[g.index[ok], "sig_fdr"] = p[ok] <= thr
    return df


def main(smoke_n: int | None = None) -> None:
    U.ensure_dirs()
    print(f"Engine: {ENGINE}")
    c = prep_cohort()
    id_index = {v: i for i, v in enumerate(c["id"].to_numpy())}
    D = build_design(c)
    print(f"cohort used: {D['n']:,} (rows with complete covariates)")

    for label, _years in U.WINDOWS:
        tgt = U.ANALYSIS / f"targets_{label}.parquet"
        kept = pd.read_csv(U.ANALYSIS / f"kept_classes_{label}.csv", dtype={"gpi4": str})
        classes = kept["gpi4"].astype(str).tolist()
        if smoke_n:
            classes = classes[:smoke_n]
        members = (pd.read_parquet(tgt)
                   .assign(gpi4=lambda d: d["gpi4"].astype(str), id=lambda d: d["id"].astype(str))
                   .groupby("gpi4")["id"].apply(list).to_dict())

        print(f"\n[{label}] fitting {len(classes)} classes x 7 contrasts x 4 models "
              f"({ENGINE}, {MAX_PAR} procs) ...", flush=True)
        out_rows = []
        with ProcessPoolExecutor(max_workers=MAX_PAR, initializer=_init_worker, initargs=(D,)) as ex:
            futs = {}
            for g in classes:
                mids = members.get(g, [])
                idx = np.fromiter((id_index[i] for i in mids if i in id_index), dtype=np.int64)
                futs[ex.submit(run_class, g, idx)] = g
            done = 0
            for f in as_completed(futs):
                out_rows.extend(f.result())
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(classes)}", flush=True)

        res = pd.DataFrame(out_rows)
        res["window"] = label
        res = add_multiplicity(res)
        res["neglog10p"] = -np.log10(res["p"].clip(lower=1e-300))
        cols = ["window", "target", "code", "contrast", "model", "n", "n_filled",
                "effect_type", "effect", "se", "ci_low", "ci_high", "p", "neglog10p",
                "sig_bon", "sig_fdr", "unstable"]
        res = res[cols].sort_values(["contrast", "model", "p"], na_position="last")
        outp = U.ANALYSIS / f"results_{label}.csv"
        res.to_csv(outp, index=False)
        n_sig = res.query("model=='m4' and contrast=='trend' and sig_fdr").shape[0]
        print(f"  wrote {outp} ({len(res):,} rows); m4 trend FDR-sig classes: {n_sig}")

    U.log_milestone(f"Step 10 MedWAS ({ENGINE}) logistic: results_1yr/5yr written.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
