"""Model fitting, class enumeration and stability -- shared by all three domains.

The three domains use different measurement models (c1: FIML Gaussian mixture,
c2: mixed nested StepMix, c3: Bernoulli StepMix) but must be enumerated and
validated identically, so the K-sweep, the selection rule and the stability checks
live here and take an opaque ``fit_fn(K, X, seed) -> model`` callback.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from .config import (LONG_ITER, N_JOBS, N_LONG_STARTS, N_SHORT_STARTS, SELECTION,
                     SEED, SHORT_ITER, SHORT_SUBSAMPLE)


@dataclass
class FitRow:
    K: int
    loglik: float
    n_params: int
    bic: float
    aic: float
    caic: float
    sabic: float
    rel_entropy: float
    min_class_frac: float
    min_class_n: int
    min_component_sd: float
    degenerate: bool
    n_iter: int
    converged: bool
    ll_replicated: bool
    seconds: float


# A Gaussian mixture likelihood is unbounded: a component that collapses onto a
# point mass drives the log-likelihood to +infinity and makes BIC meaningless.
# This is not hypothetical here -- c1's T80%/SPOT_LT80 carry a 56% exact-zero atom
# that captures a whole component, and StepMix's gaussian_diag_nan returns a silent
# NaN log-likelihood in the same situation rather than raising. Every fit is
# therefore health-checked before it is allowed into the fit-index table.
# Chosen to separate true degeneracy from a merely tight class: a collapsed
# component sits at the reg_covar floor (SD 0.001), whereas the tightest admissible
# c1 solutions observed in piloting sit at 0.017-0.096.
MIN_COMPONENT_SD = 0.01          # on the standardized scale


def _collect_covariances(node, out: list) -> None:
    """Walk a StepMix parameter tree, collecting every Gaussian covariance array.

    A nested (mixed-type) measurement model returns ``{'gaussian': {...},
    'binary': {...}, 'nominal': {...}}``, so the covariances sit one level down and a
    flat key lookup silently finds nothing.
    """
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        if isinstance(v, dict):
            _collect_covariances(v, out)
        elif "cov" in k.lower() or k.lower() in ("variances", "vars"):
            a = np.asarray(v, dtype=float)
            if a.size:
                out.append(a)


def component_sds(model) -> np.ndarray:
    """Per-component, per-indicator SDs for whichever mixture we were given.

    Returns ``inf`` when the model has no Gaussian block at all (a pure Bernoulli
    LCA cannot suffer variance collapse), so callers can treat the check uniformly.
    """
    cov = getattr(model, "covariances_", None)
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 3:                    # (K, d, d) full covariance
            cov = np.stack([np.diag(c) for c in cov])
        return np.sqrt(np.clip(cov, 0, None))
    try:
        found: list = []
        _collect_covariances(model.get_parameters().get("measurement", {}), found)
    except Exception:
        return np.array([np.inf])
    if not found:
        return np.array([np.inf])            # no Gaussian block -> nothing to collapse
    return np.sqrt(np.clip(np.concatenate([a.ravel() for a in found]), 0, None))


def health_check(model, X) -> tuple[float, bool]:
    """Return (min component SD, degenerate?) for a fitted mixture.

    A model with no Gaussian block (a pure Bernoulli LCA) cannot suffer variance
    collapse at all, so it reports NaN for the SD and is judged only on whether its
    log-likelihood is finite.
    """
    sds = component_sds(model)
    finite = sds[np.isfinite(sds)]
    ll = model.score(X)
    if finite.size == 0:
        return float("nan"), bool(not np.isfinite(ll))
    mn = float(finite.min())
    return mn, bool((not np.isfinite(ll)) or mn < MIN_COMPONENT_SD)


def _short_run(make_model, Xs, K, sd, short_iter):
    # short starts are screening runs: they are *meant* to stop before convergence,
    # so StepMix's ConvergenceWarning here is expected and would drown the log
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_model(K, sd, max_iter=short_iter)
        m.fit(Xs)
        return float(m.score(Xs)), sd


def _long_run(make_model, X, K, sd, long_iter):
    m = make_model(K, sd, max_iter=long_iter)
    m.fit(X)
    return float(m.score(X)), m


def two_stage_fit(make_model, X, K, seed=SEED, n_short=N_SHORT_STARTS,
                  short_iter=SHORT_ITER, n_long=N_LONG_STARTS, long_iter=LONG_ITER,
                  short_subsample=SHORT_SUBSAMPLE, n_jobs=N_JOBS, verbose=False):
    """Many cheap short starts, then run the best few to convergence.

    A plain ``n_init`` sweep at these sample sizes spends almost all its time
    driving hopeless starts to convergence. Screening on the short-run likelihood
    first is the standard Mplus ``STARTS m n`` idiom and reaches the same optimum
    for a fraction of the cost. The screening stage additionally runs on a random
    subsample -- it only has to *rank* starting points, and at n >= 74,000 a
    25,000-row subsample ranks them just as well for a tenth of the compute.

    Returns ``(best_model, ll_replicated)`` where ``ll_replicated`` is True when
    the top log-likelihood is reached by more than one independent start -- the
    conventional evidence that the maximum is global rather than a local trap.
    """
    seeds = [seed + 1000 * s for s in range(n_short)]
    if short_subsample and len(X) > short_subsample:
        idx = np.random.default_rng(seed).choice(len(X), short_subsample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    short = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_short_run)(make_model, Xs, K, sd, short_iter) for sd in seeds)
    short.sort(key=lambda t: -t[0])

    finals = Parallel(n_jobs=min(n_jobs, n_long), prefer="processes")(
        delayed(_long_run)(make_model, X, K, sd, long_iter)
        for _, sd in short[:n_long])
    finals.sort(key=lambda t: -t[0])
    if verbose:
        for ll, _ in finals:
            print(f"      long-start ll/n = {ll:.8f}")

    best_ll = finals[0][0]
    # "replicated" = at least one other start lands on the same maximum, judged on a
    # relative tolerance so the criterion means the same thing across domains whose
    # per-observation log-likelihoods differ by orders of magnitude
    tol = 1e-5 * max(1.0, abs(best_ll))
    replicated = sum(1 for ll, _ in finals if abs(ll - best_ll) < tol) >= 2
    return finals[0][1], replicated


def sweep(make_model, X, k_range, seed=SEED, verbose=True, **start_kw) -> tuple[pd.DataFrame, dict]:
    """Fit every K in ``k_range`` and assemble the fit-index table.

    ``start_kw`` is forwarded to :func:`two_stage_fit`, so a sensitivity arm can be
    run on a smaller start budget than the primary sweep -- an arm only has to be
    good enough to support an ARI comparison, not to establish the global optimum.
    """
    rows, models = [], {}
    for K in k_range:
        t0 = time.perf_counter()
        model, replicated = two_stage_fit(make_model, X, K, seed=seed, **start_kw)
        el = time.perf_counter() - t0
        P = model.predict_proba(X) if K > 1 else np.ones((len(X), 1))
        sizes = P.argmax(1)
        frac = np.bincount(sizes, minlength=K) / len(X)
        min_sd, degen = health_check(model, X)
        rows.append(FitRow(
            K=K, loglik=float(model.score(X) * len(X)), n_params=int(model.n_parameters),
            bic=float(model.bic(X)), aic=float(model.aic(X)), caic=float(model.caic(X)),
            sabic=float(model.sabic(X)),
            rel_entropy=float(model.relative_entropy(X)) if K > 1 else 1.0,
            min_class_frac=float(frac.min()), min_class_n=int((frac * len(X)).min()),
            min_component_sd=round(min_sd, 6), degenerate=degen,
            n_iter=int(getattr(model, "n_iter_", -1)),
            converged=bool(getattr(model, "converged_", True)),
            ll_replicated=bool(replicated), seconds=round(el, 1),
        ))
        models[K] = model
        if verbose:
            r = rows[-1]
            flag = "  *** DEGENERATE ***" if degen else ""
            print(f"    K={K}  ll={r.loglik:,.0f}  BIC={r.bic:,.0f}  relE={r.rel_entropy:.3f}  "
                  f"min={r.min_class_frac:.4f}  minSD={r.min_component_sd:.4f}  "
                  f"repl={r.ll_replicated}  {el:.0f}s{flag}")
    return pd.DataFrame([r.__dict__ for r in rows]), models


def bic_elbow(bic: np.ndarray, ks: np.ndarray, threshold=None) -> int:
    """Smallest K whose next step buys less than ``threshold`` of the first drop.

    At n = 74k-243k the information criteria fall essentially monotonically in K,
    so the minimum is uninformative. The elbow rule asks instead where the curve
    stops paying for itself, scaled by the K=1->2 improvement so the threshold
    means the same thing across domains of very different likelihood magnitude.
    """
    threshold = SELECTION["bic_elbow_threshold"] if threshold is None else threshold
    drops = -np.diff(bic)                       # drops[i] = gain going ks[i] -> ks[i+1]
    if len(drops) == 0 or drops[0] <= 0:
        return int(ks[0])
    rel = drops / drops[0]
    for i, r in enumerate(rel):
        if r < threshold:
            return int(ks[i])                   # step i+1 was not worth taking
    return int(ks[-1])


def apply_selection(fit: pd.DataFrame, stability: pd.DataFrame | None = None,
                    rule=SELECTION) -> pd.DataFrame:
    """Score every K against the pre-declared rule; returns the table with flags."""
    out = fit.copy()
    # A degenerate fit has a collapsed component, so its likelihood is unbounded and
    # its BIC is not a number that can be compared with anything. Those rows are
    # excluded from the elbow search entirely -- leaving them in lets a meaningless
    # BIC define the ceiling, which is how a variance collapse turns into a
    # "recommended" K.
    clean = out.loc[~out.degenerate.fillna(False)] if "degenerate" in out else out
    if len(clean) < 2:
        clean = out
    elbow = bic_elbow(clean.bic.to_numpy(), clean.K.to_numpy(),
                      rule["bic_elbow_threshold"])
    out["bic_elbow_K"] = elbow
    out["n_admissible_K"] = int(len(clean))
    out["max_non_degenerate_K"] = int(clean.K.max())
    # Expose the quantity the elbow rule actually thresholds, so a reader can see
    # whether the ceiling bound or simply never fired within the K range searched.
    rel = pd.Series(np.nan, index=out.index)
    drops = -np.diff(clean.bic.to_numpy())
    if len(drops) and drops[0] > 0:
        rel.loc[clean.index[:-1]] = drops / drops[0]
    out["bic_drop_rel_to_first"] = rel.round(4)
    out["elbow_ceiling_fired"] = elbow < int(clean.K.max())
    out["ok_entropy"] = out.rel_entropy >= rule["min_relative_entropy"]
    out["ok_class_size"] = ((out.min_class_frac >= rule["min_class_frac"])
                            & (out.min_class_n >= rule["min_class_n"]))
    out["ok_replicated"] = out.ll_replicated
    # a degenerate fit has an unbounded (or NaN) likelihood, so its BIC is not a
    # number that can be compared with anything -- such a K cannot be selected
    out["ok_not_degenerate"] = ~out.degenerate.fillna(False)
    if stability is not None:
        out = out.merge(stability[["K", "split_half_ari"]], on="K", how="left")
        out["ok_stability"] = out.split_half_ari >= rule["min_split_half_ari"]
    else:
        out["split_half_ari"] = np.nan
        out["ok_stability"] = np.nan
    out["n_criteria_met"] = out[["ok_entropy", "ok_class_size", "ok_replicated",
                                 "ok_stability", "ok_not_degenerate"]].sum(axis=1,
                                                                           skipna=True)
    out["at_or_below_elbow"] = out.K <= elbow
    # recommended = largest K that is at/below the elbow and passes every criterion.
    # If nothing qualifies, no K is recommended -- "K=1, no admissible latent
    # subgroups" is a real result and the rule must be able to return it.
    ok = out[out.at_or_below_elbow & out.ok_entropy & out.ok_class_size
             & out.ok_replicated & out.ok_not_degenerate
             & (out.ok_stability.fillna(True)) & (out.K > 1)]
    out["recommended"] = False
    if len(ok):
        out.loc[out.K == ok.K.max(), "recommended"] = True
    return out


def _one_split(make_model, X, K, seed, rep, max_iter):
    rng = np.random.default_rng(seed + 7919 * rep)
    perm = rng.permutation(len(X))
    a, b = perm[: len(X) // 2], perm[len(X) // 2:]
    ma = make_model(K, seed + 7919 * rep, max_iter=max_iter)
    ma.fit(X[a])
    mb = make_model(K, seed + 104729 * rep, max_iter=max_iter)
    mb.fit(X[b])
    return float(adjusted_rand_score(ma.predict(X[b]), mb.predict(X[b])))


def split_half_stability(make_model, X, k_range, n_rep=10, seed=SEED,
                         max_iter=200, n_jobs=N_JOBS, verbose=True):
    """Fit on a random half, score the other half, compare to a native fit there.

    ARI between (a) the held-out half classified by the training solution and
    (b) the same half classified by a solution fitted on itself. A K whose classes
    are real reproduces across independent halves; a K that is splitting noise does
    not. Fewer iterations than the main sweep -- this measures reproducibility of
    the solution, not the global optimum.
    """
    rows = []
    for K in k_range:
        if K == 1:
            rows.append({"K": 1, "split_half_ari": 1.0, "ari_sd": 0.0})
            continue
        aris = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_one_split)(make_model, X, K, seed, r, max_iter)
            for r in range(n_rep))
        rows.append({"K": K, "split_half_ari": float(np.mean(aris)),
                     "ari_sd": float(np.std(aris))})
        if verbose:
            print(f"    stability K={K}  ARI={rows[-1]['split_half_ari']:.3f} "
                  f"(sd {rows[-1]['ari_sd']:.3f})")
    return pd.DataFrame(rows)


def class_profile(X: pd.DataFrame, labels: np.ndarray, K: int) -> pd.DataFrame:
    """Per-class indicator means with the cohort mean and a standardized effect."""
    rows = []
    overall = X.mean()
    sd = X.std().replace(0, np.nan)
    for k in range(K):
        m = X[labels == k].mean()
        rows.append(pd.DataFrame({
            "class": k, "indicator": X.columns, "mean": m.to_numpy(),
            "cohort_mean": overall.to_numpy(),
            "std_diff": ((m - overall) / sd).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)
