"""External validation and artifact diagnostics.

AHI is never a model input. It enters only here, through 3-step estimation with the
BCH correction, which relates latent class to a distal outcome while accounting for
classification error -- so the outcome cannot influence how the classes formed.

Also holds the per-domain artifact checks that the prior clustering run's findings
made necessary: device composition for c1, intake-form-version alignment for c2,
and the healthcare-utilisation gradient for c3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

from .config import AHI_SEVERITY_BINS, AHI_SEVERITY_LABELS


def bch_weights(P: np.ndarray) -> np.ndarray:
    """Empirical BCH weights (Vermunt 2010; Bakk, Tekle & Vermunt 2013).

    Modal assignment misclassifies patients at a rate the posteriors already
    describe. D[s, t] = P(assigned = t | latent = s) is estimated from the
    posteriors, and each patient assigned to t then contributes weight
    ``inv(D)[t, s]`` to class s. Class means computed with these weights are
    consistent for the true class means, whereas naive modal-assignment means are
    biased toward the grand mean whenever entropy is below 1.

    Returns an (n, K) weight matrix; note the weights are not all positive, which
    is expected for BCH and is why weighted *variances* need care.
    """
    n, K = P.shape
    W = np.zeros_like(P)
    W[np.arange(n), P.argmax(1)] = 1.0
    D = (P.T @ W) / P.sum(0)[:, None]           # rows: latent class, cols: assigned
    return W @ np.linalg.inv(D)


def distal_bch(P: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """BCH-corrected class means/SDs of a distal outcome, plus the naive version."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    P, y = P[ok], y[ok]
    Wb = bch_weights(P)
    modal = P.argmax(1)
    rows = []
    for k in range(P.shape[1]):
        w = Wb[:, k]
        sw = w.sum()
        mu = float((w * y).sum() / sw)
        var = float((w * (y - mu) ** 2).sum() / sw)
        m = modal == k
        rows.append({
            "class": k,
            "n_modal": int(m.sum()),
            "bch_mean": round(mu, 3),
            "bch_sd": round(np.sqrt(var), 3) if var > 0 else np.nan,
            "naive_mean": round(float(y[m].mean()), 3) if m.any() else np.nan,
            "naive_median": round(float(np.median(y[m])), 3) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def kw_effect(y: np.ndarray, labels: np.ndarray) -> dict:
    """Kruskal-Wallis on the distal outcome across modal classes, with eps^2.

    Reported alongside the BCH means as the assumption-light comparator; the BCH
    estimates are the ones to quote.
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y, labels = y[ok], labels[ok]
    groups = [y[labels == k] for k in np.unique(labels)]
    if len(groups) < 2:
        return {"H": np.nan, "p": np.nan, "epsilon_sq": np.nan, "n": int(len(y))}
    H, p = kruskal(*groups)
    n = len(y)
    return {"H": float(H), "p": float(p),
            "epsilon_sq": float((H - len(groups) + 1) / (n - len(groups))),
            "n": int(n)}


def severity_mix(ahi: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Row-normalised AHI severity composition per class."""
    sev = pd.cut(pd.Series(ahi), bins=AHI_SEVERITY_BINS, labels=AHI_SEVERITY_LABELS,
                 right=False)
    return pd.crosstab(pd.Series(labels, name="class"), sev, normalize="index").round(4)


def cramers_v(a, b) -> tuple[float, pd.DataFrame]:
    """Cramer's V between two categorical vectors, with the contingency table."""
    tab = pd.crosstab(pd.Series(a).astype(str), pd.Series(b).astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan"), tab
    chi2 = chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))), tab


def composition(labels: np.ndarray, factor, name: str) -> pd.DataFrame:
    """Row-normalised composition of ``factor`` within each class.

    Used for c1's STUDY_DEVICE check (the prior run's k>=3 solutions spun a
    lab-PSG device artifact) and for c2's intake-form signature.
    """
    tab = pd.crosstab(pd.Series(labels, name="class"),
                      pd.Series(np.asarray(factor), name=name), normalize="index")
    return tab.round(4)


def agreement_with_prior(new_labels: np.ndarray, mrn: pd.Series,
                         prior_path, prior_col=None) -> dict:
    """ARI / adjusted agreement against the prior PPCA + k-means partition.

    Not a validity check -- the prior solution is not ground truth -- but a
    concrete statement of how much the model-based classes differ from the
    distance-based ones on the same patients.
    """
    from sklearn.metrics import adjusted_rand_score
    prior = pd.read_csv(prior_path, dtype={"MRN": str})
    if prior_col is None:
        cands = [c for c in prior.columns
                 if c.lower() in ("cluster", "label", "assignment", "cluster_id")]
        prior_col = cands[0] if cands else prior.columns[-1]
    merged = pd.DataFrame({"MRN": mrn.to_numpy(), "new": new_labels}).merge(
        prior[["MRN", prior_col]], on="MRN", how="inner")
    return {
        "n_overlap": int(len(merged)),
        "prior_column": prior_col,
        "ari": float(adjusted_rand_score(merged[prior_col], merged["new"])),
        "prior_k": int(merged[prior_col].nunique()),
        "new_k": int(merged["new"].nunique()),
    }
