"""Gaussian mixture (latent profile) with full/diag covariance and FIML missing data.

StepMix 3.0.0 only ships ``_nan`` (FIML) emission models for the *unit*, *spherical*
and *diagonal* covariance structures -- there is no ``gaussian_full_nan``. The c1
physiologic block has within-class correlations up to r=0.91, where a diagonal
mixture over-extracts classes to compensate for the unmodelled covariance, and it
also carries a feature that is 73% missing by design, so listwise deletion is not an
option either. This module supplies the missing combination.

Estimation is the standard EM for a Gaussian mixture under MAR (Ghahramani & Jordan):
the E-step evaluates each row against the *observed* sub-vector of its component,
and the M-step completes the missing entries with their conditional expectation plus
the conditional covariance correction. Rows are grouped by missingness pattern so
each (pattern, component) pair needs a single Cholesky per iteration.

MAR caveat: FIML assumes missingness is ignorable given the modelled variables.
Where that is doubtful (c1's supine ratio is missing for 100% of WatchPAT studies)
the result must be checked against a refit with the offending feature dropped --
see ``analysis/02_c1_fit.py``, which reports that comparison as an ARI.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

__all__ = ["FimlGaussianMixture"]

_LOG2PI = float(np.log(2.0 * np.pi))


class FimlGaussianMixture:
    """EM for a K-component Gaussian mixture tolerating NaNs (MAR / FIML).

    Parameters
    ----------
    n_components : number of latent profiles.
    covariance_type : ``"full"`` (own general covariance per class) or ``"diag"``.
    reg_covar : ridge added to each covariance diagonal for numerical stability.
    tol : relative tolerance on the mean log-likelihood between iterations.
    max_iter, n_init, random_state : usual EM controls. ``n_init`` random starts are
        run and the highest-likelihood solution is kept.
    init : ``"kmeans"`` (k-means++ style seeding on mean-imputed data) or ``"random"``.
    """

    def __init__(self, n_components=2, covariance_type="full", reg_covar=1e-6,
                 tol=1e-6, max_iter=500, n_init=1, random_state=None,
                 init="kmeans", verbose=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.init = init
        self.verbose = verbose

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _patterns(X):
        """Group row indices by missingness pattern.

        Returns a list of (observed_column_index_array, row_index_array), ordered by
        descending group size so the dominant patterns are handled first.
        """
        miss = np.isnan(X)
        codes, inv = np.unique(miss, axis=0, return_inverse=True)
        groups = []
        for p in range(len(codes)):
            rows = np.flatnonzero(inv == p)
            obs = np.flatnonzero(~codes[p])
            groups.append((obs, rows))
        groups.sort(key=lambda g: -len(g[1]))
        return groups

    def _init_params(self, X, groups, rng):
        n, d = X.shape
        K = self.n_components
        # Mean-impute once purely to seed the starting values; the EM itself never
        # sees the imputed matrix.
        col_mean = np.nanmean(X, axis=0)
        Xi = np.where(np.isnan(X), col_mean, X)

        if self.init == "kmeans":
            # k-means++ seeding, then one assignment pass -> starting means.
            idx = [rng.integers(n)]
            d2 = ((Xi - Xi[idx[0]]) ** 2).sum(1)
            for _ in range(K - 1):
                probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
                nxt = int(rng.choice(n, p=probs))
                idx.append(nxt)
                d2 = np.minimum(d2, ((Xi - Xi[nxt]) ** 2).sum(1))
            centers = Xi[idx]
            lab = np.argmin(((Xi[:, None, :] - centers[None]) ** 2).sum(2), axis=1) \
                if n * K * d < 5e7 else self._chunk_assign(Xi, centers)
            resp = np.zeros((n, K))
            resp[np.arange(n), lab] = 1.0
            resp += 1e-3
            resp /= resp.sum(1, keepdims=True)
        else:
            resp = rng.dirichlet(np.ones(K), size=n)

        self.weights_ = resp.mean(0)
        self.means_ = np.empty((K, d))
        self.covariances_ = np.empty((K, d, d))
        base = np.cov(Xi.T) + self.reg_covar * np.eye(d)
        for k in range(K):
            w = resp[:, k]
            self.means_[k] = (w[:, None] * Xi).sum(0) / w.sum()
            self.covariances_[k] = base.copy()
        if self.covariance_type == "diag":
            for k in range(K):
                self.covariances_[k] = np.diag(np.diag(self.covariances_[k]))

    @staticmethod
    def _chunk_assign(Xi, centers, chunk=20000):
        lab = np.empty(len(Xi), dtype=int)
        for s in range(0, len(Xi), chunk):
            sl = slice(s, s + chunk)
            lab[sl] = np.argmin(((Xi[sl, None, :] - centers[None]) ** 2).sum(2), axis=1)
        return lab

    # ------------------------------------------------------------------ E-step
    def _log_prob(self, X, groups):
        """Per-row, per-component log N(x_obs; mu_obs, Sigma_obs,obs)."""
        n, _ = X.shape
        K = self.n_components
        out = np.zeros((n, K))
        for obs, rows in groups:
            if obs.size == 0:            # nothing observed -> log-density 0 for all k
                continue
            Xo = X[np.ix_(rows, obs)]
            for k in range(K):
                mu = self.means_[k][obs]
                S = self.covariances_[k][np.ix_(obs, obs)]
                c, low = cho_factor(S, lower=True, check_finite=False)
                dev = Xo - mu
                sol = cho_solve((c, low), dev.T, check_finite=False).T
                maha = np.einsum("ij,ij->i", dev, sol)
                logdet = 2.0 * np.log(np.diag(c)).sum()
                out[rows, k] = -0.5 * (obs.size * _LOG2PI + logdet + maha)
        return out

    def _e_step(self, X, groups):
        weighted = self._log_prob(X, groups) + np.log(self.weights_)
        ll_row = logsumexp(weighted, axis=1)
        return np.exp(weighted - ll_row[:, None]), ll_row.mean()

    # ------------------------------------------------------------------ M-step
    def _m_step(self, X, groups, resp):
        n, d = X.shape
        K = self.n_components
        nk = resp.sum(0) + 10 * np.finfo(float).eps
        new_mean = np.zeros((K, d))
        new_cov = np.zeros((K, d, d))

        for k in range(K):
            mu_k, S_k = self.means_[k], self.covariances_[k]
            acc1 = np.zeros(d)
            acc2 = np.zeros((d, d))
            for obs, rows in groups:
                r = resp[rows, k]
                rsum = r.sum()
                if rsum <= 0:
                    continue
                mis = np.setdiff1d(np.arange(d), obs, assume_unique=False)
                Y = np.empty((len(rows), d))
                if obs.size:
                    Y[:, obs] = X[np.ix_(rows, obs)]
                if mis.size:
                    if obs.size:
                        Soo = S_k[np.ix_(obs, obs)]
                        Smo = S_k[np.ix_(mis, obs)]
                        c, low = cho_factor(Soo, lower=True, check_finite=False)
                        # B = Sigma_mo Sigma_oo^-1  (regression of missing on observed)
                        B = cho_solve((c, low), Smo.T, check_finite=False).T
                        Y[:, mis] = mu_k[mis] + (X[np.ix_(rows, obs)] - mu_k[obs]) @ B.T
                        cond_cov = S_k[np.ix_(mis, mis)] - B @ Smo.T
                    else:
                        Y[:, mis] = mu_k[mis]
                        cond_cov = S_k[np.ix_(mis, mis)]
                    # the conditional covariance is shared by every row in the pattern
                    acc2[np.ix_(mis, mis)] += rsum * cond_cov
                acc1 += r @ Y
                acc2 += (Y * r[:, None]).T @ Y
            new_mean[k] = acc1 / nk[k]
            new_cov[k] = acc2 / nk[k] - np.outer(new_mean[k], new_mean[k])
            new_cov[k] = 0.5 * (new_cov[k] + new_cov[k].T)          # enforce symmetry
            if self.covariance_type == "diag":
                new_cov[k] = np.diag(np.diag(new_cov[k]))
            new_cov[k].flat[:: d + 1] += self.reg_covar

        self.weights_ = nk / n
        self.means_ = new_mean
        self.covariances_ = new_cov

    # -------------------------------------------------------------------- fit
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        groups = self._patterns(X)
        best = None
        for start in range(self.n_init):
            rng = np.random.default_rng(
                None if self.random_state is None else self.random_state + start)
            self._init_params(X, groups, rng)
            prev = -np.inf
            for it in range(self.max_iter):
                resp, ll = self._e_step(X, groups)
                self._m_step(X, groups, resp)
                if self.verbose and it % 25 == 0:
                    print(f"    start {start} iter {it:4d} ll/n {ll:.8f}")
                if abs(ll - prev) < self.tol * max(1.0, abs(prev)):
                    break
                prev = ll
            resp, ll = self._e_step(X, groups)
            if best is None or ll > best[0]:
                best = (ll, self.weights_.copy(), self.means_.copy(),
                        self.covariances_.copy(), it + 1)
        self.lower_bound_, self.weights_, self.means_, self.covariances_, self.n_iter_ = best
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        self.converged_ = self.n_iter_ < self.max_iter
        self._groups = groups
        return self

    # ------------------------------------------------------------- inspection
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._e_step(X, self._patterns(X))[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def complete(self, X):
        """Fill missing entries with their posterior-weighted conditional expectation.

        E[x_mis | x_obs] = sum_k P(k | x_obs) * (mu_mis + Sigma_mo Sigma_oo^-1
        (x_obs - mu_obs)) -- the same quantity the EM M-step already forms. Needed
        because any downstream method that cannot take NaN (a discriminant
        projection, UMAP, t-SNE) would otherwise fall back to complete cases, and
        in c1 that silently drops every WatchPAT study: the supine ratio is missing
        for 100% of them, so "complete case" means "one device".
        """
        X = np.asarray(X, dtype=float)
        groups = self._patterns(X)
        resp, _ = self._e_step(X, groups)
        out = X.copy()
        d = X.shape[1]
        for obs, rows in groups:
            mis = np.setdiff1d(np.arange(d), obs)
            if mis.size == 0:
                continue
            acc = np.zeros((len(rows), mis.size))
            for k in range(self.n_components):
                mu_k, S_k = self.means_[k], self.covariances_[k]
                if obs.size:
                    c, low = cho_factor(S_k[np.ix_(obs, obs)], lower=True,
                                        check_finite=False)
                    B = cho_solve((c, low), S_k[np.ix_(mis, obs)].T,
                                  check_finite=False).T
                    pred = mu_k[mis] + (X[np.ix_(rows, obs)] - mu_k[obs]) @ B.T
                else:
                    pred = np.tile(mu_k[mis], (len(rows), 1))
                acc += resp[rows, k][:, None] * pred
            out[np.ix_(rows, mis)] = acc
        return out

    def score(self, X):
        """Mean log-likelihood per observation."""
        X = np.asarray(X, dtype=float)
        return self._e_step(X, self._patterns(X))[1]

    @property
    def n_parameters(self):
        K, d = self.n_components, self.n_features_in_
        per_cov = d if self.covariance_type == "diag" else d * (d + 1) // 2
        return int(K * d + K * per_cov + (K - 1))

    def _ic(self, X):
        n = len(X)
        ll = self.score(X) * n
        return ll, n, self.n_parameters

    def bic(self, X):
        ll, n, p = self._ic(X)
        return -2 * ll + p * np.log(n)

    def aic(self, X):
        ll, _, p = self._ic(X)
        return -2 * ll + 2 * p

    def caic(self, X):
        ll, n, p = self._ic(X)
        return -2 * ll + p * (np.log(n) + 1)

    def sabic(self, X):
        ll, n, p = self._ic(X)
        return -2 * ll + p * np.log((n + 2) / 24)

    def relative_entropy(self, X):
        """Normalised entropy E_K in [0,1]; 1 = perfectly separated classes."""
        P = self.predict_proba(X)
        if self.n_components == 1:
            return 1.0
        ent = -(P * np.log(np.clip(P, 1e-300, None))).sum()
        return float(1.0 - ent / (len(X) * np.log(self.n_components)))
