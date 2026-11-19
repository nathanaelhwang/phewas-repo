"""Tie-aware Aalen-Johansen pointwise inference and horizon-matched log-rank tests."""
import numpy as np
from scipy.stats import norm
from lifelines.statistics import logrank_test


def aalen_johansen_ci(t, event, is_death, grid, alpha=0.05):
    """Return CIF, variance, and log-log pointwise bounds on the supplied grid.

    Propagate the covariance of (event-free survival, cause-1 incidence) through
    each grouped transition using multinomial hazard-increment covariance. This
    is the tie-aware Greenwood / infinitesimal-jackknife variance. Censoring at
    an event time remains in its risk set. No event times are jittered.
    """
    t = np.asarray(t, float)
    if not t.size:
        return tuple(np.zeros(len(grid)) for _ in range(4))
    e1 = np.asarray(event) == 1
    e2 = (~e1) & (np.asarray(is_death) == 1)
    times, inverse, removed = np.unique(t, return_inverse=True, return_counts=True)
    risk = len(t) - np.r_[0, removed.cumsum()[:-1]]
    d1 = np.bincount(inverse, weights=e1, minlength=len(times))
    d2 = np.bincount(inverse, weights=e2, minlength=len(times))
    survival, cif, vss, vff, vsf = 1., 0., 0., 0., 0.
    estimates, variances = [], []
    for y, x1, x2 in zip(risk, d1, d2):
        a, b = x1 / y, x2 / y
        remain = 1. - a - b
        next_vff = vff + a*a*vss + 2*a*vsf + survival**2*a*(1-a)/y
        next_vsf = remain*(vsf + a*vss) - survival**2*a*remain/y
        next_vss = remain**2*vss + survival**2*(a+b)*remain/y
        cif += survival*a
        survival *= remain
        vss, vff, vsf = next_vss, max(0., next_vff), next_vsf
        estimates.append(cif)
        variances.append(vff)
    positions = np.searchsorted(times, grid, side="right") - 1
    estimate = np.where(positions >= 0, np.asarray(estimates)[positions.clip(0)], 0.)
    variance = np.where(positions >= 0, np.asarray(variances)[positions.clip(0)], 0.)
    low, high = estimate.copy(), estimate.copy()
    interior = (estimate > 0) & (estimate < 1)
    f = estimate[interior]
    transformed = np.log(-np.log(f))
    margin = norm.ppf(1-alpha/2)*np.sqrt(variance[interior]) / (f*np.abs(np.log(f)))
    low[interior] = np.exp(-np.exp(transformed+margin))
    high[interior] = np.exp(-np.exp(transformed-margin))
    return estimate, variance, low, high


def cause_specific_logrank(t, event, focal, horizon=3.):
    """Unadjusted two-sided 1-df test, death censored, through the plot horizon."""
    t = np.asarray(t, float)
    observed = (np.asarray(event) == 1) & (t <= horizon)
    t = np.minimum(t, horizon)
    focal = np.asarray(focal, bool)
    result = logrank_test(t[focal], t[~focal], observed[focal], observed[~focal])
    return float(result.p_value)
