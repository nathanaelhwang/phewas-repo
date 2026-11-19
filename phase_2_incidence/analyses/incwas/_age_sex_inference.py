"""Horizon-matched cause-specific inference for the age/sex curve extension."""
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from lifelines.statistics import multivariate_logrank_test
from statsmodels.stats.multitest import multipletests


def truncate(data, horizon=6):
    data = data.copy()
    data["event"] = ((data.etype == 1) & (data.time <= horizon)).astype(int)
    data["time"] = data.time.clip(upper=horizon)
    return data


def infer(data, group_col, order, reference, covariates, metadata):
    """Use plotted groups only; do not replace a missing reference group."""
    data = truncate(data)
    eligible = []
    for group in order:
        sub = data[data[group_col] == group]
        if len(sub) >= 20 and sub.event.sum() >= 5:
            eligible.append(group)
    data = data[data[group_col].isin(eligible)].copy()
    n_events = int(data.event.sum())
    base = {**metadata, "horizon_years": 6, "n_atrisk": len(data), "n_events": n_events,
            "groups_tested": " | ".join(eligible), "n_groups": len(eligible)}
    lr = {**base, "chi2": np.nan, "df": max(0, len(eligible)-1), "p": np.nan,
          "status": "fewer_than_two_plotted_groups"}
    if len(eligible) >= 2:
        result = multivariate_logrank_test(data.time, data[group_col], data.event)
        lr.update(chi2=float(result.test_statistic), p=float(result.p_value), status="ok")
    records = [{**base, "group": group, "reference": reference, "hr": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                "n_parameters": 0, "epv": np.nan, "low_epv": False,
                "status": "curve_omitted" if group not in eligible else "not_fitted"}
               for group in order if group != reference]
    status = None
    if reference not in eligible:
        status = "reference_curve_omitted"
    elif len(eligible) < 2:
        status = "fewer_than_two_plotted_groups"
    elif n_events < 20:
        status = "fewer_than_20_events"
    if status:
        for record in records:
            if record["status"] != "curve_omitted":
                record["status"] = status
        return lr, records
    covariates = [c for c in covariates if data[c].nunique() > 1]
    design = data[["time", "event"] + covariates].astype(float).copy()
    terms = {}
    for number, group in enumerate(eligible):
        if group != reference:
            terms[group] = f"contrast_{number}"
            design[terms[group]] = (data[group_col] == group).astype(float)
    npar = len(design.columns)-2
    for record in records:
        record.update(n_parameters=npar, epv=n_events/npar, low_epv=n_events < 10*npar)
    assert not design.isna().any().any(), "Missing design-matrix covariates"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = CoxPHFitter(penalizer=.1).fit(design, "time", "event", batch_mode=True)
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        for record in records:
            if record["group"] not in terms:
                continue
            row = model.summary.loc[terms[record["group"]]]
            stable = abs(row["coef"]) <= 10 and row["se(coef)"] <= 10
            stable = stable and np.isfinite(row[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]).all()
            record.update(hr=float(row["exp(coef)"]), ci_low=float(row["exp(coef) lower 95%"]),
                          ci_high=float(row["exp(coef) upper 95%"]), p=float(row["p"]),
                          status="convergence_warning" if convergence_warning else ("ok" if stable else "unstable"))
    except (ConvergenceError, np.linalg.LinAlgError) as error:
        for record in records:
            if record["status"] != "curve_omitted":
                record["status"] = type(error).__name__
    return lr, records


def adjust(data):
    """Two explicit families: severity panels and six-demographic-strata plots."""
    data = data.copy()
    data["q_bh"] = np.nan
    data["p_bonferroni"] = np.nan
    data["inference_eligible"] = data.status.eq("ok") & data.p.notna() & ~data.osa_control
    if "low_epv" in data:
        data["inference_eligible"] &= ~data.low_epv
    for _, indices in data[data.inference_eligible].groupby("layout").groups.items():
        values = data.loc[indices, "p"]
        data.loc[indices, "q_bh"] = multipletests(values, method="fdr_bh")[1]
        data.loc[indices, "p_bonferroni"] = np.minimum(1., len(indices)*values)
    data["sig_nominal"] = data.inference_eligible & data.p.lt(.05)
    data["sig_fdr"] = data.inference_eligible & data.q_bh.lt(.05)
    return data
