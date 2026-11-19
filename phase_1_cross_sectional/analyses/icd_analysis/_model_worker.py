"""One pyPheWAS mass-regression run (one target x one engine). Invoked as a subprocess
by 06_run_pyphewas.py so the 12 jobs can run in parallel. Writes a regressions CSV in
the exact pyPhewasModel format (metadata first line) so pyPhewasPlot + 07 read it unchanged.

engine: 'mle' -> plain maximum-likelihood logit (unshrunk OR + classical CI)
        'l1'  -> pyPheWAS default L1-regularized logit (shrunk OR)
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import statsmodels.formula.api as smf
import pyPheWAS.pyPhewasCorev2 as core


def fit_mle(model_str, model_type, model_data, phe_thresh=5, phe_key="phe"):
    if not np.where(model_data[phe_key] > 0)[0].shape[0] > phe_thresh:
        return (None, f"Less than {phe_thresh} records with phecode")
    try:
        with warnings.catch_warnings(record=True) as cx:
            if model_type == "linear":
                model = smf.glm(model_str, model_data).fit()
            else:
                model = smf.logit(model_str, model_data).fit(disp=0, maxiter=100)
            note = "".join("[" + str(w.message).replace("\n", " ") + "]" for w in cx) if cx else ""
    except Exception as e:
        return (None, f"ERROR computing regression: {str(e).strip()}")
    return (model, note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--aggnpy", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--engine", required=True, choices=["mle", "l1"])
    ap.add_argument("--outfile", required=True)
    a = ap.parse_args()
    path = Path(a.path)

    agg = np.load(a.aggnpy)
    genotypes = core.get_group_file(path, a.group)
    assert agg.shape[0] == genotypes.shape[0], \
        f"matrix rows {agg.shape[0]} != group rows {genotypes.shape[0]}"
    fm = np.array([agg, np.zeros_like(agg), np.zeros_like(agg)])

    if a.engine == "mle":
        core.fit_pheno_model = fit_mle  # else keep the package default (L1)

    t0 = time.time()
    reg, model_str = core.run_phewas(fm, genotypes, "ICD", 0,
                                     covariates=a.covariates, target=a.target, phe_thresh=5)
    header = ",".join(["model_equation", model_str, "group", a.group, "feature_matrix",
                       Path(a.aggnpy).name, "reg_type", "log", "code_type", "ICD"]) + "\n"
    with open(path / a.outfile, "w") as f:
        f.write(header)
        reg.to_csv(f, index=False)
    print(f"[{a.engine}] {a.target} -> {a.outfile}: {reg['p-val'].notna().sum()} fits in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
