# %% [markdown]
# v2 Step 3 - Run the 20 L1 model jobs (5 contrasts x 4 models) defined in model_spec.json.
# Lookup runs ONCE per unique (group, phenotype); feature matrix cached to .npy. Each run uses
# its OWN covariate string + engine (all L1 here). Plots bon+fdr per run.
#
# Usage:
#   python v2_03_run.py                 -> full run, work = icd_analysis_v2_results/pyphewas/
#   python v2_03_run.py pyphewas_smoke  -> smoke run

# %%
import os
import sys
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import utils as U

PYEXE = sys.executable
SCRIPTS = Path(PYEXE).parent / "Scripts"
WORKER = Path(__file__).resolve().parent / "_model_worker.py"
MAX_PAR = 12


def run_cli(tool, args):
    t0 = time.time()
    r = subprocess.run([PYEXE, str(SCRIPTS / tool)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  !! {tool} FAILED ({time.time()-t0:.0f}s): " + " ".join(r.stderr.splitlines()[-8:]))
        raise RuntimeError(f"{tool} failed")
    print(f"  {tool} ok [{time.time()-t0:.0f}s]")


def main(work_name="pyphewas"):
    work = U.V2_RESULTS / work_name
    wp = str(work)
    spec = json.loads((work / "model_spec.json").read_text())
    runs = spec["runs"]

    # ---- 1) lookups (unique group+pheno) + cache matrix to .npy ----
    pairs = {(r["group"], r["pheno"]) for r in runs}
    fm_npy = {}
    print(f"== Lookups ({len(pairs)}) ==")
    for group, pheno in sorted(pairs):
        fm = "fm_" + group
        print(f"- {group} / {pheno}")
        run_cli("pyPhewasLookup", ["--path", wp, "--phenotype", pheno, "--group", group,
                                   "--reg_type", "log", "--outfile", fm])
        arr = pd.read_csv(work / ("agg_measures_" + fm), skiprows=1, header=None).values.astype(float)
        npy = work / (Path(fm).stem + ".npy")
        np.save(npy, arr)
        fm_npy[(group, pheno)] = npy
        print(f"  matrix {arr.shape} -> {npy.name}")

    # ---- 2) model jobs (parallel) ----
    print(f"\n== Models ({len(runs)} jobs, up to {MAX_PAR} parallel) ==")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")

    def launch(r):
        out = f"regressions_{r['name']}.csv"
        cmd = [PYEXE, str(WORKER), "--path", wp, "--group", r["group"],
               "--aggnpy", str(fm_npy[(r["group"], r["pheno"])]),
               "--target", r["target"], "--covariates", r["covariates"],
               "--engine", r["engine"], "--outfile", out]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return r, res, time.time() - t0

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_PAR) as ex:
        futs = [ex.submit(launch, r) for r in runs]
        for f in as_completed(futs):
            r, res, dt = f.result()
            if res.returncode != 0:
                failed.append(r["name"])
                print(f"  !! {r['name']} FAILED [{dt:.0f}s]: " + " ".join(res.stderr.splitlines()[-3:]))
            else:
                print(f"  ok {r['name']} [{dt:.0f}s]: {res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ''}")
    if failed:
        print("FAILED:", failed)

    # ---- 3) plots (bon + fdr) ----
    print("\n== Plots ==")
    for r in runs:
        stat = f"regressions_{r['name']}.csv"
        if not (work / stat).exists():
            continue
        for thr in ("bon", "fdr"):
            try:
                run_cli("pyPhewasPlot", ["--path", wp, "--statfile", stat, "--thresh_type", thr,
                                         "--outfile", f"{r['name']}_{thr}.png"])
            except RuntimeError:
                print(f"  (plot {r['name']} {thr} failed -- continuing)")

    if work_name == "pyphewas":
        U.V2_PLOTS.mkdir(parents=True, exist_ok=True)
        for suf in ("_Manhattan.png", "_EffectSize.png", "_Volcano.png"):
            for png in work.glob("*" + suf):
                shutil.move(str(png), str(U.V2_PLOTS / png.name))
        print(f"\nMoved plots -> {U.V2_PLOTS}")
        U.log_milestone("v2 Step 3 complete: 20 L1 model runs.")
    print("Done." + (" (with failures)" if failed else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pyphewas")
