"""Add adjusted HRs and log-rank tests without overwriting original figures."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

from _age_sex_inference import infer, adjust

spec = importlib.util.spec_from_file_location("age_sex_curves", Path(__file__).with_name("34_age_sex_curves.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
REFERENCE = "18-<40y | Female"
GROUPS = [f"{age} | {sex}" for age in C.AGES for sex in C.SEXES]


def analyze(out):
    import incwas_utils as U
    import _cox_engine as E
    followup, blocks = E.prep_followup()
    followup = C.assign_strata(followup)
    previous, exclusion = E.prev_index(), E.exclude_map()
    events = pd.read_parquet(U.EVENTS, columns=["id", "phecode", "second_post_date"])
    meta = pd.read_csv(C.OUT / "selected_phecodes.csv", dtype={"phecode": str})
    events["phecode"] = events.phecode.astype(str)
    grouped = dict(list(events[events.phecode.isin(meta.phecode)].groupby("phecode", sort=False)))
    summaries = pd.read_csv(C.OUT / "incidence_summary.csv", dtype={"phecode": str})
    lr_rows, hr_rows = [], []
    for index, item in enumerate(meta.itertuples(index=False), 1):
        data = E.build_survival(item.phecode, grouped[item.phecode], followup, previous, exclusion)
        assert len(data) == item.n_atrisk and data.event.sum() == item.n_events
        data = data[data.age_group.notna() & data.sex.notna()].copy()
        data["etype"] = np.where(data.event == 1, 1, np.where(data.is_death, 2, 0))
        data["severity"] = data.severity.replace({"None": "No OSA"})
        data["demographic_group"] = data.age_group + " | " + data.sex
        base = {"phecode": item.phecode, "phenotype": item.phenotype, "osa_control": item.osa_control}
        for age in C.AGES:
            for sex in C.SEXES:
                sub = data[(data.age_group == age) & (data.sex == sex)]
                labels = {**base, "layout": "severity_panels", "age_group": age, "sex": sex}
                lr, hrs = infer(sub, "severity", C.SEVERITIES, "No OSA",
                                [c for c in blocks["models"]["m4"] if c != "sex_M"], labels)
                lr_rows.append(lr)
                hr_rows.extend(hrs)
        labels = {**base, "layout": "six_strata", "age_group": "All", "sex": "All"}
        covariates = [c for c in blocks["models"]["m4"] if c not in ["age_z", "sex_M"]] + E.SEV_DUMMIES
        lr, hrs = infer(data, "demographic_group", GROUPS, REFERENCE, covariates, labels)
        lr_rows.append(lr)
        hr_rows.extend(hrs)
        # Verify inference and plots use the same eligible curve groups.
        for row in lr_rows[-7:]:
            selected = summaries[(summaries.phecode == item.phecode) & summaries.curve_plotted]
            if row["layout"] == "severity_panels":
                expected = selected[(selected.age_group == row["age_group"]) & (selected.sex == row["sex"]) &
                                    (selected.severity != "All OSA severities")].severity.tolist()
            else:
                selected = selected[selected.severity == "All OSA severities"]
                expected = (selected.age_group + " | " + selected.sex).tolist()
            assert row["groups_tested"] == " | ".join(expected), "Inference/plot group mismatch"
        # Aggregate-only checkpoints permit resuming plotting without clinical inputs.
        adjust(pd.DataFrame(lr_rows)).to_csv(out / "logrank_tests.csv", index=False)
        adjust(pd.DataFrame(hr_rows)).to_csv(out / "hazard_ratios.csv", index=False)
        print(f"Inference completed: {index}/{len(meta)} outcomes", flush=True)


def ptext(value):
    return "n/a" if not np.isfinite(value) else ("<1e-300" if value == 0 else f"{value:.2g}")


def annotations(out):
    lr = pd.read_csv(out / "logrank_tests.csv", dtype={"phecode": str})
    hr = pd.read_csv(out / "hazard_ratios.csv", dtype={"phecode": str})
    result = {}
    for row in lr.itertuples(index=False):
        text = f"LR ({row.n_groups} groups): p={ptext(row.p)}; q={ptext(row.q_bh)}".replace("=<", "<")
        sub = hr[(hr.phecode == row.phecode) & (hr.layout == row.layout) &
                 (hr.age_group == row.age_group) & (hr.sex == row.sex)]
        if row.layout == "six_strata":
            text += "\nAdjusted HR (95% CI)\nReference: 18–<40y | Female = 1.00"
        else:
            text += "\nAdjusted HR (95% CI), reference: No OSA"
        for estimate in sub.itertuples(index=False):
            if estimate.status == "ok":
                value = f"{estimate.hr:.2f} [{estimate.ci_low:.2f}–{estimate.ci_high:.2f}]"
                if estimate.low_epv:
                    value += " †"
            else:
                value = "n/a (" + estimate.status.replace("_", " ") + ")"
            text += f"\n{estimate.group}: {value}"
        if sub.low_epv.any():
            text += "\n† <10 events/parameter; exploratory"
        if row.osa_control:
            text += "\nCircular control: excluded from FDR"
        result[(row.phecode, row.age_group, row.sex)] = text
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    if out is None:
        out = C.OUT / "inference_20261017"
        suffix = ord("b")
        while out.exists():
            out = C.OUT / f"inference_20261017{chr(suffix)}"
            suffix += 1
    out.mkdir(parents=True, exist_ok=args.plot_only)
    if not args.plot_only:
        analyze(out)
    C.draw_plots(annotations=annotations(out), figure_out=out)
    print(f"Saved revision: {out}", flush=True)


if __name__ == "__main__":
    main()
