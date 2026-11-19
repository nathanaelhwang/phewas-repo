"""Age/sex-stratified PheDAS incidence curves; aggregate outputs only.

Uses the original 40 selected outcomes, harmonized washout, and seeded AJ fitter.
Patient-level derived inputs stay in local memory; no raw sources are accessed.
Run with the myenv2 conda Python. --plot-only redraws saved aggregate curves.
"""
import argparse
import os
from pathlib import Path
import warnings

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _event_curves import event_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "incwas_results" / "age-sex-stratified"
AGES = ["18-<40y", "40-60y", ">60y"]
SEXES = ["Male", "Female"]
SEVERITIES = ["No OSA", "Mild", "Moderate", "Severe"]
SEV_COLORS = ["#4d4d4d", "#f4a300", "#e8601c", "#b2182b"]
AGE_COLORS = ["#0072b2", "#009e73", "#cc79a7"]
HORIZON = 6
KEYS = ["phecode", "phenotype", "osa_control", "age_group", "sex", "severity"]


def assign_strata(frame):
    """Disjoint age bands at index: [18,40), [40,60], (60,infinity)."""
    frame = frame.copy()
    age = pd.to_numeric(frame["age"], errors="coerce")
    frame["age_group"] = pd.Series(index=frame.index, dtype=object)
    for label, mask in zip(AGES, [age.ge(18) & age.lt(40), age.ge(40) & age.le(60), age.gt(60)]):
        frame.loc[mask, "age_group"] = label
    frame["sex"] = frame["sex_M"].map({1: "Male", 0: "Female"})
    return frame


def summarize_curve(data, metadata):
    t = data["time"].to_numpy(float)
    ev = data["etype"].to_numpy(int)
    in_window = t <= HORIZON
    n_events = int(((ev == 1) & in_window).sum())
    eligible = len(t) >= 20 and n_events >= 5
    support = min(float(t.max()), HORIZON) if len(t) else 0.
    risk = pd.DataFrame({
        **metadata, "time_years": np.arange(HORIZON + 1),
        "n_at_risk": [int((t >= year).sum()) for year in range(HORIZON + 1)],
        "n_events_cum": [int(((t <= year) & (ev == 1)).sum()) for year in range(HORIZON + 1)],
    })
    fitted = None
    if eligible:
        fitted = event_curve(t, ev, HORIZON)
        # Do not draw an unsupported flat tail after the last observation.
        pos = np.searchsorted(fitted.time_years, support, side="right") - 1
        endpoint = fitted.iloc[[pos]].assign(time_years=support,
                    n_at_risk=int((t >= support).sum()),
                    n_events_cum=int(((t <= support) & (ev == 1)).sum()))
        fitted = pd.concat([fitted[fitted.time_years < support], endpoint], ignore_index=True)
        assert np.all(np.diff(fitted.time_years) > 0)
        assert np.all(np.diff(fitted.cif_pct) >= -1e-10)
        assert fitted.cif_pct.between(0, 100).all()
        fitted = fitted.assign(**metadata)[KEYS + list(fitted.columns)]
    person_years = float(np.minimum(t, HORIZON).sum())
    summary = {**metadata, "n_atrisk": len(t), "n_events_total": int((ev == 1).sum()),
               "n_events_6y": n_events, "n_deaths_6y": int(((ev == 2) & in_window).sum()),
               "person_years_6y": person_years,
               "rate_per_1000py_6y": 1000*n_events/person_years if person_years else np.nan,
               "support_years": support, "curve_plotted": eligible,
               "omission_reason": "" if eligible else "n_atrisk<20 or incident_events_by_6y<5"}
    for year in [1, 3, 5, 6]:
        value = np.nan
        if fitted is not None and year <= support:
            value = float(fitted.cif_pct.iloc[np.searchsorted(fitted.time_years, year, side="right")-1])
        summary[f"cif_{year}y_pct"] = value
    return fitted, risk, summary


def analyze():
    import incwas_utils as U
    import _cox_engine as E

    meta = pd.read_csv(U.RESULTS / "survival_curves" / "fdr_phecodes.csv", dtype={"phecode": str})
    foll, _ = E.prep_followup()
    foll = assign_strata(foll)
    assert not foll.id.duplicated().any(), "Follow-up spine must be unique"
    assert foll.severity.isin(U.SEVERITIES).all(), "Unrecognized OSA severity"
    prev, exclusion = E.prev_index(), E.exclude_map()
    ev = pd.read_parquet(U.EVENTS, columns=["id", "phecode", "second_post_date"])
    ev["phecode"] = ev.phecode.astype(str)
    ev = ev[ev.phecode.isin(meta.phecode)]
    grouped = dict(list(ev.groupby("phecode", sort=False)))
    cohort = foll.groupby(["age_group", "sex", "severity"], dropna=False).size().rename("n").reset_index()
    cohort.to_csv(OUT / "cohort_counts.csv", index=False)
    curves, risks, summaries, checks = [], [], [], []
    for item in meta.itertuples(index=False):
        d = E.build_survival(item.phecode, grouped[item.phecode], foll, prev, exclusion)
        d["etype"] = np.where(d.event == 1, 1, np.where(d.is_death, 2, 0))
        assert len(d) == item.n_atrisk, "At-risk population differs from original export"
        assert int(d.event.sum()) == item.n_events, "Event count differs from original export"
        valid = d.age_group.notna() & d.sex.notna()
        checks.append({"phecode": item.phecode, "n_original_atrisk": len(d),
                       "n_stratified_atrisk": int(valid.sum()), "n_excluded_age_or_sex": int((~valid).sum()),
                       "n_original_events": int(d.event.sum()), "original_counts_match": True})
        d = d[valid].copy()
        d["severity"] = d.severity.replace({"None": "No OSA"})
        for age in AGES:
            for sex in SEXES:
                stratum = d[(d.age_group == age) & (d.sex == sex)]
                for severity in ["All OSA severities"] + SEVERITIES:
                    sub = stratum if severity == "All OSA severities" else stratum[stratum.severity == severity]
                    labels = {"phecode": item.phecode, "phenotype": item.phenotype,
                              "osa_control": item.osa_control, "age_group": age,
                              "sex": sex, "severity": severity}
                    curve, risk, summary = summarize_curve(sub, labels)
                    if curve is not None:
                        curves.append(curve)
                    risks.append(risk)
                    summaries.append(summary)
        print(f"Completed outcome {len(checks)}/{len(meta)}", flush=True)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "curve_coordinates.csv", index=False)
    pd.concat(risks, ignore_index=True).to_csv(OUT / "annual_risk_tables.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "incidence_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "validation_counts.csv", index=False)
    meta.to_csv(OUT / "selected_phecodes.csv", index=False)


def risk_table(ax, data, labels, colors):
    cells = [data[label] for label in labels]
    table = ax.table(cellText=cells, rowLabels=labels, colLabels=list(range(HORIZON+1)),
                     cellLoc="center", rowLoc="right", bbox=[0, -.58, 1, .32])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row > 0 and col == -1:
            cell.get_text().set_color(colors[row-1])


def draw_plots(annotations=None, figure_out=None):
    figure_out = OUT if figure_out is None else Path(figure_out)
    curves = pd.read_csv(OUT / "curve_coordinates.csv", dtype={"phecode": str})
    risks = pd.read_csv(OUT / "annual_risk_tables.csv", dtype={"phecode": str})
    summary = pd.read_csv(OUT / "incidence_summary.csv", dtype={"phecode": str})
    meta = pd.read_csv(OUT / "selected_phecodes.csv", dtype={"phecode": str})
    for folder in ["severity_panels", "six_strata"]:
        (figure_out / folder).mkdir(exist_ok=True)
    with PdfPages(figure_out / "severity_panels_all_outcomes.pdf") as panel_pdf, \
         PdfPages(figure_out / "six_strata_all_outcomes.pdf") as strata_pdf:
        for item in meta.itertuples(index=False):
            c = curves[curves.phecode == item.phecode]
            r = risks[risks.phecode == item.phecode]
            s = summary[summary.phecode == item.phecode]
            title = f"{item.phenotype} [{item.phecode}]"
            if item.osa_control:
                title += " | circular OSA control"
            fig, axes = plt.subplots(3, 2, figsize=(15, 18) if annotations else (13, 13), sharex=True, sharey=True)
            severity_max = c.loc[c.severity != "All OSA severities", "cif_pct"].max()
            ymax = max(1., float(severity_max)*1.08) if pd.notna(severity_max) else 1.
            for i, age in enumerate(AGES):
                for j, sex in enumerate(SEXES):
                    ax = axes[i, j]
                    counts, omitted = {}, []
                    for severity, color in zip(SEVERITIES, SEV_COLORS):
                        mask = (c.age_group == age) & (c.sex == sex) & (c.severity == severity)
                        line = c[mask]
                        if len(line):
                            ax.step(line.time_years, line.cif_pct, where="post", color=color, lw=1.7)
                        else:
                            omitted.append(severity)
                        annual = r[(r.age_group == age) & (r.sex == sex) & (r.severity == severity)]
                        counts[severity] = annual.sort_values("time_years").n_at_risk.tolist()
                    risk_table(ax, counts, SEVERITIES, SEV_COLORS)
                    ax.set_title(f"{age} | {sex}", fontsize=11)
                    if annotations:
                        ax.set_title(f"{age} | {sex}\n" + annotations[(item.phecode, age, sex)], fontsize=9)
                    ax.set_xlim(0, HORIZON)
                    ax.set_ylim(0, ymax)
                    ax.set_xticks(range(HORIZON+1))
                    ax.tick_params(labelbottom=True)
                    ax.grid(alpha=.2)
                    ax.set_xlabel("Years since index sleep study", fontsize=9)
                    if j == 0:
                        ax.set_ylabel("Cumulative incidence (%)")
                    if omitted:
                        ax.text(.02, .97, "Insufficient events: " + ", ".join(omitted),
                                transform=ax.transAxes, va="top", fontsize=7)
            fig.suptitle(title + "\nAalen–Johansen incidence by OSA severity within age/sex strata", fontsize=14, y=.99)
            fig.legend([Line2D([0], [0], color=color, lw=2) for color in SEV_COLORS],
                       SEVERITIES, loc="upper center", ncol=4, bbox_to_anchor=(.5, .945), frameon=False)
            note = "Unadjusted curves; adjusted cause-specific HR (95% CI). LR: unadjusted log-rank; death censored.\n" if annotations else "Unadjusted descriptive curves; death competing. Tables: number at risk at years 0–6.\n"
            fig.text(.5, .015, note +
                     "Age at index: 18–<40, 40–60 inclusive, >60. Omitted curves: n<20 or <5 incident events by year 6.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(top=.85 if annotations else .88, bottom=.135,
                                hspace=1.7 if annotations else 1.05, wspace=.22)
            fig.savefig(figure_out / "severity_panels" / f"{item.phecode.replace('.', '_')}_cif.png", dpi=160)
            panel_pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(15, 9) if annotations else (10.5, 8.5))
            counts, labels, colors = {}, [], []
            for age, color in zip(AGES, AGE_COLORS):
                for sex, style in zip(SEXES, ["-", "--"]):
                    label = f"{age} | {sex}"
                    line = c[(c.age_group == age) & (c.sex == sex) & (c.severity == "All OSA severities")]
                    row = s[(s.age_group == age) & (s.sex == sex) & (s.severity == "All OSA severities")].iloc[0]
                    suffix = f" (n={row.n_atrisk:,})" if row.curve_plotted else " (insufficient events)"
                    ax.step(line.time_years, line.cif_pct, where="post", color=color, ls=style, lw=2,
                            label=label + suffix)
                    annual = r[(r.age_group == age) & (r.sex == sex) & (r.severity == "All OSA severities")]
                    counts[label] = annual.sort_values("time_years").n_at_risk.tolist()
                    labels.append(label)
                    colors.append(color)
            risk_table(ax, counts, labels, colors)
            ax.set(xlim=(0, HORIZON), ylim=(0, None), xticks=range(HORIZON+1),
                   xlabel="Years since index sleep study", ylabel="Cumulative incidence (%)")
            ax.set_title(title + "\nAalen–Johansen incidence: six age/sex strata", fontsize=13, pad=12)
            if annotations:
                text = annotations[(item.phecode, "All", "All")]
                ax.set_title(title + "\nSix age/sex strata\n" + text.split("\n")[0], fontsize=11, pad=12)
                fig.text(.69, .83, "\n".join(text.split("\n")[1:]), fontsize=9, va="top", linespacing=1.7)
            ax.legend(fontsize=9, framealpha=.9, loc="upper left")
            ax.grid(alpha=.2)
            note = "All OSA severities pooled; unadjusted curves; adjusted cause-specific HR (95% CI); death competing.\n" if annotations else "All OSA severity groups pooled; unadjusted descriptive curves; death competing.\n"
            fig.text(.5, .02, note +
                     "Age at index: 18–<40, 40–60 inclusive, >60. Table: number at risk at years 0–6.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=.2 if annotations else .24, right=.66 if annotations else .96,
                                top=.85 if annotations else .9, bottom=.38)
            fig.savefig(figure_out / "six_strata" / f"{item.phecode.replace('.', '_')}_cif.png", dpi=160)
            strata_pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved both plot layouts for {len(meta)} outcomes.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message="(?s).*Tied event times were detected.*")
    if not args.plot_only:
        analyze()
    draw_plots()


if __name__ == "__main__":
    main()
