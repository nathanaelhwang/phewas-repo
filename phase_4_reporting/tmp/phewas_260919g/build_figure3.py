"""Eight event-time AJ plots from the existing aggregate curve exports only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEVERITY_SOURCE = ROOT / "results/incwas_results/survival_curves/curve_severity.csv"
PAP_SOURCE = ROOT / "results/incwas_results/survival_curves_landmark/curve_adherence.csv"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
    "path.simplify": False,
})
curve = pd.read_csv(SEVERITY_SOURCE, dtype={"phecode": str}, float_precision="round_trip")
adherence = pd.read_csv(PAP_SOURCE, dtype={"phecode": str}, float_precision="round_trip")
specs = [("401.1", "Essential hypertension"), ("250.2", "Type 2 diabetes"),
         ("428.2", "Heart failure NOS"), ("427.2", "Atrial fibrillation/flutter")]
severity_groups = ["No OSA", "Mild", "Moderate", "Severe"]
severity_colors = ["#4D4D4D", "#E69F00", "#D55E00", "#B2182B"]
pap_groups = ["4+ hr/night", "2-4 hr/night", "0-2 hr/night", "never-started", "unknown"]
pap_colors = ["#2166AC", "#67A9CF", "#EF8A62", "#B2182B", "#888888"]
fig = plt.figure(figsize=(13, 18.4))
fig.text(.04, .987, "Figure 3  Cumulative incidence by OSA severity and PAP use",
         fontsize=16, fontweight="bold", va="top")
fig.text(.04, .954, "A  Incidence stratified by OSA severity",
         fontsize=13, fontweight="bold", va="top")
fig.text(.04, .419, "B  Incidence stratified by PAP use",
         fontsize=13, fontweight="bold", va="top")
fig.text(.04, .402, "Among patients with OSA; follow-up begins at the 180-day landmark",
         fontsize=9, va="top")
audit = []
missing = []


def draw_line(ax, data, group, color, panel, code, linestyle="-"):
    selected = data[(data.group == group) & (data.time_years <= 3)].sort_values("time_years")
    if selected.empty:
        missing.append({"panel": panel, "phecode": code, "group": group})
        return None
    assert selected.time_years.iloc[0] == 0 and selected.time_years.iloc[-1] == 3
    assert (np.diff(selected.time_years) > 0).all()
    assert (np.diff(selected.cif_pct) >= 0).all()
    n0 = int(selected.n_at_risk.iloc[0])
    line, = ax.step(selected.time_years, selected.cif_pct, where="post", lw=1.5,
                    color=color, linestyle=linestyle, label=f"{group} (n={n0:,})")
    np.testing.assert_array_equal(line.get_xdata(), selected.time_years)
    np.testing.assert_array_equal(line.get_ydata(), selected.cif_pct)
    audit.append({"panel": panel, "phecode": code, "group": group, "points": len(selected),
                  "n_at_risk_0": n0, "cif_3year_pct": float(selected.cif_pct.iloc[-1]),
                  "source_coordinates_preserved_exactly": True})
    return selected


def finish_axes(ax, title, xlabel):
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    ax.set(xlim=(0, 3), ylim=(0, None), xlabel=xlabel, ylabel="Cumulative incidence (%)")
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(color="#E6E6E6", linewidth=.5)
    ax.legend(frameon=False, loc="upper left", borderpad=.1, labelspacing=.4)


for i, (code, title) in enumerate(specs):
    panel = f"A{i+1}"
    ax = fig.add_axes([.07 + (i % 2) * .49, .765 if i < 2 else .512, .415, .16])
    data = curve[curve.phecode == code]
    risks = []
    for group, color in zip(severity_groups, severity_colors):
        selected = draw_line(ax, data, group, color, panel, code)
        assert selected is not None, f"Missing required severity group: {code}, {group}"
        selected = selected.set_index("time_years")
        risks.append([f"{int(selected.loc[year, 'n_at_risk']):,}" for year in range(4)])
    finish_axes(ax, f"{panel}  {title}", "Years since index sleep study")
    table = ax.table(cellText=risks, rowLabels=severity_groups,
                     colLabels=["0 years", "1 year", "2 years", "3 years"],
                     cellLoc="center", rowLoc="right", bbox=[.16, -.44, .84, .28])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for cell in table.get_celld().values():
        cell.set_linewidth(0)
        cell.set_facecolor("white")

for i, (code, title) in enumerate(specs):
    panel = f"B{i+1}"
    ax = fig.add_axes([.07 + (i % 2) * .49, .233 if i < 2 else .032, .415, .145])
    data = adherence[(adherence.phecode == code) & (adherence.window_days == 180)]
    for group, color in zip(pap_groups, pap_colors):
        draw_line(ax, data, group, color, panel, code, ":" if group == "unknown" else "-")
    finish_axes(ax, f"{panel}  {title}", "Years after the 180-day landmark")
    omitted = [entry["group"] for entry in missing if entry["panel"] == panel]
    if omitted:
        ax.text(.02, .02, "Not available in export: " + ", ".join(omitted),
                transform=ax.transAxes, fontsize=8, va="bottom", wrap=True)

path = OUT / "figure3_panel.png"
fig.savefig(path, dpi=300, facecolor="white")
fig.savefig(path.with_suffix(".pdf"), facecolor="white")
plt.close(fig)
caption = (
    "Figure 3. Aalen–Johansen cumulative incidence with death as a competing event. "
    "A, incidence stratified by OSA severity; A1–A4 show essential hypertension (401.1), "
    "type 2 diabetes (250.2), heart failure NOS (428.2), and atrial fibrillation/flutter (427.2), "
    "respectively. Numbers at risk at 0, 1, 2, and 3 years are shown below each severity plot. "
    "B, incidence for the same outcomes (B1–B4, respectively) after the 180-day landmark among "
    "patients with OSA, stratified by PAP-use category; unknown adherence is dotted. "
    "The PAP legends show numbers at risk at the landmark. Curves retain every incident-event "
    "step over the displayed three-year horizon and are crude, descriptive estimates. "
    "For essential hypertension, the fully adjusted PAP-use comparisons were FDR-null."
)
if missing:
    caption += " PAP strata absent from the aggregate export are identified within the relevant plots and are not reconstructed."
record = {"number": 3, "file": str(path), "width_inches": 13, "height_inches": 18.4,
          "page_width_inches": 14, "page_height_inches": 20.4, "caption": caption}
(OUT / "survival_manifest.json").write_text(json.dumps([record], ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "survival_audit.json").write_text(json.dumps({
    "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (SEVERITY_SOURCE, PAP_SOURCE)],
    "horizon_years": 3, "pap_landmark_days": 180, "curves": audit,
    "missing_strata": missing, "curve_count": len(audit),
    "source_coordinates_preserved_exactly": all(row["source_coordinates_preserved_exactly"] for row in audit),
    "sampling_or_interpolation": False,
}, indent=2), encoding="utf-8")
print(f"Figure 3 generated: {len(audit)} curves, {sum(row['points'] for row in audit):,} exact source coordinates; {len(missing)} missing strata.")
