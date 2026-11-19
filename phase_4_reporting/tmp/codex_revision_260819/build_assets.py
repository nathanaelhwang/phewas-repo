from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\natha\Projects\work\phewas")
OUT = Path(
    r"C:\Users\natha\.codex\visualizations\2026\08\19\01a01bfd-c51d-7512-a8e3-a54bed66b63a\revision_260819\assets"
)
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2F6690"
BLUE_LIGHT = "#82A6C4"
RED = "#B33A3A"
ORANGE = "#D9822B"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"
GRID = "#E6E6E6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def fmt_p(value: float, with_prefix: bool = True) -> str:
    if pd.isna(value):
        return ""
    prefix = "P" if with_prefix else ""
    if value < 0.0001:
        return f"{prefix}<0.0001"
    if value < 0.01:
        return f"{prefix}={value:.4f}" if with_prefix else f"{value:.4f}"
    return f"{prefix}={value:.3f}" if with_prefix else f"{value:.3f}"


def format_n_pct(n: int, denom: int) -> str:
    return f"{n:,} ({100 * n / denom:.1f}%)"


def summarize_table1() -> tuple[pd.DataFrame, dict]:
    cohort_path = ROOT / "results/icd_analysis_v2_results/cohort/cohort.parquet"
    cols = ["AHI", "age", "sex_M", "bmi", "RACE_ETH", "smoking_status", "income_gp"]
    cohort = pd.read_parquet(cohort_path, columns=cols)
    cohort["group"] = np.where(cohort["AHI"] >= 5, "OSA", "No OSA")

    masks = {
        "Overall": np.ones(len(cohort), dtype=bool),
        "OSA": cohort["group"].eq("OSA"),
        "No OSA": cohort["group"].eq("No OSA"),
    }
    denoms = {name: int(mask.sum()) for name, mask in masks.items()}

    rows: list[dict] = []

    def add_row(label: str, values: dict[str, str], row_type: str = "data") -> None:
        rows.append({"row_type": row_type, "Characteristic": label, **values})

    add_row("Participants, n", {k: f"{v:,}" for k, v in denoms.items()})
    for col, label in [
        ("age", "Age, years, mean (SD)"),
        ("AHI", "AHI, events/h, mean (SD)"),
        ("bmi", "BMI, kg/m2, mean (SD)"),
    ]:
        vals = {}
        for name, mask in masks.items():
            x = cohort.loc[mask, col].dropna()
            vals[name] = f"{x.mean():.1f} ({x.std(ddof=1):.1f})"
        add_row(label, vals)

    for sex_value, label in [(1, "Male, n (%)"), (0, "Female, n (%)")]:
        vals = {}
        for name, mask in masks.items():
            n = int((cohort.loc[mask, "sex_M"] == sex_value).sum())
            vals[name] = format_n_pct(n, denoms[name])
        add_row(label, vals)

    add_row("Race/ethnicity, n (%)", {k: "" for k in masks}, "section")
    race_order = [
        ("Hispanic", "Hispanic"),
        ("White", "White"),
        ("Asian", "Asian"),
        ("Black", "Black"),
        ("Pacific Islander", "Pacific Islander"),
        ("Native Am Alaskan", "American Indian or Alaska Native"),
        ("Multiple", "Multiple"),
        ("Other", "Other"),
        ("Unknown", "Unknown"),
    ]
    for value, label in race_order:
        vals = {}
        for name, mask in masks.items():
            n = int(cohort.loc[mask, "RACE_ETH"].eq(value).sum())
            vals[name] = format_n_pct(n, denoms[name])
        add_row(f"  {label}", vals)

    add_row("Smoking status, n (%)", {k: "" for k in masks}, "section")
    for value in ["Never", "Former", "Current", "Unknown"]:
        vals = {}
        for name, mask in masks.items():
            n = int(cohort.loc[mask, "smoking_status"].eq(value).sum())
            vals[name] = format_n_pct(n, denoms[name])
        add_row(f"  {value}", vals)

    add_row("Neighborhood income group, n (%)", {k: "" for k in masks}, "section")
    for value in ["High", "Mid", "Low", "Missing"]:
        vals = {}
        for name, mask in masks.items():
            n = int(cohort.loc[mask, "income_gp"].eq(value).sum())
            vals[name] = format_n_pct(n, denoms[name])
        add_row(f"  {value}", vals)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table1.csv", index=False)

    overall = cohort
    metrics = {
        "n": denoms,
        "age_mean": float(overall["age"].mean()),
        "age_sd": float(overall["age"].std(ddof=1)),
        "ahi_mean": float(overall["AHI"].mean()),
        "ahi_sd": float(overall["AHI"].std(ddof=1)),
        "bmi_mean": float(overall["bmi"].mean()),
        "bmi_sd": float(overall["bmi"].std(ddof=1)),
        "male_pct": float(100 * overall["sex_M"].mean()),
        "race_pct": {
            value: float(100 * overall["RACE_ETH"].eq(value).mean()) for value, _ in race_order
        },
    }
    del cohort
    return out, metrics


def prepare_manhattan(df: pd.DataFrame, code_col: str, cat_col: str) -> tuple[pd.DataFrame, list[tuple[float, str]]]:
    d = df.copy()
    d[cat_col] = d[cat_col].fillna("other")
    d["code_num"] = pd.to_numeric(d[code_col].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0], errors="coerce")
    d = d.sort_values([cat_col, "code_num", code_col], kind="stable").reset_index(drop=True)
    x = []
    ticks: list[tuple[float, str]] = []
    cursor = 0
    for category, idx in d.groupby(cat_col, sort=False).groups.items():
        idx = list(idx)
        positions = list(range(cursor, cursor + len(idx)))
        x.extend(positions)
        ticks.append((float(np.mean(positions)), str(category)))
        cursor += len(idx) + 6
    d["x"] = x
    return d, ticks


def draw_manhattan(
    ax: plt.Axes,
    df: pd.DataFrame,
    code_col: str,
    name_col: str,
    cat_col: str,
    p_col: str,
    sig_col: str,
    labels: list[str],
    cap: float,
) -> None:
    d, ticks = prepare_manhattan(df, code_col, cat_col)
    p = pd.to_numeric(d[p_col], errors="coerce").clip(lower=1e-300)
    d["y"] = np.minimum(-np.log10(p), cap)
    d["sig"] = as_bool(d[sig_col])
    cats = list(dict.fromkeys(d[cat_col].tolist()))
    palette = [BLUE, "#6C8EAD"]
    for i, category in enumerate(cats):
        sub = d[d[cat_col] == category]
        ax.scatter(sub["x"], sub["y"], s=8, color=palette[i % 2], alpha=0.60, linewidth=0)
    sig = d[d["sig"]]
    ax.scatter(sig["x"], sig["y"], s=12, color=RED, alpha=0.78, linewidth=0)
    bon = -math.log10(0.05 / max(len(d), 1))
    fdr_rows = d[d["sig"]]
    if not fdr_rows.empty:
        fdr_p = pd.to_numeric(fdr_rows[p_col], errors="coerce").max()
        ax.axhline(-math.log10(max(fdr_p, 1e-300)), color=ORANGE, ls="--", lw=0.9, label="BH-FDR 0.05")
    ax.axhline(bon, color=GRAY, ls=":", lw=0.9, label="Bonferroni 0.05")
    for label in labels:
        hit = d[d[name_col].eq(label)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        p_value = pd.to_numeric(pd.Series([row[p_col]]), errors="coerce").iloc[0]
        display_label = label
        if pd.isna(p_value) or p_value <= 0 or -math.log10(max(float(p_value), 1e-300)) > cap:
            display_label = f"{label} (P<0.0001; capped)"
        ax.annotate(
            display_label,
            (row["x"], row["y"]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=6.5,
            color="#222222",
        )
    ax.set_xticks([x for x, _ in ticks])
    ax.set_xticklabels([label.replace(" diseases", "") for _, label in ticks], rotation=48, ha="right")
    ax.set_ylabel("-log10(P)")
    ax.set_xlabel("Phecodes grouped by category")
    ax.set_ylim(0, cap * 1.05)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="upper right", frameon=False, ncol=2)


def draw_volcano(
    ax: plt.Axes,
    df: pd.DataFrame,
    name_col: str,
    effect_col: str,
    p_col: str,
    sig_col: str,
    labels: list[str],
    cap: float,
    exclude_names: set[str] | None = None,
) -> None:
    d = df.copy()
    if exclude_names:
        d = d[~d[name_col].isin(exclude_names)]
    d["effect_plot"] = np.log2(pd.to_numeric(d[effect_col], errors="coerce"))
    d["y"] = np.minimum(-np.log10(pd.to_numeric(d[p_col], errors="coerce").clip(lower=1e-300)), cap)
    d["sig"] = as_bool(d[sig_col])
    colors = np.where(d["sig"] & (d["effect_plot"] > 0), RED, np.where(d["sig"], BLUE, LIGHT_GRAY))
    ax.scatter(d["effect_plot"], d["y"], s=12, c=colors, alpha=0.72, linewidth=0)
    ax.axvline(0, color=GRAY, lw=0.8)
    fdr_rows = d[d["sig"]]
    if not fdr_rows.empty:
        fdr_p = pd.to_numeric(fdr_rows[p_col], errors="coerce").max()
        ax.axhline(-math.log10(max(fdr_p, 1e-300)), color=ORANGE, ls="--", lw=0.9)
    for i, label in enumerate(labels):
        hit = d[d[name_col].eq(label)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        offset = (4, 4 + (i % 3) * 5)
        ax.annotate(label, (row["effect_plot"], row["y"]), xytext=offset, textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("log2 effect ratio")
    ax.set_ylabel("-log10(P)")
    ax.set_ylim(0, cap * 1.05)
    ax.grid(axis="y", color=GRID, lw=0.5)


def draw_forest(
    ax: plt.Axes,
    df: pd.DataFrame,
    code_col: str,
    name_col: str,
    effect_col: str,
    low_col: str,
    high_col: str,
    sig_col: str,
    codes: list[str],
    ratio_label: str,
) -> None:
    rows = []
    for code in codes:
        hit = df[df[code_col].astype(str).eq(code)]
        if not hit.empty:
            rows.append(hit.iloc[0])
    d = pd.DataFrame(rows)
    y = np.arange(len(d))[::-1]
    effects = pd.to_numeric(d[effect_col], errors="coerce")
    lows = pd.to_numeric(d[low_col], errors="coerce")
    highs = pd.to_numeric(d[high_col], errors="coerce")
    sig = as_bool(d[sig_col])
    for yi, effect, low, high, is_sig in zip(y, effects, lows, highs, sig):
        color = RED if is_sig and effect > 1 else BLUE if is_sig else ORANGE
        ax.errorbar(effect, yi, xerr=[[effect - low], [high - effect]], fmt="o", color=color, ecolor=GRAY, capsize=2, ms=4)
    labels = d[name_col].astype(str).str.replace("Delirium dementia and amnestic and other cognitive disorders", "Broad cognitive-disorder phecode", regex=False)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(1, color="#333333", ls="--", lw=0.8)
    ax.set_xscale("log")
    xmin = max(0.45, float(lows.min()) * 0.90)
    xmax = min(3.0, float(highs.max()) * 1.12)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(f"{ratio_label} (95% CI); log scale")
    ax.grid(axis="x", color=GRID, lw=0.5)
    for yi, effect in zip(y, effects):
        ax.text(xmax / 1.02, yi, f"{effect:.2f}", ha="right", va="center", fontsize=6.5, color="#333333")


def make_phewas_figures() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prev_all = pd.read_csv(
        ROOT / "results/icd_analysis_v2_results/summary/phewas_results_combined.csv",
        low_memory=False,
        dtype={"PheWAS Code": str},
    )
    prev = prev_all[(prev_all["contrast"] == "ahi_ge5") & (prev_all["model"] == "m4_bmi")].copy()
    prev["sig_fdr"] = as_bool(prev["sig_fdr"])
    prev_labels = [
        "Essential hypertension",
        "Elevated blood pressure reading without diagnosis of hypertension",
        "Obesity",
        "Type 2 diabetes",
        "Heart failure NOS",
        "Palpitations",
    ]
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_manhattan(ax_a, prev, "PheWAS Code", "PheWAS Name", "category_string", "p-val", "sig_fdr", prev_labels[:4], 120)
    draw_volcano(ax_b, prev, "PheWAS Name", "OR", "p-val", "sig_fdr", prev_labels, 120)
    draw_forest(
        ax_c,
        prev,
        "PheWAS Code",
        "PheWAS Name",
        "OR",
        "OR_low",
        "OR_high",
        "sig_fdr",
        ["401.1", "402", "278.1", "250.2", "272.1", "428.2", "427.21", "290.1", "292.2", "292.3", "427.9", "285", "563"],
        "Odds ratio",
    )
    ax_a.set_title("A  Manhattan plot")
    ax_b.set_title("B  Volcano plot")
    ax_c.set_title("C  Selected prevalent conditions")
    fig.suptitle("Prevalence disease PheWAS: OSA (AHI ≥5) vs No OSA (AHI <5), m4", fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(OUT / "figure1_prevalence_phewas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    inc_all = pd.read_csv(
        ROOT / "results/incwas_results/summary/cox_results_combined.csv",
        low_memory=False,
        dtype={"phecode": str},
    )
    inc = inc_all[(inc_all["contrast"] == "ahi_ge5") & (inc_all["model"] == "m4")].copy()
    inc["sig_fdr"] = as_bool(inc["sig_fdr"])
    inc_labels = [
        "Essential hypertension",
        "Type 2 diabetes",
        "Impaired fasting glucose",
        "Heart failure NOS",
        "Atrial fibrillation and flutter",
        "Anxiety disorder",
    ]
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_manhattan(ax_a, inc, "phecode", "phenotype", "category", "p", "sig_fdr", ["Sleep apnea", *inc_labels[:5]], 55)
    draw_volcano(
        ax_b,
        inc,
        "phenotype",
        "effect",
        "p",
        "sig_fdr",
        inc_labels,
        45,
        exclude_names={"Sleep apnea"},
    )
    draw_forest(
        ax_c,
        inc,
        "phecode",
        "phenotype",
        "effect",
        "ci_low",
        "ci_high",
        "sig_fdr",
        ["250.41", "401.1", "250.2", "272.1", "278.1", "428.2", "427.2", "290.1", "300.1", "785"],
        "Hazard ratio",
    )
    ax_a.set_title("A  Manhattan plot")
    ax_b.set_title("B  Volcano plot (sleep-apnea control omitted)")
    ax_c.set_title("C  Selected incident conditions")
    fig.suptitle("Incidence disease PheWAS: OSA (AHI ≥5) vs No OSA (AHI <5), m4", fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(OUT / "figure2_incidence_phewas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    model_counts = {}
    for model in ["m1_crude", "m2_demo", "m3_smk_ses_win", "m4_bmi"]:
        sub = prev_all[(prev_all["contrast"] == "ahi_ge5") & (prev_all["model"] == model)]
        model_counts[model] = int(as_bool(sub["sig_fdr"]).sum())
    pos = prev[(prev["sig_fdr"]) & (pd.to_numeric(prev["OR"], errors="coerce") > 1)]
    cardiometabolic = pos["category_string"].isin(["circulatory system", "endocrine/metabolic"])

    metrics = {
        "prevalence_model_fdr_counts": model_counts,
        "prevalence_positive_fdr": int(len(pos)),
        "prevalence_positive_cardiometabolic": int(cardiometabolic.sum()),
    }
    return prev, inc, metrics


def make_table2() -> pd.DataFrame:
    rows: list[dict] = []

    def append_row(panel: str, domain: str, window: str, outcome: str, row: pd.Series, note: str = "") -> None:
        rows.append(
            {
                "panel": panel,
                "domain": domain,
                "window": window,
                "outcome": outcome,
                "n": int(float(row["n"])),
                "measure": str(row["effect_type"]),
                "effect": float(row["effect"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "p": float(row["p"]),
                "sig_fdr": bool(as_bool(pd.Series([row["sig_fdr"]])).iloc[0]),
                "note": note,
            }
        )

    # A. LabWAS
    lab_labels = {"2571-8": "Triglycerides, median value", "2085-9": "HDL cholesterol, median value", "3016-3": "Thyrotropin order rate"}
    for window in ["1yr", "5yr"]:
        d = pd.read_csv(ROOT / f"results/labwas_results/analysis/results_{window}.csv")
        d = d[(d["contrast"] == "ahi_ge5") & (d["model"] == "m4")]
        for code, target in [("2571-8", "median"), ("3016-3", "order_rate")]:
            hit = d[(d["loinc_code"].astype(str) == code) & (d["target"] == target)]
            append_row("A", "LabWAS", window, lab_labels[code], hit.iloc[0])
        if window == "5yr":
            hit = d[(d["loinc_code"].astype(str) == "2085-9") & (d["target"] == "median")]
            append_row("A", "LabWAS", window, lab_labels["2085-9"], hit.iloc[0])
    prop = pd.read_csv(ROOT / "results/labwas_results/analysis/results_propensity_5yr.csv")
    prop = prop[(prop["contrast"] == "ahi_ge5") & (prop["model"] == "m4") & (prop["loinc_code"].astype(str) == "2132-9")]
    append_row("A", "LabWAS", "5yr", "Vitamin B12, any-order propensity", prop.iloc[0], "utilization outcome")

    # B. MedWAS
    gpi = pd.read_csv(ROOT / "results/medwas_results/crosswalk/gpi_crosswalk.csv", dtype={"gpi4": str})
    gpi["gpi4"] = gpi["gpi4"].str.zfill(4)
    gpi_names = dict(zip(gpi["gpi4"], gpi["drug_class_name"]))
    for window in ["1yr", "5yr"]:
        d = pd.read_csv(ROOT / f"results/medwas_results/analysis/results_{window}.csv", dtype={"code": str})
        d["code"] = d["code"].str.zfill(4)
        d = d[(d["contrast"] == "ahi_ge5") & (d["model"] == "m4")]
        for code in (["3699"] if window == "1yr" else ["3699", "5812"]):
            hit = d[d["code"] == code]
            append_row("B", "MedWAS", window, f"Any fill: {gpi_names.get(code, code)}", hit.iloc[0])

    # C. ProcWAS
    for window in ["1yr", "5yr"]:
        d = pd.read_csv(ROOT / f"results/procwas_results/analysis/results_{window}.csv")
        d = d[(d["contrast"] == "ahi_ge5") & (d["model"] == "m4")]
        hit = d[d["feature"].str.contains(r"> 48 Hrs Up to 21 Days", case=False, na=False)]
        append_row("C", "ProcWAS", window, "Extended ambulatory ECG monitoring (>48 h to 21 d)", hit.iloc[0])
        if window == "5yr":
            hit = d[d["feature"].eq("Spirometry, Pre and Post Bronchodilator")]
            append_row("C", "ProcWAS", window, "Pre/post-bronchodilator spirometry", hit.iloc[0])

    # D. CRMWAS
    d = pd.read_csv(ROOT / "results/procwas_results/measwas/analysis/results_5yr.csv")
    d = d[(d["contrast"] == "ahi_ge5") & (d["model"] == "m4")]
    crm = [
        ("LV_mass_ASE_g", "Echocardiography: left-ventricular mass"),
        ("e_prime_septal_cms", "Echocardiography: septal e' velocity"),
        ("E_over_e_prime_septal", "Echocardiography: septal E/e'"),
        ("FEV1_FVC_pre", "Spirometry: pre-bronchodilator FEV1/FVC"),
    ]
    for feature, label in crm:
        append_row("D", "CRMWAS", "5yr", label, d[d["feature"].eq(feature)].iloc[0])

    # E. UtilWAS
    for window in ["1yr", "5yr"]:
        s = pd.read_csv(ROOT / f"results/utilwas_results/analysis/results_specialty_{window}.csv")
        s = s[(s["contrast"] == "ahi_ge5") & (s["model"] == "m4") & (s["family"] == "physician_level") & (s["feature"] == "Neurology")]
        append_row("E", "UtilWAS", window, "Physician-level neurology encounter rate", s.iloc[0])
        d = pd.read_csv(ROOT / f"results/utilwas_results/analysis/results_phecode_{window}.csv")
        d = d[(d["contrast"] == "ahi_ge5") & (d["model"] == "m4")]
        hit = d[(d["combo"] == "physician_primary") & (d["feature"].astype(str) == "401.1") & (d["part"] == "presence")]
        append_row("E", "UtilWAS", window, "Primary-care hypertension presence", hit.iloc[0], "hurdle presence")
        if window == "5yr":
            hit = d[(d["combo"] == "ED_secondary") & (d["feature"].astype(str) == "401.1") & (d["part"] == "count_among_present")]
            append_row("E", "UtilWAS", window, "ED-secondary hypertension count among carriers", hit.iloc[0], "conditional count")

    # F. BehWAS
    b = pd.read_csv(ROOT / "results/behwas_results/analysis/results_all.csv")
    b = b[(b["contrast"] == "ahi_ge5") & (b["model"] == "m4")]
    beh = [
        ("1yr", "adherent", "Medication adherence (PDC >=0.80)"),
        ("5yr", "adherent", "Medication adherence (PDC >=0.80)"),
        ("1yr", "flu_received", "Influenza vaccination receipt"),
        ("5yr", "flu_received", "Influenza vaccination receipt"),
        ("lifetime", "any_noshow", "Any missed appointment"),
        ("lifetime", "any_off_work", "Any off-work order"),
    ]
    for window, outcome, label in beh:
        append_row("F", "BehWAS", window, label, b[(b["window"] == window) & (b["outcome"] == outcome)].iloc[0], "descriptive" if window == "lifetime" else "")

    # G. QWAS
    q = pd.read_csv(ROOT / "results/qwas_results/analysis/results.csv")
    q = q[(q["contrast"] == "ahi_ge5") & (q["model"] == "m4")]
    qrows = [
        ("STOP1", "binary", "STOP snoring item", "ascertainment-sensitive"),
        ("ESS_TOTAL", "binary", "ESS >=11", ""),
        ("ESS_TOTAL", "continuous", "ESS total, continuous", ""),
        ("FOSQ_MEAN", "continuous", "FOSQ-10 mean", ""),
        ("PHQ2_TOTAL", "binary", "PHQ-2 >=3", "FDR-null"),
    ]
    for feature, form, label, note in qrows:
        append_row("G", "QWAS", "pre-index", label, q[(q["feature"] == feature) & (q["form"] == form)].iloc[0], note)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table2.csv", index=False)
    return out


def make_table3(inc: pd.DataFrame) -> pd.DataFrame:
    fdr = inc[inc["sig_fdr"]].copy()
    fdr["status"] = np.where(as_bool(fdr["sig_bon"]), "FDR + Bonf.", "FDR")
    af = inc[inc["phecode"].astype(str).eq("427.2")].copy()
    if not af.empty and not af["phecode"].astype(str).isin(fdr["phecode"].astype(str)).any():
        af["status"] = "Nominal; not FDR"
        fdr = pd.concat([fdr, af], ignore_index=True)
    fdr["sort_control"] = np.where(fdr["phenotype"].eq("Sleep apnea"), 0, np.where(fdr["status"].eq("Nominal; not FDR"), 2, 1))
    fdr = fdr.sort_values(["sort_control", "p", "phenotype"], kind="stable")
    cols = ["phecode", "phenotype", "category", "n_atrisk", "n_events", "effect", "ci_low", "ci_high", "p", "status"]
    fdr[cols].to_csv(OUT / "table3.csv", index=False)
    return fdr[cols]


def make_survival_figures() -> dict:
    curve = pd.read_csv(ROOT / "results/incwas_results/survival_curves/curve_severity.csv", dtype={"phecode": str})
    codes = ["401.1", "250.2", "428.2", "427.2"]
    titles = {
        "401.1": "Essential hypertension",
        "250.2": "Type 2 diabetes",
        "428.2": "Heart failure NOS",
        "427.2": "Atrial fibrillation/flutter",
    }
    group_order = ["No OSA", "Mild", "Moderate", "Severe"]
    colors = {"No OSA": "#4D4D4D", "Mild": "#E69F00", "Moderate": "#D55E00", "Severe": "#B2182B"}
    metrics: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    for ax, code in zip(axes.flat, codes):
        sub = curve[(curve["phecode"] == code) & (curve["time_years"] <= 3.0001)].copy()
        metrics[code] = {}
        for group in group_order:
            g = sub[sub["group"] == group]
            n0 = int(g.loc[g["time_years"].idxmin(), "n_at_risk"])
            ax.step(g["time_years"], g["cif_pct"], where="post", lw=1.8, color=colors[group], label=f"{group} (n={n0:,})")
            last = g.iloc[(g["time_years"] - 3).abs().argsort()[:1]]
            metrics[code][group] = float(last["cif_pct"].iloc[0])
        ax.set_title(titles[code], fontweight="bold")
        ax.set_xlim(0, 3)
        ax.set_xlabel("Years since index sleep study")
        ax.set_ylabel("Cumulative incidence (%)")
        ax.grid(color=GRID, lw=0.5)
        ax.legend(frameon=False, loc="upper left")
        times = [0, 1, 2, 3]
        risk_rows = []
        for group in group_order:
            g = sub[sub["group"] == group]
            vals = []
            for t in times:
                row = g.iloc[(g["time_years"] - t).abs().argsort()[:1]]
                vals.append(f"{int(row['n_at_risk'].iloc[0]):,}")
            risk_rows.append(vals)
        tab = ax.table(cellText=risk_rows, rowLabels=group_order, colLabels=[str(t) for t in times], cellLoc="center", rowLoc="right", bbox=[0.12, -0.50, 0.82, 0.31])
        tab.auto_set_font_size(False)
        tab.set_fontsize(6)
        for cell in tab.get_celld().values():
            cell.set_linewidth(0)
            cell.set_facecolor("white")
    fig.subplots_adjust(hspace=0.78, wspace=0.28, bottom=0.11, top=0.92)
    fig.suptitle("Aalen-Johansen cumulative incidence by OSA severity (death as competing event)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "figure3_severity_cif.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    adherence = pd.read_csv(ROOT / "results/incwas_results/survival_curves_landmark/curve_adherence.csv", dtype={"phecode": str})
    sub = adherence[(adherence["phecode"] == "401.1") & (adherence["window_days"] == 180) & (adherence["time_years"] <= 3.0001)].copy()
    order = ["4+ hr/night", "2-4 hr/night", "0-2 hr/night", "never-started", "unknown"]
    colors_a = {
        "4+ hr/night": "#2166AC",
        "2-4 hr/night": "#67A9CF",
        "0-2 hr/night": "#EF8A62",
        "never-started": "#B2182B",
        "unknown": "#888888",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    pap_risk = {}
    for group in order:
        g = sub[sub["group"] == group]
        n0 = int(g.loc[g["time_years"].idxmin(), "n_at_risk"])
        ls = ":" if group == "unknown" else "-"
        ax.step(g["time_years"], g["cif_pct"], where="post", lw=1.9, ls=ls, color=colors_a[group], label=f"{group} (n={n0:,})")
        last = g.iloc[(g["time_years"] - 3).abs().argsort()[:1]]
        pap_risk[group] = float(last["cif_pct"].iloc[0])
    ax.set_xlim(0, 3)
    ax.set_xlabel("Years after the 180-day landmark")
    ax.set_ylabel("Cumulative incidence of essential hypertension (%)")
    ax.set_title("Essential hypertension cumulative incidence by PAP-use category", fontweight="bold")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "figure4_pap_landmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pap_results = pd.read_csv(ROOT / "results/incwas_results/analysis/results_cpap_landmark_180.csv", dtype={"phecode": str})
    htn = pap_results[(pap_results["phecode"] == "401.1") & (pap_results["model"] == "m4")]
    pap_effects = {
        row["contrast"]: {
            "effect": float(row["effect"]),
            "ci_low": None if pd.isna(row["ci_low"]) else float(row["ci_low"]),
            "ci_high": None if pd.isna(row["ci_high"]) else float(row["ci_high"]),
            "p": float(row["p"]),
            "sig_fdr": bool(as_bool(pd.Series([row["sig_fdr"]])).iloc[0]),
        }
        for _, row in htn.iterrows()
    }
    return {"severity_3y_cif_pct": metrics, "pap_3y_cif_pct": pap_risk, "pap_htn_m4": pap_effects}


def make_supplement_tables(prev_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = prev_all[prev_all["contrast"] == "ahi_ge5"].copy()
    selected = primary[(primary["model"] == "m4_bmi") & as_bool(primary["sig_fdr"])][
        ["PheWAS Code", "PheWAS Name", "category_string", "p-val"]
    ].drop_duplicates("PheWAS Code")
    selected = selected.rename(columns={"p-val": "m4_sort_p"})

    model_map = {"m1_crude": "m1", "m2_demo": "m2", "m3_smk_ses_win": "m3", "m4_bmi": "m4"}
    s1 = selected.copy()
    for source_model, prefix in model_map.items():
        d = primary[primary["model"] == source_model][["PheWAS Code", "OR", "p-val", "sig_fdr"]].copy()
        d = d.rename(columns={"OR": f"{prefix}_or", "p-val": f"{prefix}_p", "sig_fdr": f"{prefix}_fdr"})
        s1 = s1.merge(d, on="PheWAS Code", how="left")
    m4_ci = primary[primary["model"] == "m4_bmi"][["PheWAS Code", "OR_low", "OR_high"]]
    s1 = s1.merge(m4_ci, on="PheWAS Code", how="left")
    s1 = s1.sort_values(["category_string", "m4_sort_p", "PheWAS Name"], kind="stable")
    s1.to_csv(OUT / "supp_table_s1.csv", index=False)

    dose = pd.read_csv(
        ROOT / "results/icd_analysis_v2_results/summary/dose_response.csv",
        low_memory=False,
        dtype={"PheWAS Code": str},
    )
    s2 = selected.merge(dose, on=["PheWAS Code", "PheWAS Name", "category_string"], how="left")
    s2 = s2.sort_values(["category_string", "m4_sort_p", "PheWAS Name"], kind="stable")
    s2.to_csv(OUT / "supp_table_s2.csv", index=False)
    return s1, s2


def main() -> None:
    _, baseline_metrics = summarize_table1()
    prev, inc, phewas_metrics = make_phewas_figures()
    table2 = make_table2()
    table3 = make_table3(inc)
    survival_metrics = make_survival_figures()
    s1, s2 = make_supplement_tables(
        pd.read_csv(
            ROOT / "results/icd_analysis_v2_results/summary/phewas_results_combined.csv",
            low_memory=False,
            dtype={"PheWAS Code": str},
        )
    )

    dose = pd.read_csv(
        ROOT / "results/icd_analysis_v2_results/summary/dose_response.csv",
        dtype={"PheWAS Code": str},
    )
    key_dose = {}
    for code in ["401.1", "250.2", "272.1", "428.2"]:
        row = dose[dose["PheWAS Code"].astype(str).eq(code)].iloc[0]
        key_dose[code] = {k: float(row[k]) for k in ["OR_mild", "OR_mod", "OR_sev", "OR_ge5"]}

    prev_af = prev[prev["PheWAS Code"].astype(str).eq("427.21")].iloc[0]
    prev_dementia = prev[prev["PheWAS Code"].astype(str).eq("290.1")].iloc[0]
    prev_mci = prev[prev["PheWAS Code"].astype(str).eq("292.2")].iloc[0]
    inc_af = inc[inc["phecode"].astype(str).eq("427.2")].iloc[0]
    metrics = {
        "baseline": baseline_metrics,
        **phewas_metrics,
        **survival_metrics,
        "key_prevalence_dose_response": key_dose,
        "prevalence_af": {"or": float(prev_af["OR"]), "low": float(prev_af["OR_low"]), "high": float(prev_af["OR_high"]), "p": float(prev_af["p-val"]), "fdr": bool(prev_af["sig_fdr"])},
        "prevalence_dementia": {"or": float(prev_dementia["OR"]), "low": float(prev_dementia["OR_low"]), "high": float(prev_dementia["OR_high"]), "p": float(prev_dementia["p-val"]), "fdr": bool(prev_dementia["sig_fdr"])},
        "prevalence_mci": {"or": float(prev_mci["OR"]), "low": float(prev_mci["OR_low"]), "high": float(prev_mci["OR_high"]), "p": float(prev_mci["p-val"]), "fdr": bool(prev_mci["sig_fdr"])},
        "incidence_af": {"hr": float(inc_af["effect"]), "low": float(inc_af["ci_low"]), "high": float(inc_af["ci_high"]), "p": float(inc_af["p"]), "fdr": bool(inc_af["sig_fdr"])},
        "table2_rows": int(len(table2)),
        "table3_rows": int(len(table3)),
        "supp_s1_rows": int(len(s1)),
        "supp_s2_rows": int(len(s2)),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Wrote aggregate assets: Table 1={len(pd.read_csv(OUT / 'table1.csv'))} rows; "
        f"Table 2={len(table2)}; Table 3={len(table3)}; S1={len(s1)}; S2={len(s2)}."
    )


if __name__ == "__main__":
    main()
