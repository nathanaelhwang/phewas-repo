"""Table 2 omnibus tests from released aggregate statistics only."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import f
from statsmodels.stats.oneway import anova_generic

ROOT = Path(r"C:\Users\natha\Projects\work\phewas")
SOURCE = ROOT / "results/utilwas_results/website_exports/utilwas_utilization_profile.csv"
OUT = Path(__file__).resolve().parent
GROUPS = ("None", "Mild", "Moderate", "Severe")
METRICS = (
    "outpatient_rate", "inpatient_rate", "ed_rate", "urgent_care_rate",
    "primary_care_rate", "specialty_rate", "hospital_los_days",
)

with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 70
assert all(row["suppressed"].lower() == "false" for row in rows)

results = []
for window in ("1yr", "5yr"):
    for metric in METRICS:
        groups = [next(row for row in rows if
                       (row["window"], row["metric_id"], row["severity"])
                       == (window, metric, group)) for group in GROUPS]
        ns = np.array([int(row["n_nonmissing"]) for row in groups])
        means = np.array([float(row["mean"]) for row in groups])
        variances = np.array([float(row["sd"]) ** 2 for row in groups])
        if metric != "hospital_los_days":
            assert ns.tolist() == [60618, 67040, 44957, 49494]
        calculated = anova_generic(means, variances, ns, use_var="unequal")
        # Independently verify Welch's F and Satterthwaite denominator df.
        weights = ns / variances
        weighted_mean = np.sum(weights * means) / weights.sum()
        correction_sum = np.sum((1 - weights / weights.sum()) ** 2 / (ns - 1))
        expected_f = (np.sum(weights * (means - weighted_mean) ** 2) / 3
                      / (1 + 4 * correction_sum / 15))
        expected_df = 15 / (3 * correction_sum)
        np.testing.assert_allclose(calculated.statistic, expected_f, rtol=1e-12)
        np.testing.assert_allclose(calculated.df[1], expected_df, rtol=1e-12)
        np.testing.assert_allclose(calculated.pvalue, f.sf(expected_f, 3, expected_df),
                                   rtol=1e-8, atol=0)
        classic = anova_generic(means, variances, ns, use_var="equal")
        results.append({
            "window": window, "metric_id": metric,
            "welch_f": float(calculated.statistic), "df_numerator": 3,
            "df_denominator": float(calculated.df[1]),
            "p_value": float(calculated.pvalue),
            "display_p": "<0.0001" if calculated.pvalue < 0.0001
                         else f"{calculated.pvalue:.4f}",
            "classical_anova_f_sensitivity": float(classic.statistic),
            "classical_anova_p_sensitivity": float(classic.pvalue),
            "groups": [{"severity": group, "n": int(n), "mean": float(mean),
                        "sd": float(variance ** 0.5)}
                       for group, n, mean, variance in zip(GROUPS, ns, means, variances)],
            "unit": groups[0]["unit"],
            "admission_level_exploratory": metric == "hospital_los_days",
        })

audit = {"source": str(SOURCE),
         "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         "method": "Welch one-way omnibus ANOVA from aggregate n, mean, sample SD",
         "multiplicity_adjustment": "none; 14 descriptive comparisons",
         "limitations": ["Source summaries have six-decimal precision.",
                         "Normality cannot be assessed from aggregate statistics.",
                         "Admission-level LOS tests assume independent admissions and do not account for repeated admissions within patients."],
         "results": results}
(OUT / "anova_results.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
for result in results:
    print(result["window"], result["metric_id"], result["display_p"])
print("All 14 tests verified against an independent Welch calculation.")
