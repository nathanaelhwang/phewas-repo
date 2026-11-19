"""Rebuild plot tables from the existing no-ID survival exports, entirely locally.

No patient rows are printed. Original day-resolution durations are recovered
from the exports' five-decimal years; the seeded AJ fit remains unchanged.
Existing curve values are checked at every old coordinate before replacement.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from _event_curves import event_curve

ROOT = Path(__file__).resolve().parents[2] / "results" / "incwas_results"
SEVERITY = ROOT / "survival_curves"
LANDMARK = ROOT / "survival_curves_landmark"


def load_long(path):
    data = pd.read_parquet(path)
    days = data["time_years"].to_numpy(float) * 365.25
    if not np.all(np.abs(days - np.rint(days)) <= 0.001827):
        raise ValueError("Saved durations do not match five-decimal, whole-day follow-up")
    data["time_years"] = np.rint(days) / 365.25
    return data


def curve(data, metadata, horizon):
    if len(data) < 20 or (data["etype"] == 1).sum() < 5:
        return None
    fitted = event_curve(data["time_years"], data["etype"], horizon)
    return pd.DataFrame({**metadata, **fitted.to_dict("list")})


def validate_against_existing(path, updated, keys):
    old = pd.read_csv(path, dtype={"phecode": str})
    old_groups = dict(list(old.groupby(keys, sort=False)))
    new_groups = dict(list(updated.groupby(keys, sort=False)))
    if old_groups.keys() != new_groups.keys():
        raise ValueError(f"{path.name}: curve membership changed")
    maximum_error = 0.0
    for key, prior in old_groups.items():
        new = new_groups[key]
        times = new["time_years"].to_numpy()
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError(f"{path.name}: invalid event-time coordinates")
        if np.any(np.diff(new["cif_pct"]) < 0):
            raise ValueError(f"{path.name}: decreasing cumulative incidence")
        pos = np.searchsorted(times, prior["time_years"], side="right") - 1
        error = np.max(np.abs(new["cif_pct"].to_numpy()[pos] - prior["cif_pct"].to_numpy()))
        maximum_error = max(maximum_error, error)
    if maximum_error > 0.000050001:
        raise ValueError(f"{path.name}: prior CIF mismatch exceeds rounding tolerance ({maximum_error:.8g})")
    print(f"{path.name}: {len(new_groups)} curves; {len(updated):,} coordinates; "
          f"maximum prior-grid difference {maximum_error:.8g} percentage points")


def main():
    warnings.filterwarnings("ignore", message="(?s).*Tied event times were detected.*")
    severity_parts, cpap_parts, landmark_parts = [], [], []
    data = load_long(SEVERITY / "survival_long.parquet")
    for (phecode, phenotype), sub in data.groupby(["phecode", "phenotype"], sort=False):
        base = {"phecode": phecode, "phenotype": phenotype}
        for group in ("No OSA", "Mild", "Moderate", "Severe"):
            rows = curve(sub[sub["severity"] == group], {**base, "group": group}, 6)
            if rows is not None:
                severity_parts.append(rows)
        for panel in ("Mild", "Moderate", "Severe"):
            rows = curve(sub[sub["severity"] == "No OSA"], {**base, "panel": panel, "group": "No OSA"}, 6)
            if rows is not None:
                cpap_parts.append(rows)
            for group in ("<2 hr/night", "2-4 hr/night", "4+ hr/night"):
                rows = curve(sub[(sub["severity"] == panel) & (sub["cpap_bin"] == group)],
                             {**base, "panel": panel, "group": group}, 6)
                if rows is not None:
                    cpap_parts.append(rows)
    del data
    data = load_long(LANDMARK / "survival_long_landmark.parquet")
    for (phecode, phenotype, window), sub in data.groupby(["phecode", "phenotype", "window_days"], sort=False):
        for group in ("4+ hr/night", "0-2 hr/night", "2-4 hr/night", "never-started", "unknown"):
            rows = curve(sub[sub["adherence"] == group],
                         {"phecode": phecode, "phenotype": phenotype, "window_days": window, "group": group}, 5)
            if rows is not None:
                landmark_parts.append(rows)
    outputs = [
        (SEVERITY / "curve_severity.csv", pd.concat(severity_parts, ignore_index=True), ["phecode", "group"]),
        (SEVERITY / "curve_cpap.csv", pd.concat(cpap_parts, ignore_index=True), ["phecode", "panel", "group"]),
        (LANDMARK / "curve_adherence.csv", pd.concat(landmark_parts, ignore_index=True), ["phecode", "window_days", "group"]),
    ]
    for path, updated, keys in outputs:
        validate_against_existing(path, updated, keys)
    for path, updated, _ in outputs:
        temporary = path.with_suffix(".csv.tmp")
        updated.to_csv(temporary, index=False)
        temporary.replace(path)


if __name__ == "__main__":
    main()
