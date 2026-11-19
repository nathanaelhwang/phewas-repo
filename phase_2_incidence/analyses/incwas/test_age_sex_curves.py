"""Synthetic boundary, competing-risk, and sparse-stratum checks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("age_sex", Path(__file__).with_name("34_age_sex_curves.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AgeSexCurveTests(unittest.TestCase):
    def test_age_boundaries_and_missing_sex(self):
        data = pd.DataFrame({"age": [17.9, 18, 39.99, 40, 60, 60.01, np.nan, 50],
                             "sex_M": [1, 1, 0, 1, 0, 1, 0, np.nan]})
        actual = analysis.assign_strata(data)
        self.assertEqual(actual.age_group.fillna("excluded").tolist(),
                         ["excluded", "18-<40y", "18-<40y", "40-60y", "40-60y", ">60y", "excluded", "40-60y"])
        self.assertTrue(pd.isna(actual.sex.iloc[-1]))
        self.assertEqual(actual.sex.iloc[2], "Female")

    def test_competing_deaths_and_horizon_counts(self):
        # Ten deaths precede five incident events: AJ risk = 5/20, not 5/10.
        data = pd.DataFrame({"time": np.arange(1, 21)/4,
                             "etype": [2]*10 + [1]*5 + [0]*5})
        labels = dict(zip(analysis.KEYS, ["synthetic", "Example", False, "40-60y", "Male", "No OSA"]))
        curve, risk, summary = analysis.summarize_curve(data, labels)
        self.assertAlmostEqual(curve.cif_pct.iloc[-1], 25)
        self.assertEqual(summary["n_deaths_6y"], 10)
        self.assertEqual(risk.n_at_risk.iloc[0], 20)
        self.assertEqual(risk.n_at_risk.iloc[6], 0)
        self.assertEqual(curve.time_years.iloc[-1], 5)
        self.assertTrue(np.isnan(summary["cif_6y_pct"]))

    def test_sparse_events_and_events_beyond_display(self):
        data = pd.DataFrame({"time": [1.]*4 + [7.]*16, "etype": [1]*20})
        curve, _, summary = analysis.summarize_curve(data, {})
        self.assertIsNone(curve)
        self.assertFalse(summary["curve_plotted"])
        self.assertEqual(summary["n_events_6y"], 4)
        curve, risk, summary = analysis.summarize_curve(data.iloc[:0], {})
        self.assertIsNone(curve)
        self.assertEqual(summary["n_atrisk"], 0)
        self.assertTrue((risk.n_at_risk == 0).all())


if __name__ == "__main__":
    unittest.main()
