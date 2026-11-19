"""Synthetic inference checks; no clinical files are loaded."""
import unittest
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from _age_sex_inference import infer, adjust, truncate


class AgeSexInferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(143)
        group = np.repeat(["reference", "exposed"], 120)
        event_time = rng.exponential(np.where(group == "exposed", 2., 5.))
        censor = rng.uniform(3., 9., len(group))
        self.data = pd.DataFrame({"time": np.minimum(event_time, censor),
                                  "etype": np.where(event_time <= censor, 1, 2),
                                  "group": group, "covariate": rng.normal(size=len(group)),
                                  "constant_sex": 0.})
        self.meta = {"layout": "severity_panels", "osa_control": False}

    def test_matches_direct_horizon_truncated_estimators(self):
        lr, hr = infer(self.data, "group", ["reference", "exposed"], "reference",
                       ["covariate", "constant_sex"], self.meta)
        data = truncate(self.data)
        expected_lr = multivariate_logrank_test(data.time, data.group, data.event)
        self.assertEqual(lr["p"], expected_lr.p_value)
        design = data[["time", "event", "covariate"]].assign(exposed=(data.group == "exposed").astype(float))
        expected_hr = CoxPHFitter(penalizer=.1).fit(design, "time", "event", batch_mode=True)
        self.assertAlmostEqual(hr[0]["hr"], expected_hr.hazard_ratios_["exposed"], places=12)
        self.assertEqual(hr[0]["n_parameters"], 2)
        self.assertEqual(lr["df"], 1)
        self.assertLess(hr[0]["ci_low"], hr[0]["hr"])
        self.assertGreater(hr[0]["ci_high"], hr[0]["hr"])

    def test_sparse_reference_is_not_silently_replaced(self):
        self.data.loc[self.data.group == "reference", "etype"] = 0
        lr, hr = infer(self.data, "group", ["reference", "exposed"], "reference", [], self.meta)
        self.assertTrue(np.isnan(lr["p"]))
        self.assertEqual(hr[0]["status"], "reference_curve_omitted")

    def test_deaths_and_later_diagnoses_are_censored(self):
        data = pd.DataFrame({"time": [1., 7., 3., 6.], "etype": [2, 1, 0, 1]})
        actual = truncate(data)
        self.assertEqual(actual.event.tolist(), [0, 0, 0, 1])
        self.assertEqual(actual.time.tolist(), [1., 6., 3., 6.])

    def test_fdr_excludes_controls_sparse_models_and_failures(self):
        rows = pd.DataFrame({"p": [.01, .04, 1e-20, 1e-20, 1e-20],
                             "osa_control": [False, False, True, False, False],
                             "status": ["ok", "ok", "ok", "ok", "unstable"],
                             "low_epv": [False, False, False, True, False],
                             "layout": ["severity_panels"]*5})
        result = adjust(rows)
        np.testing.assert_allclose(result.q_bh.iloc[:2], [.02, .04])
        self.assertTrue(result.q_bh.iloc[2:].isna().all())
        self.assertFalse(result.sig_fdr.iloc[2:].any())


if __name__ == "__main__":
    unittest.main()
