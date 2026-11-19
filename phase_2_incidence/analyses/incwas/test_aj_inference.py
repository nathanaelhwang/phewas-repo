"""Synthetic checks of grouped AJ inference; no clinical input is loaded."""
import unittest
import numpy as np
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from _aj_inference import aalen_johansen_ci, cause_specific_logrank
from _pheno_engine import aalen_johansen


class AJInferenceTests(unittest.TestCase):
    def test_untied_variance_matches_lifelines(self):
        t = np.arange(1, 41, dtype=float)
        status = np.resize([1, 0, 2, 1, 0], 40)
        fitted = AalenJohansenFitter().fit(t, status, event_of_interest=1)
        f, v, low, high = aalen_johansen_ci(t, status == 1, status == 2, t)
        np.testing.assert_allclose(f, fitted.cumulative_density_.iloc[1:, 0], atol=1e-14)
        np.testing.assert_allclose(v, fitted.variance_.iloc[1:], atol=1e-14)
        bounds = fitted.confidence_interval_.iloc[1:].to_numpy()
        np.testing.assert_allclose(low, bounds.min(axis=1), atol=1e-14)
        np.testing.assert_allclose(high, bounds.max(axis=1), atol=1e-14)

    def test_ties_preserve_cif_and_permutation(self):
        t = np.resize(np.arange(1, 16), 120)/5
        status = np.resize([1, 2, 0, 1, 0, 0, 2], 120)
        grid = np.arange(0, 3.01, .1)
        result = aalen_johansen_ci(t, status == 1, status == 2, grid)
        np.testing.assert_allclose(result[0], aalen_johansen(t, status == 1, status == 2, grid), atol=1e-14)
        order = np.random.default_rng(18).permutation(len(t))
        other = aalen_johansen_ci(t[order], status[order] == 1, status[order] == 2, grid)
        for a, b in zip(result, other):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(result[2] <= result[0]))
        self.assertTrue(np.all(result[0] <= result[3]))

    def test_tied_single_cause_matches_greenwood(self):
        t = np.repeat(np.arange(1, 16), 8)
        event = np.resize([1, 1, 0], len(t))
        grid = np.arange(1, 16)
        fitted = KaplanMeierFitter().fit(t, event)
        f, v, _, _ = aalen_johansen_ci(t, event, np.zeros(len(t)), grid)
        table = fitted.event_table.iloc[1:]
        greenwood = np.cumsum(table.observed/(table.at_risk*(table.at_risk-table.observed)))
        survival = fitted.survival_function_.iloc[1:, 0]
        np.testing.assert_allclose(f, 1-survival, atol=1e-14)
        np.testing.assert_allclose(v, survival**2*greenwood, atol=1e-14)

    def test_no_censoring_matches_multinomial_variance(self):
        t = np.array([1, 1, 2, 2, 3, 3])
        status = np.array([1, 2, 1, 2, 1, 2])
        f, v, _, _ = aalen_johansen_ci(t, status == 1, status == 2, [0, 1, 2, 3])
        np.testing.assert_allclose(v, f*(1-f)/len(t), atol=1e-14)

    def test_horizon_censors_later_events(self):
        t = np.array([.5, 1., 4., 5., .7, 1.5, 4.5, 6.])
        event = np.array([1, 0, 1, 1, 0, 1, 1, 1])
        focal = np.arange(len(t)) < 4
        expected = cause_specific_logrank(np.minimum(t, 3), event*(t <= 3), focal)
        self.assertEqual(expected, cause_specific_logrank(t, event, focal))


if __name__ == "__main__":
    unittest.main()
