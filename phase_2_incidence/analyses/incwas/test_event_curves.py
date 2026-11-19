"""Synthetic checks: every AJ jump survives export, including tied events."""
import unittest
import warnings

import numpy as np
from lifelines import AalenJohansenFitter

from _event_curves import event_curve


class EventCurveTests(unittest.TestCase):
    def test_preserves_entire_fitted_step_function(self):
        t = np.array([0.1, 0.101, 0.101, 0.8, 1.01, 1.01, 1.8, 2.1])
        ev = np.array([1, 1, 2, 0, 1, 2, 1, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = AalenJohansenFitter(calculate_variance=False, seed=0).fit(t, ev, 1)
            exported = event_curve(t, ev, 2)
        cif = fitted.cumulative_density_.iloc[:, 0]
        jumps = cif.index.to_numpy()[cif.diff().fillna(0).to_numpy() > 0]
        self.assertTrue(set(jumps[jumps <= 2]).issubset(exported.time_years))
        self.assertTrue({0, 1, 2}.issubset(exported.time_years))
        probes = np.unique(np.r_[cif.index.to_numpy(), np.linspace(0, 2, 1001)])
        probes = probes[(probes >= 0) & (probes <= 2)]
        expected = cif.to_numpy()[np.searchsorted(cif.index, probes, side="right") - 1] * 100
        actual = exported.cif_pct.to_numpy()[np.searchsorted(exported.time_years, probes, side="right") - 1]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(exported.n_at_risk, (t[:, None] >= exported.time_years.to_numpy()).sum(axis=0))
        np.testing.assert_array_equal(exported.n_events_cum, (t[ev == 1, None] <= exported.time_years.to_numpy()).sum(axis=0))


if __name__ == "__main__":
    unittest.main()
