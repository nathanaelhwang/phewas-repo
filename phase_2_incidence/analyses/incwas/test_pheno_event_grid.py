"""Synthetic checks for the tie-aware phenotype CIF event-time display."""
import unittest
import numpy as np
import _pheno_engine as P


class PhenotypeEventGridTests(unittest.TestCase):
    def test_keeps_every_event_and_ties_without_sampling(self):
        t = np.array([0.1, 0.1, 0.101, 0.4, 0.9, 1.01, 2.99, 3.0, 3.1])
        event = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1])
        death = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0])
        grid = P.event_time_grid(t, event)
        np.testing.assert_array_equal(grid, [0, 0.1, 0.101, 1, 1.01, 2, 3])
        cif = P.aalen_johansen(t, event, death, grid)
        probes = np.unique(np.r_[t[t <= 3], np.linspace(0, 3, 1000)])
        expected = P.aalen_johansen(t, event, death, probes)
        actual = cif[np.searchsorted(grid, probes, side="right") - 1]
        np.testing.assert_array_equal(actual, expected)
        # Same-day events share a risk set and produce one combined jump.
        self.assertAlmostEqual(cif[1], 2 / 9)

    def test_empty_incident_events_have_only_checkpoints(self):
        np.testing.assert_array_equal(P.event_time_grid([0.5, 2.0], [0, 0]), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
