"""Synthetic release-contract checks; no cohort files are opened."""
import importlib.util
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def load_validator():
    path = Path(__file__).with_name("29_octant_export_validate.py")
    spec = importlib.util.spec_from_file_location("octant_export_validate", path)
    module = importlib.util.module_from_spec(spec)
    # The validator only needs paths from these modules; stubs avoid loading any
    # unrelated analysis dependencies while exercising its release checks.
    stubs = {
        "incwas_utils": types.SimpleNamespace(),
        "_pheno_engine": types.SimpleNamespace(OUT=Path("synthetic")),
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


V = load_validator()


class DisclosurePathValidationTests(unittest.TestCase):
    def test_http_urls_do_not_match_local_drive_prefixes(self):
        for value in ("https://safe", "http://safe", '"policy": "https://safe/policy"'):
            with self.subTest(value=value):
                self.assertIsNone(V.PATH_RE.search(value))

    def test_local_paths_match_standalone_and_embedded(self):
        for value in ("C:/", "D:\\", "/Users/foo", "\\Users\\",
                      "path: C:/some folder/export.csv", r'"path": "D:\\some folder\\file"',
                      "path=/Users/foo/some folder", r"path: \\Users\\foo"):
            with self.subTest(value=value):
                self.assertIsNotNone(V.PATH_RE.search(value))


class ManifestExtensionPreservationTests(unittest.TestCase):
    def manifests(self):
        old = {
            "files": [],
            "disclosure": {"note": "Audited race aggregate release.",
                           "suppressed_cells_by_file": {}, "suppressed_rows_by_file": {}},
            "unavailable_or_excluded": [{"field": "CPAP treatment", "reason": "Withheld"}],
        }
        new = {
            "files": [], "disclosure": {"note": "Old note"},
            "unavailable_or_excluded": [
                {"field": "race/ethnicity breakdown"}, {"field": "CPAP treatment"},
                {"field": "adjusted curves", "reason": "Unchanged"}],
        }
        for section, names in V.MANIFEST_EXTENSION_FILES.items():
            approved = section == "race_ethnicity"
            old[section] = {"files": dict(enumerate(names)), "release_approved": approved}
            for name in names:
                entry = {"name": name, "sha256": "a" * 64, "bytes": 1, "internal": False}
                new["files"].append(entry)
                old["files"].append({**entry, "schema": ["synthetic"], "rows": 1,
                                     "public_website_release_approved": approved,
                                     "disclosure_audit_status": "PASS"})
                old["disclosure"]["suppressed_cells_by_file"][name] = 2
                old["disclosure"]["suppressed_rows_by_file"][name] = 1
        return new, old

    def test_preserves_extensions_and_approvals_only_for_identical_artifacts(self):
        new, old = self.manifests()
        V._preserve_manifest_extensions(new, old)
        for section in V.MANIFEST_EXTENSION_FILES:
            self.assertEqual(new[section], old[section])
        self.assertEqual(new["files"], old["files"])
        self.assertEqual(new["disclosure"], old["disclosure"])
        self.assertEqual(new["unavailable_or_excluded"], [
            {"field": "adjusted curves", "reason": "Unchanged"},
            {"field": "CPAP treatment", "reason": "Withheld"}])

    def test_rejects_stale_or_missing_artifacts_before_preserving_any_approval(self):
        new, old = self.manifests()
        for change in ("hash", "missing", "bytes"):
            altered = copy.deepcopy(new)
            if change == "hash":
                altered["files"][-1]["sha256"] = "b" * 64
            elif change == "bytes":
                altered["files"][-1]["bytes"] = 2
            else:
                altered["files"].pop()
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                V._preserve_manifest_extensions(altered, old)
            self.assertNotIn("race_ethnicity", altered)

    def test_rejects_artifacts_without_prior_extension_decisions(self):
        new, old = self.manifests()
        del old["cpap_treatment_summary"]
        with self.assertRaises(RuntimeError):
            V._preserve_manifest_extensions(new, old)


class EventTimelineValidationTests(unittest.TestCase):
    def test_accepts_variable_full_precision_event_times(self):
        for time in (
            [0, 1, 2, 3],
            [0, 0.001234567891234, 0.001234567891235, 1, 1.99, 2, 3],
            [0, 0.1234567890123456, 1, 2, 2.9999999999999996, 3],
        ):
            with self.subTest(time=time):
                self.assertTrue(V._event_times_valid(pd.Series(time)))

    def test_rejects_missing_checkpoints_and_sampled_grid(self):
        for time in ([0, 1, 2, 2.9999], [0, 0.5, 2, 3],
                     np.linspace(0, 3, 120)):
            with self.subTest(time=time):
                self.assertFalse(V._event_times_valid(pd.Series(time)))

    def test_rejects_duplicate_unsorted_and_invalid_times(self):
        for time in ([0, 1, 1, 2, 3], [0, 2, 1, 3], [0, 1, 2, 3.1],
                     [0, 1, np.nan, 2, 3], [0, 1, np.inf, 2, 3],
                     [0, 1, "invalid", 2, 3], [-0.1, 0, 1, 2, 3], []):
            with self.subTest(time=time):
                self.assertFalse(V._event_times_valid(pd.Series(time)))


class SamplingEquivalenceValidationTests(unittest.TestCase):
    def evidence(self, prefix="focal"):
        return {
            f"{prefix}_sampled_equivalent": True,
            f"{prefix}_max_abs_diff_vs_previous_sampled_pp": 0.0000005,
            f"{prefix}_risk_table_unchanged": True,
            f"{prefix}_metadata_unchanged": True,
        }

    def test_accepts_only_old_rounding_error(self):
        for prefix in ("focal", "overall"):
            evidence = self.evidence(prefix)
            self.assertTrue(V._sampling_evidence_valid(evidence, prefix))
            evidence[f"{prefix}_max_abs_diff_vs_previous_sampled_pp"] = 0.00000052
            self.assertFalse(V._sampling_evidence_valid(evidence, prefix))

    def test_rejects_changed_aggregates_or_missing_comparison(self):
        for key in ("sampled_equivalent", "risk_table_unchanged", "metadata_unchanged"):
            evidence = self.evidence()
            evidence[f"focal_{key}"] = False
            self.assertFalse(V._sampling_evidence_valid(evidence, "focal"))
        for value in (None, np.nan, np.inf, -0.1):
            evidence = self.evidence()
            evidence["focal_max_abs_diff_vs_previous_sampled_pp"] = value
            self.assertFalse(V._sampling_evidence_valid(evidence, "focal"))

    def test_parses_event_time_audit_and_fails_missing_evidence(self):
        aud = {
            "curve_granularity": "event_time", "horizon_years": "3.0",
            **{key: str(value) for prefix in ("focal", "overall")
               for key, value in self.evidence(prefix).items()},
        }
        parsed = V._overall_evidence(aud)
        self.assertEqual(parsed["curve_granularity"], "event_time")
        self.assertEqual(parsed["horizon_years"], 3.0)
        self.assertTrue(V._sampling_evidence_valid(parsed, "focal"))
        self.assertTrue(V._sampling_evidence_valid(parsed, "overall"))
        missing = V._overall_evidence({})
        self.assertFalse(V._sampling_evidence_valid(missing, "focal"))


if __name__ == "__main__":
    unittest.main()
