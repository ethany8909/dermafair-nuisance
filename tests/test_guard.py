"""Q4: the protected-stage guard, on synthetic records only. No stage, model, data or protected outcome is executed.

Adapted from the research-track lead's review harness (2026-10-18), which found four guard defects:
a newer failed stage-0 run could be skipped, missing or unknown QC could pass, recorded software versions were not
compared, and a string could pass the eligibility check. The fixtures are the committed stage-0 evidence
(run stage0-20261013T012829Z and its qc/Q2_admissible.json), both aggregate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuisance import stages

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class StageGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.qc = Path(self.tmp.name)
        self.record = json.loads((FIXTURES / "stage0_run_record.json").read_text(encoding="utf-8"))
        self.adm = json.loads((FIXTURES / "Q2_admissible.json").read_text(encoding="utf-8"))
        self.software = copy.deepcopy(self.record["software"])
        self.rows = [self.record]
        self.path = self.qc / "Q2_admissible.json"
        self.write_admissibility()
        (self.qc / "Q3_review.md").write_text("Reviewed: yes (synthetic fixture only)\n", encoding="utf-8")
        for target, value in (("QC_DIR", self.qc), ("ADMISSIBLE", self.path)):
            p = mock.patch.object(stages, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.auth = self.patch(stages.P, "authorized_stages", return_value={0, 1})
        self.patch(stages.P, "records", side_effect=lambda n=None: self.rows if n == 0 else [])
        self.patch(stages.P, "code_changed_since", return_value=False)
        self.versions = self.patch(stages.P, "versions", side_effect=lambda: self.software)

    def patch(self, obj, name, **kwargs):
        p = mock.patch.object(obj, name, **kwargs)
        result = p.start()
        self.addCleanup(p.stop)
        return result

    def write_admissibility(self):
        raw = (json.dumps(self.adm, indent=2) + "\n").encode()
        self.path.write_bytes(raw)
        self.record["outputs_sha256"]["qc/Q2_admissible.json"] = hashlib.sha256(raw).hexdigest()

    def assert_rejected(self):
        with self.assertRaises(PermissionError):
            stages.guard(1)

    # positive controls
    def test_valid_retained_24_cell_record_is_admitted(self):
        auth, adm = stages.guard(1)
        self.assertEqual(auth["stage0_run"], self.record["run_id"])
        self.assertEqual(adm["n_surviving"], 24)
        self.assertEqual(adm["families"]["A"]["interval"], [20, 75])

    def test_valid_25_cell_record_is_admitted(self):
        self.adm["families"]["A"] = {"k": 5, "interval": [10, 75]}
        self.adm["surviving_cp"].append("A5")
        self.adm["reclassified_to_ca"] = []
        self.adm["n_surviving"] = 25
        self.record["qc"]["Q2_colour_preservation"] = "pass"
        self.write_admissibility()
        self.assertEqual(stages.guard(1)[1]["n_surviving"], 25)

    # the four repaired defects
    def test_later_failed_stage0_cannot_fall_back_to_an_older_pass(self):
        failure = copy.deepcopy(self.record)
        failure.update(run_id="stage0-20261014T012829Z", status="failed")
        self.rows.append(failure)
        self.assert_rejected()

    def test_empty_qc_record_is_rejected(self):
        self.record["qc"] = {}
        self.assert_rejected()

    def test_missing_q1_is_rejected(self):
        del self.record["qc"]["Q1_identity"]
        self.assert_rejected()

    def test_missing_q2_is_rejected(self):
        del self.record["qc"]["Q2_colour_preservation"]
        self.assert_rejected()

    def test_missing_q3_handoff_is_rejected(self):
        del self.record["qc"]["Q3_visual_check"]
        self.assert_rejected()

    def test_missing_q4_is_rejected(self):
        del self.record["qc"]["Q4_unit_tests"]
        self.assert_rejected()

    def test_q2_unknown_status_is_rejected(self):
        self.record["qc"]["Q2_colour_preservation"] = "not evaluated"
        self.assert_rejected()

    def test_q2_verdict_must_match_retained_cell_count(self):
        self.record["qc"]["Q2_colour_preservation"] = "pass with reclassification: 23 of 25 CP cells survive"
        self.assert_rejected()

    def test_each_software_version_must_match_stage0(self):
        for key in self.software:
            with self.subTest(key=key):
                original = self.software[key]
                self.software[key] = original + ".changed"
                try:
                    self.assert_rejected()
                finally:
                    self.software[key] = original

    def test_missing_software_record_is_rejected(self):
        del self.record["software"]
        self.assert_rejected()

    def test_missing_recorded_codec_is_rejected(self):
        del self.record["software"]["libjpeg_turbo"]
        self.assert_rejected()

    def test_empty_recorded_version_is_rejected(self):
        self.record["software"]["numpy"] = ""
        self.assert_rejected()

    def test_unavailable_current_codec_is_rejected(self):
        self.software["libjpeg_turbo"] = None
        self.assert_rejected()

    def test_string_false_eligibility_is_rejected(self):
        self.adm["stage1_eligible"] = "false"
        self.write_admissibility()
        self.assert_rejected()

    # boundaries that were already in place still hold
    def test_missing_authorization_blocks_before_dependency_reads(self):
        self.auth.return_value = {0}
        self.assert_rejected()
        self.versions.assert_not_called()

    def test_changed_admissibility_hash_is_rejected(self):
        self.path.write_text(self.path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assert_rejected()

    def test_false_eligibility_is_rejected(self):
        self.adm["stage1_eligible"] = False
        self.adm["n_surviving"] = 12
        self.write_admissibility()
        self.assert_rejected()

    def test_unreviewed_contact_sheets_are_rejected(self):
        (self.qc / "Q3_review.md").write_text("Reviewed: no\n", encoding="utf-8")
        self.assert_rejected()

    def test_changed_harness_is_rejected(self):
        with mock.patch.object(stages.P, "code_changed_since", return_value=True):
            self.assert_rejected()
