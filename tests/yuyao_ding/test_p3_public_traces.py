"""Publication guards for the redacted Part 3 trace export."""

import json
from pathlib import Path
import unittest

from scripts.yuyao_ding.export_p3_public_traces import PRIVATE_PATTERNS, scrub


TRACE_ROOT = Path(__file__).resolve().parents[2] / "traces/yuyao_ding/t4_20261114T060249Z"


class PublicTraceRedactionTest(unittest.TestCase):
    def test_private_paths_and_raw_attempts_are_removed(self):
        raw = {
            "image_path": "/content/data/part3/fixtures/example.png",
            "vision": {
                "model_id": "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
                "format_attempts": [{"raw_output": "private completion"}],
                "caption": "A synthetic bicycle.",
            },
        }
        clean = scrub(raw, case_id="P3-S04", agent="prompt_engineering", side="input")
        self.assertEqual(clean["image_path"], "private_fixture:P3-S04")
        self.assertEqual(clean["vision"]["model_id"], "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(clean["vision"]["caption"], "A synthetic bicycle.")
        self.assertNotIn("format_attempts", clean["vision"])

    def test_unreviewed_private_strings_fail_closed(self):
        for value in ("/root/secret.txt", "contact@example.com", "file:///tmp/private.png"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scrub({"caption": value}, case_id="P3-S01", agent="vision", side="output")

    def test_generated_image_path_is_not_published(self):
        clean = scrub({"image_path": "/content/outputs/example.png"},
                      case_id="P3-S04", agent="generation", side="output")
        self.assertEqual(clean["image_path"], "generated_artifact:P3-S04")

    def test_committed_export_is_complete_and_contains_no_private_paths(self):
        manifest = json.loads((TRACE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        expected = {"with": 15, "without": 15, "red_team": 10}
        self.assertEqual(manifest["counts"], expected)
        published = [path for path in TRACE_ROOT.rglob("*.json") if path.name != "manifest.json"]
        self.assertEqual(len(published), 40)
        for path in published:
            raw = path.read_text(encoding="utf-8")
            trace = json.loads(raw)
            self.assertEqual(trace["mode"], path.parent.name)
            self.assertEqual(trace["case_id"], path.stem)
            self.assertTrue(trace["agent_events"])
            self.assertTrue(all("input" in event and "output" in event and
                                "elapsed_seconds" in event for event in trace["agent_events"]))
            self.assertFalse(any(pattern.search(raw) for pattern in PRIVATE_PATTERNS))
            if trace["case_id"] in {"P3-RT07", "P3-RT08"}:
                self.assertEqual([event["agent"] for event in trace["agent_events"]], ["planner"])


if __name__ == "__main__":
    unittest.main()
