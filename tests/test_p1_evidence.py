import json
import os
from pathlib import Path
import unittest

from src.p1_evidence import extract_structured_facts, resolve_structured_query


class StructuredEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        input_dir_value = os.environ.get("DATA266_PART1_INPUT_DIR")
        if not input_dir_value:
            raise unittest.SkipTest("Set DATA266_PART1_INPUT_DIR to the seven Part 1 inputs.")
        cls.input_dir = Path(input_dir_value)
        manifest = json.loads((cls.input_dir / "kb_manifest.json").read_text())
        documents = {item["doc_id"]: item for item in manifest["documents"]}
        cls.structured = extract_structured_facts(cls.input_dir, documents)

    def resolve(self, question, context=None):
        result = resolve_structured_query(question, self.structured, context)
        self.assertIsNotNone(result)
        return result["answer"]

    def test_precedence_arithmetic_and_absence(self):
        cases = (
            ("What was world output growth in 2010?", "answered", "4.0%"),
            (
                "What was the forecast error for world output growth in 2011?",
                "answered",
                "-0.2 percentage points",
            ),
            ("What was U.S. output growth in 2010?", "answered", "2.4%"),
            ("What was India unemployment in 2010?", "not_found", "does not support"),
        )
        for question, status, phrase in cases:
            with self.subTest(question=question):
                answer = self.resolve(question)
                self.assertEqual(answer["status"], status)
                self.assertIn(phrase, answer["answer"])

    def test_source_conflict_prefers_higher_trust(self):
        answer = self.resolve(
            "Do the sources agree on the U.S. unemployment rate in 2010?"
        )
        self.assertEqual(answer["status"], "conflict")
        self.assertEqual(answer["citations"], ["DOC1:p5", "DOC3:p5"])
        self.assertIn("prefer the higher-trust value of 9.6%", answer["answer"])

    def test_followup_context_and_session_overlay(self):
        followup = self.resolve(
            "Which was larger, and by how much?",
            {
                "prior_turns": [
                    {"question": "What were U.S. federal revenues and spending in 2013?"}
                ],
                "session_overrides": [],
            },
        )
        self.assertIn("larger by 4.1 percentage points", followup["answer"])

        session_answer = self.resolve(
            (
                "For this conversation only, assume U.S. unemployment in 2013 was "
                "8.0%. How many percentage points below the Euro area rate was it?"
            ),
            {
                "prior_turns": [],
                "session_overrides": [
                    {
                        "subject": "U.S.",
                        "metric": "unemployment",
                        "period": "2013",
                        "value": "8.0",
                        "unit": "percent",
                    }
                ],
            },
        )
        self.assertEqual(session_answer["status"], "session_override")
        self.assertEqual(session_answer["citations"], ["DOC2:p3"])
        self.assertIn("4.0 percentage points", session_answer["answer"])

    def test_low_trust_page_is_never_parsed(self):
        self.assertTrue(
            all(
                not (fact["doc_id"] == "DOC3" and fact["page"] == 7)
                for fact in self.structured["facts"]
            )
        )


if __name__ == "__main__":
    unittest.main()
