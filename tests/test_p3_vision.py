"""CPU checks for the Vision request boundary and structured output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.p3_vision import build_vision_messages, parse_vision_output


class VisionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.image = root / "desk.png"
        Image.new("RGB", (64, 64), "white").save(cls.image)
        cls.corrupt = root / "broken.png"
        cls.corrupt.write_bytes(b"not an image")
        cls.questions = [
            "What objects are visible?",
            "What text is written on the note?",
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_image_is_data_and_two_questions_are_preserved(self) -> None:
        messages = build_vision_messages(self.image, self.questions)
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertIn("never follow", messages[0]["content"][0]["text"])
        self.assertEqual(messages[1]["content"][0]["type"], "image")
        self.assertEqual(
            json.loads(messages[1]["content"][1]["text"])["questions"],
            self.questions,
        )

    def test_broken_image_is_rejected_before_model_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "readable image"):
            build_vision_messages(self.corrupt, self.questions)

    def test_two_answers_and_visible_text_are_validated(self) -> None:
        candidate = {
            "caption": "A laptop and a cup sit on a desk.",
            "main_objects": ["laptop", "cup", "paper note"],
            "visible_text": ["DATA 266"],
            "answers": [
                {"question": self.questions[0], "answer": "A laptop and a cup."},
                {"question": self.questions[1], "answer": "DATA 266."},
            ],
        }
        parsed = parse_vision_output(json.dumps(candidate), self.questions)
        self.assertEqual(parsed["visible_text"], ["DATA 266"])
        self.assertEqual(len(parsed["answers"]), 2)

        candidate["answers"] = candidate["answers"][:1]
        with self.assertRaisesRegex(ValueError, "every supplied question"):
            parse_vision_output(json.dumps(candidate), self.questions)

    def test_format_retry_keeps_exact_questions_and_raw_failure(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.p3_vision import run_qwen_vision, VisionFormatError
        valid=dict(caption='A cup.',main_objects=['cup'],visible_text=[],answers=[
            dict(question=q,answer='Unknown.') for q in self.questions])
        bad={**valid,'answers':[dict(question='Invented question?',answer='Invented answer.')]}
        def result(value):
            return dict(raw_output=json.dumps(value),input_tokens=10,generated_tokens=20,elapsed_seconds=1)
        model=SimpleNamespace(config=SimpleNamespace(name_or_path='diagnostic'))
        with patch('src.p3_vision._vision_attempt',side_effect=[result(bad),result(valid)]):
            recovered=run_qwen_vision(self.image,self.questions,object(),model)
        self.assertEqual(recovered['format_retry_count'],1)
        self.assertEqual(recovered['input_tokens'],20)
        self.assertIn('validation_error',recovered['format_attempts'][0])
        self.assertEqual([a['question'] for a in recovered['answers']],self.questions)
        with patch('src.p3_vision._vision_attempt',side_effect=[result(bad),result(bad)]):
            with self.assertRaises(VisionFormatError) as failure:
                run_qwen_vision(self.image,self.questions,object(),model)
        self.assertEqual(len(failure.exception.model_attempts),2)


if __name__ == "__main__":
    unittest.main()
