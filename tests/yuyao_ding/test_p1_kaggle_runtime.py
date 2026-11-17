"""Saved-record and tensor-reference regressions; no CUDA/model inference."""

import ast
import csv
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval/yuyao_ding/part1/results/kaggle_20261117T003803178245Z"
NOTEBOOK = ROOT / "notebooks/Yuyao_Ding/p1_kaggle_submission.ipynb"
spec = importlib.util.spec_from_file_location(
    "verify_p1_kaggle", ROOT / "scripts/yuyao_ding/verify_p1_kaggle_results.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def function_node():
    notebook = json.loads(NOTEBOOK.read_text())
    return next(
        node for cell in notebook["cells"] if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_qwen_generation"
    )


class KaggleRecordTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "records"
        shutil.copytree(RESULTS, self.directory)

    def write_csv(self, name, update):
        path = self.directory / name
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames, list(reader)
        update(rows)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        # Test semantic checks even if an invalid file has a matching new hash.
        self.refresh_hash(name)

    def refresh_hash(self, name):
        path = self.directory / "artifact_hashes.json"
        hashes = json.loads(path.read_text())
        hashes["files"][name] = hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
        path.write_text(json.dumps(hashes))

    def test_success_preserves_prior_failure_without_scoring_answers(self):
        result = verifier.verify(self.directory)
        self.assertEqual(result["row_count"], 40)
        self.assertEqual(result["earlier_failed_probes"], 1)
        self.assertEqual(result["earlier_completed_probes"], 1)
        self.assertEqual(result["competition_model_calls"], 22)
        self.assertFalse(result["human_correctness_or_citation_support_scored"])
        self.assertFalse(result["gpu_inference_rerun_by_verifier"])

    def test_original_artifact_tampering_is_rejected(self):
        path = self.directory / "submission.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            verifier.verify(self.directory)

    def test_row_reordering_is_rejected(self):
        self.write_csv("submission.csv", lambda rows: rows.reverse())
        with self.assertRaisesRegex(ValueError, "order differs"):
            verifier.verify(self.directory)

    def test_excluded_source_is_rejected(self):
        self.write_csv("submission.csv", lambda rows: rows[0].update(citations="DOC3:p7"))
        with self.assertRaisesRegex(ValueError, "Excluded page"):
            verifier.verify(self.directory)

    def test_competition_error_is_not_confused_with_an_earlier_probe(self):
        self.write_csv("p1_run_diagnostics.csv", lambda rows: rows[0].update(run_status="error"))
        with self.assertRaisesRegex(ValueError, "Competition execution error"):
            verifier.verify(self.directory)

    def test_cached_decode_guard_is_checked(self):
        name = "generation_diagnostics.jsonl"
        path = self.directory / name
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[-1]["forward_input_lengths"][1] = 2
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        self.refresh_hash(name)
        with self.assertRaisesRegex(ValueError, "Cached decode"):
            verifier.verify(self.directory)


class AttentionLifetimeTest(unittest.TestCase):
    def predicate(self):
        call = next(
            node for node in ast.walk(function_node())
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "object" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "use_gqa_in_sdpa"
        )
        self.assertFalse(any(k.arg == "return_value" for k in call.keywords))
        expression = next(k.value for k in call.keywords if k.arg == "new")
        predicate = eval(compile(ast.Expression(body=expression), "<predicate>", "eval"), {})
        self.assertIsInstance(predicate, FunctionType)
        return predicate

    def test_real_notebook_override_does_not_retain_tensor_arguments(self):
        class TensorStandIn:
            pass

        predicate = self.predicate()
        owner = SimpleNamespace(decide=lambda *args: True)
        references = []
        with patch.object(owner, "decide", new=predicate):
            for _ in range(1000):
                tensor = TensorStandIn()
                references.append(weakref.ref(tensor))
                self.assertFalse(owner.decide(None, tensor))
                del tensor
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))

    def test_old_mock_reproduces_retention_until_call_history_is_cleared(self):
        class TensorStandIn:
            pass

        owner = SimpleNamespace(decide=lambda *args: True)
        with patch.object(owner, "decide", return_value=False) as recorder:
            tensor = TensorStandIn()
            reference = weakref.ref(tensor)
            owner.decide(None, tensor)
            del tensor
            gc.collect()
            self.assertIsNotNone(reference())
            recorder.reset_mock()
            gc.collect()
            self.assertIsNone(reference())

    def test_override_is_restored_on_failure(self):
        original = lambda *args: True
        owner = SimpleNamespace(decide=original)
        with self.assertRaises(RuntimeError):
            with patch.object(owner, "decide", new=self.predicate()):
                raise RuntimeError("Synthetic generation failure")
        self.assertIs(owner.decide, original)

    def test_published_function_matches_the_actual_uploaded_runtime(self):
        live = ast.parse((RESULTS / "run_qwen_generation_no_tensor_recording.py").read_text())
        self.assertEqual(ast.dump(ast.Module(body=[function_node()], type_ignores=[]),
                                  include_attributes=False), ast.dump(live, include_attributes=False))

    def test_public_preview_outputs_do_not_contain_course_text(self):
        notebook = json.loads(NOTEBOOK.read_text())
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                text = "".join(output.get("text", []))
                self.assertNotIn("Sample text:", text)
                self.assertNotIn("page 1 preview:", text)
                self.assertFalse(any(k.startswith("image/") for k in output.get("data", {})))


if __name__ == "__main__":
    unittest.main()
