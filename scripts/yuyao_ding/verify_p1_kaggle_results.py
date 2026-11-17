"""Verify saved competition records on CPU; never load a model or score answers."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


SUBMISSION_COLUMNS = [
    "id", "conversation_id", "question_id", "answer",
    "figure_references", "citations", "status",
]


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def true(value):
    return str(value).lower() == "true"


def verify(directory, questions=None):
    directory = Path(directory)
    hash_path = directory / "artifact_hashes.json"
    if hash_path.is_file():
        hashes = json.loads(hash_path.read_text())["files"]
        for name, expected in hashes.items():
            require(Path(name).name == name, "Invalid artifact filename.")
            require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected,
                    f"Original artifact hash differs: {name}")
    columns, rows = read_csv(directory / "submission.csv")
    _, diagnostics = read_csv(directory / "p1_run_diagnostics.csv")
    audit = json.loads((directory / "p1_run_audit.json").read_text())
    provenance = json.loads((directory / "kaggle_run_provenance.json").read_text())
    generations = [
        json.loads(line)
        for line in (directory / "generation_diagnostics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(columns == SUBMISSION_COLUMNS, "Submission columns differ.")
    require(bool(rows), "Submission is empty.")
    require(len(rows) == len(diagnostics) == audit["row_count"] == provenance["question_rows"],
            "Row counts differ.")
    for key in ("id", "question_id"):
        require(len({r[key] for r in rows}) == len(rows), f"Duplicate {key}.")
        require([r[key] for r in rows] == [d[key] for d in diagnostics],
                f"Diagnostic {key} order differs.")
    require(len({r["conversation_id"] for r in rows}) == audit["conversation_count"],
            "Conversation count differs.")
    require(dict(Counter(r["status"] for r in rows)) == audit["status_distribution"],
            "Status distribution differs.")
    allowed = {"answered", "not_found", "conflict", "false_premise", "session_override"}
    for row, diagnostic in zip(rows, diagnostics):
        require(row["status"] in allowed and bool(row["answer"].strip()), "Invalid answer/status.")
        require(len(row["answer"].split()) <= 120, "Answer exceeds the recorded word limit.")
        require(row["conversation_id"] == diagnostic["conversation_id"], "Conversation IDs differ.")
        require(row["status"] == diagnostic["answer_status"], "Diagnostic status differs.")
        require(diagnostic["run_status"] == "completed", "Competition execution error.")
        require(not diagnostic["error_type"] and not diagnostic["error_message"],
                "Diagnostic error fields are not empty.")
        require("DOC3:p7" not in row["citations"].split(","), "Excluded page cited.")
        require("Figure C-1" not in row["figure_references"] + row["citations"],
                "Excluded figure referenced.")
        if row["status"] == "not_found":
            require(row["citations"] == row["figure_references"] == "NONE",
                    "not_found has sources.")
    require(sum(r["citations"] != "NONE" for r in rows) == audit["rows_with_citations"],
            "Citation-presence count differs.")
    require(sum(r["figure_references"] != "NONE" for r in rows) == audit["rows_with_figures"],
            "Figure-presence count differs.")
    for field, key in (("repair_used", "answer_repair_count"),
                       ("answer_fallback_used", "answer_fallback_count"),
                       ("override_repair_used", "override_repair_count")):
        require(sum(true(d[field]) for d in diagnostics) == audit[key], f"{key} differs.")
    require(round(median(float(d["wall_seconds"]) for d in diagnostics), 3)
            == audit["median_seconds_per_question"], "Median request time differs.")
    expected_input = sum(int(d["answer_input_tokens"]) + int(d["override_input_tokens"])
                         for d in diagnostics)
    expected_generated = sum(int(d["answer_generated_tokens"]) + int(d["override_generated_tokens"])
                             for d in diagnostics)
    require(expected_input == audit["total_input_tokens"], "Input-token count differs.")
    require(expected_generated == audit["total_generated_tokens"], "Output-token count differs.")
    model_calls = sum(int(d["answer_generated_tokens"]) > 0 for d in diagnostics)
    model_calls += sum(int(d["override_generated_tokens"]) > 0 for d in diagnostics)
    model_calls += audit["answer_repair_count"] + audit["override_repair_count"]
    competition = generations[-model_calls:] if model_calls else []
    earlier = generations[:-model_calls] if model_calls else generations
    require(all(g["status"] == "completed" for g in competition),
            "Final competition generation includes a failure.")
    require(sum(g["input_tokens"] for g in competition) == expected_input,
            "Competition generation input tokens differ.")
    require(sum(g["generated_tokens"] for g in competition) == expected_generated,
            "Competition generation output tokens differ.")
    require(all(all(length == 1 for length in g["forward_input_lengths"][1:])
                for g in competition), "Cached decode processed multiple prompt tokens.")
    require(provenance["status"] == "completed", "Run is not marked completed.")
    require(provenance["official_validator_returncode"] == 0, "Recorded validator failed.")
    validator = (directory / "official_validator.txt").read_text().strip()
    require(validator == f"valid: {len(rows)} rows", "Recorded validator output differs.")
    require(audit["all_checks_passed"] and all(audit["checks"].values()),
            "Reported Kaggle audit did not pass.")
    require(provenance["golden_evaluation_executed"] is False, "Unexpected golden rerun.")
    require(provenance["used_gpu"] == "cuda:0", "Recorded GPU placement differs.")
    require(provenance["peak_allocated_gib"] <= provenance["peak_reserved_gib"] <= 14.01,
            "Recorded allocation is inconsistent with the allocator budget.")
    if questions is not None:
        questions = Path(questions)
        require(hashlib.sha256(questions.read_bytes()).hexdigest() == provenance["question_file_sha256"],
                "Question-file hash differs.")
        _, question_rows = read_csv(questions)
        require([(r["id"], r["conversation_id"], r["question_id"]) for r in rows]
                == [(q["id"], q["conversation_id"], q["question_id"]) for q in question_rows],
                "Submission differs from original question identifiers/order.")
    return {
        "row_count": len(rows), "conversation_count": audit["conversation_count"],
        "status_distribution": audit["status_distribution"],
        "median_seconds_per_question": audit["median_seconds_per_question"],
        "peak_allocated_gib": provenance["peak_allocated_gib"],
        "peak_reserved_gib": provenance["peak_reserved_gib"],
        "elapsed_seconds_since_setup": provenance["elapsed_seconds_since_setup"],
        "within_90_minutes": provenance["elapsed_seconds_since_setup"] <= 5400,
        "competition_model_calls": model_calls,
        "earlier_completed_probes": sum(g["status"] == "completed" for g in earlier),
        "earlier_failed_probes": sum(g["status"] == "failed" for g in earlier),
        "original_question_file_checked": questions is not None,
        "recorded_official_validator_returncode": 0,
        "gpu_inference_rerun_by_verifier": False,
        "human_correctness_or_citation_support_scored": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--questions", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.questions), indent=2))


if __name__ == "__main__":
    main()
