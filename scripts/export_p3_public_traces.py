"""Export redacted Part 3 T4 request traces from the saved private run.

The source archive is read only. The output contains agent inputs, outputs and
timings, but never local file paths, image bytes, or raw model retry payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARCHIVE_SHA256 = "bd70123ea99d06c48f3d8da6d959ec8bcc30da96ff02e7bdbe8c239d52ba99c1"
EXPECTED_SOURCE_COMMIT = "53c92a3351d8c9bc9d4c33b4d42ab43df9057753"
GROUPS = {"with": ("with.json", 15), "without": ("without.json", 15), "red_team": ("red_team.json", 10)}
RETRY_PAYLOAD_KEYS = {"format_attempts", "rewrite_attempts", "model_attempts"}
PRIVATE_PATTERNS = (
    re.compile(r"/(?:content|Users|root|home|tmp|kaggle|mnt|private|var)/|[A-Za-z]:\\|file://", re.IGNORECASE),
    re.compile(r"data:image/|(?:hf_|github_pat_)[A-Za-z0-9_-]{15,}", re.IGNORECASE),
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_member(names: set[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive member ending in {suffix!r}; found {len(matches)}")
    return matches[0]


def scrub(value, *, case_id: str, agent: str, side: str, key: str = ""):
    """Keep evaluated content; replace private paths and omit retry transcripts."""
    if isinstance(value, dict):
        return {
            child_key: scrub(child, case_id=case_id, agent=agent, side=side, key=child_key)
            for child_key, child in value.items()
            if child_key not in RETRY_PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [scrub(item, case_id=case_id, agent=agent, side=side, key=key) for item in value]
    if key.endswith("model_id") and isinstance(value, str) and value.startswith("/"):
        match = re.search(r"/models--([^/]+)/snapshots/[0-9a-f]{40}$", value)
        if not match:
            raise ValueError(f"Unrecognized local model cache path in {case_id}/{agent}/{side}/{key}")
        return match.group(1).replace("--", "/", 1)
    if key in {"source", "input_image"} or (key == "image_path" and not (agent == "generation" and side == "output")):
        if not isinstance(value, str):
            raise ValueError(f"Expected a source image path in {case_id}/{agent}/{side}/{key}")
        return f"private_fixture:{case_id}"
    if key in {"target"} or (agent == "critique" and side == "input" and key == "output") or (agent == "generation" and side == "output" and key == "image_path"):
        if not isinstance(value, str):
            raise ValueError(f"Expected a generated artifact path in {case_id}/{agent}/{side}/{key}")
        return f"generated_artifact:{case_id}"
    if isinstance(value, str) and any(pattern.search(value) for pattern in PRIVATE_PATTERNS):
        raise ValueError(f"Unreviewed sensitive-looking string in {case_id}/{agent}/{side}/{key}")
    return value


def export_case(record: dict, *, mode: str, raw_trace: dict, raw_bytes: bytes, public_case: dict, manifest: dict) -> dict:
    case_id = record["case_id"]
    if raw_trace["run_id"] != record["trace"]["run_id"]:
        raise ValueError(f"Run ID mismatch for {mode}/{case_id}")
    for field in ("request", "status", "agent_events"):
        if raw_trace[field] != record["trace"][field]:
            raise ValueError(f"Raw and evaluation trace mismatch: {mode}/{case_id}/{field}")
    if raw_trace["request"] != public_case["request"]:
        raise ValueError(f"Request differs from frozen public case: {case_id}")
    if Path(raw_trace["input_image"]).name != Path(public_case["fixture"]).name:
        raise ValueError(f"Fixture differs from frozen public case: {case_id}")
    if case_id in {"P3-RT07", "P3-RT08"} and [e["agent"] for e in raw_trace["agent_events"]] != ["planner"]:
        raise ValueError(f"Portrait case needs separate manual privacy review: {case_id}")

    events = []
    for event in raw_trace["agent_events"]:
        if set(event) != {"agent", "input", "status", "output", "elapsed_seconds"}:
            raise ValueError(f"Unexpected agent event fields in {mode}/{case_id}")
        if not isinstance(event["input"], dict) or not isinstance(event["output"], dict):
            raise ValueError(f"Missing agent input or output in {mode}/{case_id}")
        if not isinstance(event["elapsed_seconds"], (int, float)) or event["elapsed_seconds"] < 0:
            raise ValueError(f"Invalid agent timing in {mode}/{case_id}")
        events.append({
            "agent": event["agent"],
            "input": scrub(event["input"], case_id=case_id, agent=event["agent"], side="input"),
            "status": event["status"],
            "output": scrub(event["output"], case_id=case_id, agent=event["agent"], side="output"),
            "elapsed_seconds": event["elapsed_seconds"],
        })
    if not events or events[0]["agent"] != "planner":
        raise ValueError(f"Planner event missing in {mode}/{case_id}")

    result = {
        "case_id": case_id,
        "mode": mode,
        "source_run_id": raw_trace["run_id"],
        "source_trace_sha256": digest(raw_bytes),
        "source_commit": manifest["source_commit"],
        "gpu": manifest["gpu"],
        "seed": manifest["seed"],
        "request": public_case["request"],
        "input_image": f"private_fixture:{case_id}",
        "critique_enabled": raw_trace["critique_enabled"],
        "status": raw_trace["status"],
        "message": scrub(raw_trace["message"], case_id=case_id, agent="workflow", side="output"),
        "elapsed_seconds": raw_trace["elapsed_seconds"],
        "agent_events": events,
        "redaction": {
            "private_file_paths": "replaced with stable case-scoped identifiers",
            "raw_model_retry_payloads": "omitted; processed agent output and available retry counters retained",
            "images": "not included",
        },
    }
    encoded = json.dumps(result, ensure_ascii=False)
    if any(pattern.search(encoded) for pattern in PRIVATE_PATTERNS):
        raise ValueError(f"Private-looking content remains in {mode}/{case_id}")
    return result


def export(archive: Path, output_root: Path) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_hash = digest(archive.read_bytes())
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from the reviewed T4 run")
    if output_root.exists():
        raise FileExistsError(f"Use a fresh output directory: {output_root}")

    public_cases = {}
    for name in ("p3_scenarios.json", "p3_red_team.json"):
        spec = json.loads((ROOT / "eval" / "yuyao_ding" / name).read_text(encoding="utf-8"))
        for case in spec["cases"]:
            if case["id"] in public_cases:
                raise ValueError(f"Duplicate frozen case: {case['id']}")
            public_cases[case["id"]] = case

    outputs: dict[Path, dict] = {}
    counts = {}
    statuses = {}
    seen_run_ids = set()
    with ZipFile(archive) as z:
        names = set(z.namelist())
        if z.testzip() is not None:
            raise ValueError("Archive failed CRC verification")
        manifest = json.loads(z.read(find_member(names, "manifest.json")))
        if manifest.get("status") != "completed" or not manifest.get("actual_t4_verified"):
            raise ValueError("The source run is not a completed actual T4 run")
        if manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT or manifest.get("seed") != 266:
            raise ValueError("Unexpected source commit or seed")
        for relative, expected in manifest["frozen_hashes"].items():
            if digest((ROOT / relative).read_bytes()) != expected:
                raise ValueError(f"Frozen evaluation file changed: {relative}")

        for mode, (group_file, expected_count) in GROUPS.items():
            group = json.loads(z.read(find_member(names, group_file)))
            records = group["records"]
            if len(records) != expected_count or not group.get("execution_complete"):
                raise ValueError(f"Incomplete {mode} evaluation")
            counts[mode] = len(records)
            statuses[mode] = dict(Counter(record["trace"]["status"] for record in records))
            for record in records:
                case_id = record["case_id"]
                run_id = record["trace"]["run_id"]
                if run_id in seen_run_ids or case_id not in public_cases:
                    raise ValueError(f"Duplicate run or unknown case: {mode}/{case_id}")
                seen_run_ids.add(run_id)
                member = find_member(names, f"requests/{run_id}/trace.json")
                raw_bytes = z.read(member)
                raw_trace = json.loads(raw_bytes)
                result = export_case(record, mode=mode, raw_trace=raw_trace, raw_bytes=raw_bytes,
                                     public_case=public_cases[case_id], manifest=manifest)
                relative = Path(mode) / f"{case_id}.json"
                if relative in outputs:
                    raise ValueError(f"Duplicate published path: {relative}")
                outputs[relative] = result

    if len(outputs) != 40 or len(seen_run_ids) != 40:
        raise ValueError("Expected exactly 40 formal request traces")
    summary = {
        "description": "Redacted export of the completed Colab T4 Part 3 evaluation",
        "source_archive_sha256": archive_hash,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "seed": 266,
        "gpu": "Tesla T4",
        "scored": False,
        "human_review": "pending",
        "counts": counts,
        "status_distribution": statuses,
        "published_trace_count": len(outputs),
        "diagnostic_trace_included": False,
        "redaction_policy": "Private file paths replaced; raw model retry payloads and image bytes omitted; processed agent inputs, outputs and timings retained.",
    }
    output_root.mkdir(parents=True)
    for relative, result in outputs.items():
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    summary = export(args.archive, args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
