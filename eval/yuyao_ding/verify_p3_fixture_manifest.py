"""Verify the exact local bytes used by the frozen Part 3 evaluation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((EVAL / "p3_fixture_manifest.json").read_text())
    plans = [
        json.loads((EVAL / name).read_text())
        for name in ("p3_scenarios.json", "p3_red_team.json")
    ]
    expected_cases = {
        case["id"]: case["fixture"]
        for plan in plans
        for case in plan["cases"]
    }
    recorded_cases = {}
    errors = []
    pending = []

    for item in manifest["files"]:
        path = ROOT / item["path"]
        for case_id in item["case_ids"]:
            if case_id in recorded_cases:
                errors.append(f"Duplicate case in manifest: {case_id}")
            recorded_cases[case_id] = item["path"]
        if item["status"] != "ready":
            pending.extend(item["case_ids"])
            continue
        if not path.is_file():
            errors.append(f"Missing ready fixture: {path}")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"] or len(data) != item["size_bytes"]:
            errors.append(f"Hash or size mismatch: {path}")

    if recorded_cases != expected_cases:
        errors.append("Manifest case-to-fixture mapping differs from the test plans")
    if args.require_complete and pending:
        errors.append(f"Pending cases: {', '.join(sorted(pending))}")

    print(f"Verified ready files: {sum(i['status'] == 'ready' for i in manifest['files'])}")
    print(f"Pending cases: {', '.join(sorted(pending)) or 'none'}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
