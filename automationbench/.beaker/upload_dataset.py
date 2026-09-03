"""Convert frozen AutomationBench splits to Beaker JSONL and upload them.

Writes generated files only into an OS temporary directory, then uploads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _rows(split: str, limit: int | None) -> list[dict[str, object]]:
    from automationbench_skills.data import load_split

    samples = load_split(split)
    if limit is not None:
        samples = samples[:limit]
    rows: list[dict[str, object]] = []
    for sample in samples:
        assertions = sample.info.get("assertions") if isinstance(sample.info, dict) else []
        rows.append(
            {
                "id": sample.task_name,
                "input": {"task_name": sample.task_name},
                "expected": {"assertions": assertions or []},
                "metadata": {"domain": sample.domain},
                "group_key": sample.domain,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="automationbench-skills")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args(argv)

    train_rows = _rows("train", args.train_limit)
    test_rows = _rows("test", args.test_limit)
    if not train_rows or not test_rows:
        print("both train and test splits must contain rows", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        dataset_dir = Path(temp_dir)
        for split_name, rows in (("train", train_rows), ("test", test_rows)):
            path = dataset_dir / f"{split_name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

        command = [
            "beaker",
            "dataset",
            "upload",
            str(dataset_dir),
            "--name",
            args.name,
            "--total-count",
            str(len(train_rows) + len(test_rows)),
            "--split",
            f"train={len(train_rows)}",
            "--split",
            f"test={len(test_rows)}",
            "--json",
        ]
        if args.agent:
            command.extend(["--agent", args.agent])
        upload = subprocess.run(command, check=True, capture_output=True, text=True)
        print(upload.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
