"""Official ae evaluator entry point; safe to copy into an ae temporary workspace.

Run through the project's environment: uv run ae ...
The installed names_poc package resolves data/artifacts independently of CWD.
"""
import argparse
from pathlib import Path

from names_poc.evaluation import evaluate, FAILURE_SCORE
from names_poc.storage import read_json, root_path, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    root = root_path()
    try:
        baseline = read_json(root / "artifacts/baseline.json")
        # Leave time for recording diagnostics before ae's outer timeout fires.
        result = evaluate(root, args.program_dir / "initial_program.py", timeout=baseline["timeout_seconds"] * .9)
        payload = {"score": result["score"], "insights": result["insights"]}
    except Exception as exc:
        payload = {"score": FAILURE_SCORE, "insights": [{"label": "error", "text": str(exc)[:1500]}]}
    write_json(args.output_file, payload)


if __name__ == "__main__":
    main()
