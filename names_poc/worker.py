"""One candidate per process; parent owns timeout and persistent results."""
import argparse
from pathlib import Path
import traceback

from .storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    directory = args.request.parent
    try:
        from .training import train, test_checkpoints
        function = test_checkpoints if request["phase"] == "test" else train
        result = function(request, Path(request["root"]), directory)
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "failed", "score": -1e12, "error": str(exc),
                  "insights": [{"label": "error", "text": str(exc)[:1500]}]}
    write_json(directory / "worker_result.json", result)


if __name__ == "__main__":
    main()
