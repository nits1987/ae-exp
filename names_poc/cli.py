"""Commands shared by the manual runbook and repeatable automation."""
import argparse
import json
from pathlib import Path

from .storage import root_path


def main():
    parser = argparse.ArgumentParser(description="Names-model AlphaEvolve proof of concept")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare local or SSA names and freeze deterministic splits")
    prep.add_argument("--ssa-year", type=int, default=2025)
    sources = prep.add_mutually_exclusive_group()
    sources.add_argument("--archive", type=Path, help="Use an already-downloaded SSA national ZIP")
    sources.add_argument("--input-file", type=Path, help="Local ranked male/female TSV or one-name-per-line text")
    commands.add_parser("baseline", help="Train seed, calibrate limits, freeze inputs")
    tuning = commands.add_parser("tune", help="Run/resume ordinary architecture tuning")
    tuning.add_argument("--trials", type=int, default=50, help="Total trial count, not additional trials")
    evolution = commands.add_parser("evolve", help="Start/reconnect official ae create/start/run workflow (billable)")
    evolution.add_argument("--max-programs", type=int, default=50)
    evolution.add_argument("--experiment", help="Attach to an existing manual experiment")
    attach = commands.add_parser("attach", help="Record a manually created experiment")
    attach.add_argument("--experiment", required=True)
    export = commands.add_parser("export", help="Archive all remote candidates and metadata locally")
    export.add_argument("--experiment")
    final = commands.add_parser("finalize", help="Select on validation, freeze, then compare on test")
    final.add_argument("--experiment")
    commands.add_parser("report", help="Write report from local artifacts; never trains or uses cloud")
    args = parser.parse_args()
    root = root_path()
    try:
        if args.command == "prepare":
            from .data import prepare
            result = prepare(root, args.ssa_year, args.archive, args.input_file)
        elif args.command == "baseline":
            from .evaluation import baseline
            result = baseline(root)
        elif args.command == "tune":
            from .tuning import tune
            result = tune(root, args.trials)
        elif args.command in {"evolve", "attach", "export"}:
            from . import cloud
            if args.command == "evolve":
                result = cloud.evolve(root, args.max_programs, args.experiment)
            else:
                result = getattr(cloud, args.command)(root, args.experiment)
        elif args.command == "finalize":
            from .finalization import finalize
            result = finalize(root, args.experiment)
        else:
            from .reporting import report
            result = {"report": str(report(root))}
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Completed evaluations are cached; reconnect to the same experiment.\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
