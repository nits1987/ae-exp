"""Validation-only selection followed by a single locked test comparison."""
from __future__ import annotations

import difflib
import shutil
from pathlib import Path

from filelock import FileLock

from . import cloud
from .evaluation import evaluate
from .storage import config, read_json, sha256, code_sha256, verify_frozen, write_json


def finalize(root: Path, experiment: str | None = None) -> dict:
    verify_frozen(root)
    cfg = config(root)
    selection_path = root / "artifacts/final_selection.json"
    with FileLock(str(root / "artifacts/finalize.lock"), timeout=0):
        if selection_path.exists():
            selection = read_json(selection_path)
            if experiment:
                manifest = cloud.attach(root, experiment)
                if manifest["experiment"] != selection["experiment"]:
                    raise ValueError("Final selection belongs to another experiment")
        else:
            if not (root / "artifacts/tuning_best.json").exists():
                raise ValueError("Run tune before finalize")
            exported = cloud.export(root, experiment)
            if exported["state"] not in {"COMPLETED", "SUCCEEDED", "FAILED", "CANCELLED"}:
                raise ValueError("Finish or terminate the cloud search before final selection")
            seed_source = root / "experiment/initial_program.py"
            seen = {code_sha256(seed_source)}
            candidates = []
            for row in exported["programs"]:
                if row["score"] is not None and row["score"] > 0 and row.get("source_sha256") not in seen:
                    candidates.append(row)
                    seen.add(row["source_sha256"])
                if len(candidates) == 3:
                    break
            if not candidates:
                raise ValueError("No valid evolved architecture distinct from the seed; report the search without claiming improvement")
            timeout = read_json(root / "artifacts/baseline.json")["timeout_seconds"]
            # Longer final training: same allowance for all models, includes 5 seeds.
            final_timeout = timeout * (cfg["final_steps"]/cfg["search_steps"]) * (len(cfg["final_seeds"])/len(cfg["search_seeds"]))
            validations = []
            for row in candidates:
                result = evaluate(root, Path(row["source"]), phase="final_validation", steps=cfg["final_steps"],
                                  seeds=cfg["final_seeds"], timeout=final_timeout)
                validations.append({"program": row["name"], "result": result})
            write_json(root / "artifacts/final_validation_candidates.json", validations)
            eligible = [row for row in validations if row["result"]["status"] == "ok"]
            if not eligible:
                raise ValueError("All finalists failed constraints during full training; retain baseline")
            chosen = min(eligible, key=lambda row: row["result"]["validation_nll"])
            best_tuning = read_json(root / "artifacts/tuning_best.json")
            sources = {"baseline": seed_source, "optuna": Path(best_tuning["source"])}
            models = {"evolved": {"program": chosen["program"], "directory": chosen["result"]["directory"]}}
            for name, source in sources.items():
                result = evaluate(root, source, phase="final_validation", steps=cfg["final_steps"], seeds=cfg["final_seeds"], timeout=final_timeout)
                if result["status"] != "ok":
                    raise ValueError(f"{name} failed final validation: {result.get('error')}")
                models[name] = {"directory": result["directory"]}
            for model in models.values():
                directory = Path(model["directory"])
                model["source_sha256"] = code_sha256(directory / "initial_program.py")
                model["result_sha256"] = sha256(directory / "result.json")
                model["checkpoint_hashes"] = read_json(directory / "result.json")["checkpoint_hashes"]
            selection = {"experiment": exported["experiment"], "frozen_id": verify_frozen(root)["id"],
                         "selection_metric": "mean validation NLL", "models": models,
                         "final_timeout": final_timeout, "tuning_trials": best_tuning["trials"],
                         "evaluated_evolved_finalists": len(candidates)}
            # This file is written BEFORE any test data is loaded.
            write_json(selection_path, selection)
        comparisons = {}
        for name, model in selection["models"].items():
            directory = Path(model["directory"])
            if code_sha256(directory / "initial_program.py") != model["source_sha256"] or sha256(directory / "result.json") != model["result_sha256"]:
                raise ValueError("Frozen finalist source/results changed")
            for filename, expected in model["checkpoint_hashes"].items():
                if sha256(directory / filename) != expected:
                    raise ValueError("Frozen finalist weights changed")
            result = evaluate(root, directory / "initial_program.py", phase="test", steps=cfg["final_steps"],
                              seeds=cfg["final_seeds"], timeout=selection["final_timeout"], training_directory=directory)
            if result["status"] != "ok":
                raise ValueError(f"Test evaluation for {name} failed: {result.get('error')}")
            comparisons[name] = result
            destination = root / "artifacts/final" / name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory / "initial_program.py", destination / "initial_program.py")
            for checkpoint in directory.glob("seed_*.pt"):
                shutil.copy2(checkpoint, destination / checkpoint.name)
            write_json(destination / "samples.json", result["samples"])
            (destination / "samples.txt").write_text("\n".join(result["samples"]["names"]) + "\n", encoding="utf-8")
        seed = (root / "experiment/initial_program.py").read_text(encoding="utf-8")
        evolved = (root / "artifacts/final/evolved/initial_program.py").read_text(encoding="utf-8")
        diff = "".join(difflib.unified_diff(seed.splitlines(True), evolved.splitlines(True), fromfile="baseline/initial_program.py", tofile="evolved/initial_program.py"))
        (root / "artifacts/final/evolved_vs_baseline.diff").write_text(diff, encoding="utf-8")
        result = {"selection": selection, "models": comparisons}
        write_json(root / "artifacts/final_results.json", result)
        return result
