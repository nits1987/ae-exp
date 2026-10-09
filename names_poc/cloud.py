"""Thin orchestration of the pinned official CLI; no replacement AlphaEvolve API."""
from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from .storage import config, digest, read_json, code_sha256, verify_frozen, write_json


def parse_json_output(output: str):
    # Some official commands print human status lines before their JSON result.
    decoder = json.JSONDecoder()
    for offset, character in enumerate(output):
        if character not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(output[offset:])
            if isinstance(value, (dict, list)) and not output[offset+end:].strip():
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("ae did not return a JSON object/list; inspect artifacts/ae_logs")


def ae(root: Path, *args: str, json_output=True):
    command = [sys.executable, "-c", "from ae_cli.cli import app; app()"]
    if json_output:
        command.append("--json")
    command.extend(str(a) for a in args)
    directory = root / "artifacts/ae_logs"
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{time.time_ns()}.txt"
    environment = dict(os.environ, NAMES_POC_ROOT=str(root), PYTHONUTF8="1", NO_COLOR="1",
                       MPLCONFIGDIR=str(root / "artifacts/matplotlib"))
    if not json_output:
        result = subprocess.run(command, cwd=root, env=environment, check=False)
        log.write_text(f"Arguments: {args}\nExit: {result.returncode}\n", encoding="utf-8")
        if result.returncode:
            raise ValueError(f"ae exited {result.returncode}; rerun against the same experiment to reconnect")
        return None
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", check=False)
    log.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"ae {' '.join(args[:2])} failed; see {log}: {result.stderr[-1000:]}")
    return parse_json_output(result.stdout)


def state_of(description: dict) -> str:
    return description.get("state", "").removeprefix("EXPERIMENT_STATE_")


def attach(root: Path, experiment: str) -> dict:
    frozen = verify_frozen(root)
    description = ae(root, "experiment", "describe", experiment)
    name = description["name"]
    settings = description.get("config", {}).get("runSettings", {})
    budget = int(settings.get("maxPrograms", 0))
    if not 1 <= budget <= 50:
        raise ValueError("Experiment must have a generation cap between 1 and 50")
    path = root / "artifacts/run.json"
    prior = read_json(path) if path.exists() else {}
    if prior.get("experiment") and prior["experiment"] != name:
        raise ValueError("This project is already attached to another experiment; use a fresh project")
    if prior.get("frozen_id") and prior["frozen_id"] != frozen["id"]:
        raise ValueError("Experiment fingerprint differs")
    manifest = {**prior, "experiment": name, "nickname": description.get("nickname", experiment),
                "frozen_id": frozen["id"], "max_programs": budget, "state": state_of(description)}
    manifest.pop("create_pending", None)
    write_json(path, manifest)
    write_json(root / "artifacts/experiment.json", description)
    return manifest


def evolve(root: Path, max_programs: int, experiment: str | None = None):
    if not 1 <= max_programs <= 50:
        raise ValueError("max-programs must be 1..50, including failed generations")
    frozen = verify_frozen(root)
    baseline = read_json(root / "artifacts/baseline.json")
    path = root / "artifacts/run.json"
    with FileLock(str(root / "artifacts/cloud.lock"), timeout=0):
        if experiment:
            manifest = attach(root, experiment)
        elif path.exists():
            manifest = read_json(path)
            if manifest.get("create_pending"):
                raise ValueError("Creation outcome is uncertain. Use ae experiment list, then attach the existing experiment; do not create another")
            manifest = attach(root, manifest["experiment"])
        else:
            # ae 0.2.8 prints a human connectivity report even with --json.
            ae(root, "config", "test", json_output=False)
            # Persist intent before a potentially successful-but-disconnected API call.
            write_json(path, {"create_pending": True, "frozen_id": frozen["id"], "max_programs": max_programs})
            created = ae(root, "experiment", "create", "--max-programs", str(max_programs), "--concurrency", "1",
                         "--problem-file", str(root / "problem_description.md"), "--title", "makemore names architecture search",
                         "--models", config(root)["generation_model"])
            manifest = attach(root, created["name"])
        name = manifest["experiment"]
        state = manifest["state"]
        if state in {"CREATED", "INITIALIZED"}:
            # Recover if a prior CLI start uploaded the seed but lost its connection
            # before activating generation. Do not upload a duplicate seed.
            from ae_cli import client as client_module, config as ae_config
            client = client_module.AlphaEvolveClient(ae_config.load_config())
            programs = list(client.list_programs(name))
            if programs:
                seed = (root / "experiment/initial_program.py").read_text(encoding="utf-8")
                if len(programs) != 1 or candidate_code(programs[0]) != seed:
                    raise ValueError("Unexpected existing programs in an unstarted experiment")
                client.start_experiment(name)
            else:
                ae(root, "experiment", "start", name, "--program-dir", str(root / "experiment"), "--score", str(baseline["score"]))
        elif state == "PAUSED":
            ae(root, "experiment", "resume", name)
        elif state in {"COMPLETED", "SUCCEEDED", "FAILED", "CANCELLED"}:
            return export(root, name)
        elif state not in {"ACTIVE", "RUNNING", "STARTING"}:
            raise ValueError(f"Unexpected experiment state {state}; inspect with ae experiment describe")
        ae(root, "experiment", "run", name, "--evaluator", str(root / "evaluator.py"),
           "--backend", "local", "--timeout", str(baseline["timeout_seconds"]),
           "--dashboard", str(root / "artifacts/dashboard.md"), json_output=False)
        return export(root, name)


def candidate_code(program: dict) -> str:
    content = program.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = ast.literal_eval(content)
    files = content.get("files", [])
    if len(files) != 1 or files[0].get("path") != "initial_program.py":
        raise ValueError("Expected exactly one candidate file: initial_program.py")
    return files[0]["content"]


def program_score(program: dict) -> float | None:
    scores = program.get("evaluation", {}).get("scores", {}).get("scores", [])
    score = next((s.get("score") for s in scores if s.get("metric") == "score"), None)
    if score is None:
        return None
    number = float(score)
    return number if math.isfinite(number) else None


def export(root: Path, experiment: str | None = None) -> dict:
    manifest = attach(root, experiment or read_json(root / "artifacts/run.json")["experiment"])
    programs = ae(root, "program", "list", manifest["experiment"])
    write_json(root / "artifacts/cloud_programs.json", programs)
    rows = []
    for program in programs:
        directory = root / "artifacts/candidates" / digest(program["name"])[:20]
        directory.mkdir(parents=True, exist_ok=True)
        # Fetch full content when list responses omit it.
        if not program.get("content"):
            program = ae(root, "program", "show", program["name"])
        write_json(directory / "program.json", program)
        try:
            code = candidate_code(program)
            source = directory / "initial_program.py"
            source.write_text(code, encoding="utf-8", newline="\n")
            rows.append({"name": program["name"], "nickname": program.get("nickname"),
                         "score": program_score(program), "source": str(source), "source_sha256": code_sha256(source)})
        except (ValueError, KeyError) as exc:
            rows.append({"name": program["name"], "score": None, "error": str(exc)})
    rows.sort(key=lambda row: row["score"] if row["score"] is not None else -math.inf, reverse=True)
    result = {"experiment": manifest["experiment"], "state": manifest["state"], "programs": rows}
    write_json(root / "artifacts/candidates.json", result)
    return result
