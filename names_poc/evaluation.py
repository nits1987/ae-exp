"""Persistent, timeout-bounded evaluation shared by manual and automated runs."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from .storage import config, digest, fingerprint, read_json, sha256, code_sha256, verify_frozen, write_json

FAILURE_SCORE = -1e12
START, END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"


def validate_source(source: str, seed: str) -> None:
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError("Exactly one evolution block is required")
    before, rest = source.split(START)
    _, after = rest.split(END)
    seed_before, seed_rest = seed.split(START)
    _, seed_after = seed_rest.split(END)
    if (before, after) != (seed_before, seed_after):
        raise ValueError("Candidate changed code outside the evolution block")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(n.name.split('.')[0] not in {"torch", "math"} for n in node.names):
            raise ValueError("Candidate imports are limited to torch and math")
        if isinstance(node, ast.ImportFrom) and (node.module or "").split('.')[0] not in {"torch", "math"}:
            raise ValueError("Candidate imports are limited to torch and math")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "exec", "eval", "compile", "__import__"}:
            raise ValueError(f"Candidate may not call {node.func.id}")
    # These checks catch accidental scope violations; they are not a security sandbox.


def evaluate(root: Path, source: Path, *, phase="search", steps=None, seeds=None,
             timeout=None, latency_cap_ms=None, enforce_frozen=True, training_directory=None) -> dict:
    cfg = config(root)
    identity = verify_frozen(root) if enforce_frozen else fingerprint(root)
    baseline_path = root / "artifacts/baseline.json"
    baseline = read_json(baseline_path) if baseline_path.exists() else None
    if phase != "baseline" and baseline is not None:
        latency_cap_ms = baseline["latency_cap_ms"] if latency_cap_ms is None else latency_cap_ms
        timeout = baseline["timeout_seconds"] if timeout is None else timeout
    steps = cfg["search_steps"] if steps is None else steps
    seeds = cfg["search_seeds"] if seeds is None else seeds
    request = {"root": str(root), "phase": phase, "config": cfg, "steps": steps,
               "seeds": seeds, "source_sha256": code_sha256(source), "frozen_id": identity["id"],
               "latency_cap_ms": latency_cap_ms}
    if training_directory is not None:
        request["training_directory"] = str(training_directory)
        request["training_result_sha256"] = sha256(training_directory / "result.json")
        request["checkpoint_hashes"] = {p.name: sha256(p) for p in sorted(training_directory.glob("seed_*.pt"))}
    key = digest(request)
    directory = root / "artifacts/evaluations" / key
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "evaluation.lock")):
        if (directory / "result.json").exists():
            cached = read_json(directory / "result.json")
            if cached.get("source_sha256") != code_sha256(directory / "initial_program.py"):
                raise ValueError("Cached candidate source changed")
            for filename, expected in cached.get("checkpoint_hashes", {}).items():
                if sha256(directory / filename) != expected:
                    raise ValueError("Cached checkpoint changed")
            return dict(cached, cached=True)
        code = source.read_text(encoding="utf-8")
        (directory / "initial_program.py").write_text(code, encoding="utf-8", newline="\n")
        write_json(directory / "request.json", request)
        started = time.perf_counter()
        try:
            validate_source(code, (root / "experiment/initial_program.py").read_text(encoding="utf-8"))
            environment = dict(os.environ, NAMES_POC_ROOT=str(root), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
            with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
                completed = subprocess.run([sys.executable, "-m", "names_poc.worker", "--request", str(directory / "request.json")],
                                           cwd=root, env=environment, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            if completed.returncode != 0:
                raise ValueError(f"Worker exited {completed.returncode}; see stderr.log")
            result = read_json(directory / "worker_result.json")
        except subprocess.TimeoutExpired:
            result = {"status": "failed", "score": FAILURE_SCORE, "error": f"Evaluation timed out after {timeout}s"}
        except Exception as exc:
            result = {"status": "failed", "score": FAILURE_SCORE, "error": str(exc)}
        result.update({"evaluation_id": key, "source_sha256": code_sha256(source), "phase": phase,
                       "elapsed_seconds": time.perf_counter()-started, "directory": str(directory),
                       "checkpoint_hashes": {p.name: sha256(p) for p in sorted(directory.glob("seed_*.pt"))}})
        if result["status"] != "ok":
            result["insights"] = [{"label": "error", "text": result.get("error", "Evaluation failed")[:1500]}]
        write_json(directory / "result.json", result)
        return result


def baseline(root: Path) -> dict:
    output = root / "artifacts/baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "artifacts/baseline.lock")):
        if output.exists():
            verify_frozen(root)
            return read_json(output)
        if (root / "artifacts/frozen.json").exists():
            verify_frozen(root)
        result = evaluate(root, root / "experiment/initial_program.py", phase="baseline", enforce_frozen=False, timeout=3600)
        if result["status"] != "ok":
            raise ValueError(f"Baseline failed: {result.get('error')}; see {result['directory']}")
        import math
        import torch
        cfg = config(root)
        if torch.cuda.is_available():
            multiplier = float(cfg.get("gpu_latency_multiplier", cfg.get("latency_multiplier", 2.0)))
            min_cap_ms = 0.0
            device = "cuda"
        else:
            multiplier = float(cfg.get("cpu_latency_multiplier", cfg.get("latency_multiplier", 10.0)))
            min_cap_ms = float(cfg.get("cpu_min_latency_cap_ms", 0.6))
            device = "cpu"
        result["device"] = device
        result["latency_multiplier"] = multiplier
        result["latency_cap_ms"] = max(min_cap_ms, max(r["inference_p95_ms"] for r in result["seeds"]) * multiplier)
        result["timeout_seconds"] = max(180, math.ceil(result["elapsed_seconds"] * 3))
        write_json(root / "artifacts/frozen.json", fingerprint(root))
        write_json(output, result)
        return result
