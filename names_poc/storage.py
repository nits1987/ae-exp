"""Project paths, atomic artifacts, and experiment identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def root_path() -> Path:
    return Path(os.environ.get("NAMES_POC_ROOT", Path(__file__).resolve().parent.parent)).resolve()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha256(path: Path) -> str:
    """ae rewrites files on Windows; LF/CRLF must identify the same program."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config(root: Path) -> dict:
    return read_json(root / "config.json")


def fingerprint(root: Path) -> dict:
    files = [root / "config.json", root / "upstream.json", root / "uv.lock", root / "pyproject.toml", root / ".python-version",
             root / "evaluator.py", root / "experiment/initial_program.py",
             root / "problem_description.md", root / "names.txt"]
    files += sorted((root / "names_poc").glob("*.py"))
    files += sorted((root / "data").glob("*.json"))
    files += sorted((root / "data").glob("*.txt"))
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise ValueError(f"Missing frozen inputs: {missing}. Run uv sync and prepare first.")
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    return {"files": hashes, "id": digest(hashes)}


def verify_frozen(root: Path) -> dict:
    frozen = read_json(root / "artifacts/frozen.json")
    current = fingerprint(root)
    if current != frozen:
        changed = sorted(k for k in set(current["files"]) | set(frozen["files"])
                         if current["files"].get(k) != frozen["files"].get(k))
        raise ValueError(f"Frozen inputs changed: {changed}. Use a fresh project for a different experiment.")
    return frozen
