import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from names_poc.evaluation import evaluate, validate_source, START, END


def candidate(root, body, filename="bad.py"):
    seed = (root / "experiment/initial_program.py").read_text(encoding="utf-8")
    prefix, rest = seed.split(START)
    _, suffix = rest.split(END)
    target = root / filename
    target.write_text(prefix + START + "\n" + body + "\n" + END + suffix, encoding="utf-8")
    return target


def test_repeatability_and_cache(frozen_project):
    source = frozen_project / "experiment/initial_program.py"
    first = evaluate(frozen_project, source)
    assert first["status"] == "ok", first
    again = evaluate(frozen_project, source)
    assert again["cached"] is True
    assert again["evaluation_id"] == first["evaluation_id"]
    crlf_source = frozen_project / "same_code_crlf.py"
    crlf_source.write_bytes(source.read_text(encoding="utf-8").replace("\n", "\r\n").encode())
    assert evaluate(frozen_project, crlf_source)["evaluation_id"] == first["evaluation_id"]
    independent = evaluate(frozen_project, source, phase="repeatability")
    assert independent["status"] == "ok", independent
    assert independent["validation_nll"] == pytest.approx(first["validation_nll"], abs=1e-10)


def test_failed_candidates_and_timeout(frozen_project):
    oversized = candidate(frozen_project, 'def build_model(config):\n    return nn.Linear(8, 30000)')
    result = evaluate(frozen_project, oversized)
    assert result["status"] == "failed" and result["score"] < 0
    assert "Parameter count" in result["error"]
    infinite = candidate(frozen_project, 'def build_model(config):\n    while True:\n        pass', "infinite.py")
    result = evaluate(frozen_project, infinite, timeout=3)
    assert result["status"] == "failed" and "timed out" in result["error"]
    assert (Path(result["directory"]) / "initial_program.py").exists()
    assert evaluate(frozen_project, infinite, timeout=3)["cached"]


def test_scope_violations(project):
    seed = (project / "experiment/initial_program.py").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        validate_source(seed + "print('changed')", seed)
    with pytest.raises(ValueError, match="imports"):
        validate_source(seed.replace(START, START + "\nimport os"), seed)


def test_official_cli_runs_evaluator_from_temporary_directory(frozen_project):
    environment = dict(os.environ, NAMES_POC_ROOT=str(frozen_project), PYTHONUTF8="1",
                       MPLCONFIGDIR=str(frozen_project / "artifacts/matplotlib"))
    command = [sys.executable, "-c", "from ae_cli.cli import app; app()", "--json", "program", "evaluate",
               "--program-dir", str(frozen_project / "experiment"), "--evaluator", str(frozen_project / "evaluator.py"),
               "--backend", "local", "--timeout", "180"]
    result = subprocess.run(command, cwd=frozen_project, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=190)
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert output["success"] and output["score"] > 0
    assert list((frozen_project / "artifacts/evaluations").glob("*/result.json"))
