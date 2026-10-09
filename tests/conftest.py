import shutil
import zipfile
from pathlib import Path

import pytest

from names_poc.data import prepare
from names_poc.storage import read_json, write_json


@pytest.fixture
def project(tmp_path):
    original = Path(__file__).resolve().parents[1]
    root = tmp_path / "project"
    root.mkdir()
    for filename in ("config.json", "upstream.json", "uv.lock", "pyproject.toml", ".python-version", "evaluator.py", "problem_description.md"):
        shutil.copy2(original / filename, root / filename)
    shutil.copytree(original / "names_poc", root / "names_poc", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(original / "experiment", root / "experiment", ignore=shutil.ignore_patterns("__pycache__"))
    cfg = read_json(root / "config.json")
    cfg.update(search_steps=3, final_steps=4, batch_size=16, latency_warmup=2,
               latency_repeats=5, sample_count=12, latency_multiplier=1000.0)
    write_json(root / "config.json", cfg)
    archive = root / "fixture.zip"
    # 100 unique alphabetic spellings plus duplicate and invalid rows.
    words = [f"name{chr(97+i//10)}{chr(97+i%10)}" for i in range(100)]
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("yob2025.txt", "\n".join(f"{word},F,10" for word in words) + "\nNAMEAA,M,5\nnot-a-name,F,8\n")
    prepare(root, 2025, archive)
    return root


@pytest.fixture
def frozen_project(project):
    from names_poc.evaluation import baseline
    baseline(project)
    return project
