"""Check the exact official CLI surfaces used in steps.md without cloud calls."""
import pytest
from typer.testing import CliRunner


@pytest.mark.parametrize("command, flags", [
    (["config"], ["--project", "--engine", "--models"]),
    (["config", "switch"], ["profile"]),
    (["config", "test"], []),
    (["config", "discover"], []),
    (["experiment", "create"], ["--max-programs", "--concurrency", "--problem-file", "--models"]),
    (["experiment", "start"], ["--program-dir", "--score"]),
    (["experiment", "run"], ["--evaluator", "--backend", "--timeout", "--dashboard"]),
    (["experiment", "resume"], []),
    (["experiment", "describe"], []),
    (["program", "evaluate"], ["--program-dir", "--evaluator", "--timeout"]),
    (["program", "show"], ["--output-file", "--experiment"]),
    (["program", "diff"], ["--experiment"]),
    (["results", "best"], ["--top"]),
])
def test_official_help_contract(command, flags, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    from ae_cli.cli import app
    result = CliRunner().invoke(app, [*command, "--help"], terminal_width=160)
    assert result.exit_code == 0, result.output
    for flag in flags:
        assert flag in result.output
