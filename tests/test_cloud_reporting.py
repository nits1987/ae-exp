from pathlib import Path

import pytest

from names_poc import cloud
from names_poc.reporting import comparison
from names_poc.storage import read_json, write_json


def test_cli_json_with_status_prefix():
    assert cloud.parse_json_output('Starting...\n{"name": "example"}\n') == {"name": "example"}
    assert cloud.parse_json_output('[dim]note[/dim]\n[{"name":"example"}]') == [{"name": "example"}]
    with pytest.raises(ValueError):
        cloud.parse_json_output("no result")


def test_resume_does_not_create_or_start(frozen_project, monkeypatch):
    calls = []
    name = "projects/1/experiments/example"
    def fake_ae(root, *args, **kwargs):
        calls.append(args)
        if args[:2] == ("experiment", "describe"):
            return {"name": name, "state": "RUNNING", "config": {"runSettings": {"maxPrograms": 50}}}
        return None
    monkeypatch.setattr(cloud, "ae", fake_ae)
    monkeypatch.setattr(cloud, "export", lambda root, name: {"experiment": name})
    cloud.attach(frozen_project, name)
    cloud.evolve(frozen_project, 50)
    assert any(args[:2] == ("experiment", "run") for args in calls)
    assert not any(args[:2] in [("experiment", "create"), ("experiment", "start"), ("experiment", "resume")] for args in calls)


def test_ambiguous_creation_requires_recovery(frozen_project, monkeypatch):
    write_json(frozen_project / "artifacts/run.json", {"create_pending": True})
    monkeypatch.setattr(cloud, "ae", lambda *args, **kwargs: pytest.fail("must not call cloud"))
    with pytest.raises(ValueError, match="uncertain"):
        cloud.evolve(frozen_project, 50)


def test_new_run_creates_and_starts_exactly_once(frozen_project, monkeypatch):
    from ae_cli import client as client_module, config as ae_config
    name = "projects/1/experiments/new"
    remote = {"state": "CREATED"}
    calls = []
    def fake_ae(root, *args, **kwargs):
        calls.append(args[:2])
        if args[:2] == ("config", "test"):
            assert kwargs["json_output"] is False
        if args[:2] == ("experiment", "create"):
            return {"name": name}
        if args[:2] == ("experiment", "describe"):
            return {"name": name, "state": remote["state"], "config": {"runSettings": {"maxPrograms": 50}}}
        if args[:2] == ("experiment", "start"):
            remote["state"] = "RUNNING"
            return {"name": "operation-1"}
    class Client:
        def list_programs(self, name):
            return []
    monkeypatch.setattr(ae_config, "load_config", lambda: None)
    monkeypatch.setattr(client_module, "AlphaEvolveClient", lambda cfg: Client())
    monkeypatch.setattr(cloud, "ae", fake_ae)
    monkeypatch.setattr(cloud, "export", lambda root, name: {"experiment": name})
    cloud.evolve(frozen_project, 50)
    cloud.evolve(frozen_project, 50)
    assert calls.count(("experiment", "create")) == 1
    assert calls.count(("experiment", "start")) == 1
    assert calls.count(("experiment", "run")) == 2


def test_success_criteria_require_four_of_five():
    baseline = [{"seed": n, "test_nll": 2.0} for n in range(5)]
    winner = [{"seed": n, "test_nll": 1.9} for n in range(5)]
    assert comparison(winner, baseline)["passes"]
    inconsistent = [{"seed": n, "test_nll": 1.0 if n < 3 else 2.1} for n in range(5)]
    assert not comparison(inconsistent, baseline)["passes"]


def test_manual_export_preserves_sources_and_identifiers(frozen_project, monkeypatch):
    code = (frozen_project / "experiment/initial_program.py").read_text(encoding="utf-8")
    name = "projects/1/experiments/manual"
    program = {"name": name + "/programs/1", "content": {"files": [{"path": "initial_program.py", "content": code}]},
               "evaluation": {"scores": {"scores": [{"metric": "score", "score": .2}]}}}
    def fake_ae(root, *args, **kwargs):
        if args[:2] == ("experiment", "describe"):
            return {"name": name, "state": "COMPLETED", "config": {"runSettings": {"maxPrograms": 50}}}
        return [program]
    monkeypatch.setattr(cloud, "ae", fake_ae)
    exported = cloud.export(frozen_project, name)
    assert exported["programs"][0]["name"] == program["name"]
    assert Path(exported["programs"][0]["source"]).read_text(encoding="utf-8") == code
    assert read_json(frozen_project / "artifacts/run.json")["experiment"] == name
