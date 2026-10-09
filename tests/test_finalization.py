from names_poc import finalization
from names_poc.reporting import report
from names_poc.storage import read_json, sha256
from names_poc.tuning import tune


def test_tuning_final_selection_test_and_resume(frozen_project, monkeypatch):
    root = frozen_project
    tuned = tune(root, trials=2)
    assert tuned["trials"] == 2
    assert tune(root, trials=2)["trials"] == 2  # not two additional trials
    source = root / "evolved.py"
    seed = (root / "experiment/initial_program.py").read_text(encoding="utf-8")
    source.write_text(seed.replace('config["hidden_size"]', '64'), encoding="utf-8")
    exported = {"experiment": "manual-experiment", "state": "COMPLETED", "programs": [
        {"name": "program-1", "source": str(source), "source_sha256": sha256(source), "score": .5}]}
    monkeypatch.setattr(finalization.cloud, "export", lambda *a: exported)
    real_evaluate = finalization.evaluate
    def checked_evaluate(*args, **kwargs):
        if kwargs.get("phase") == "test":
            assert (root / "artifacts/final_selection.json").exists()
            selection = read_json(root / "artifacts/final_selection.json")
            assert set(selection["models"]) == {"baseline", "optuna", "evolved"}
        return real_evaluate(*args, **kwargs)
    monkeypatch.setattr(finalization, "evaluate", checked_evaluate)
    first = finalization.finalize(root)
    assert all(len(result["seeds"]) == 5 for result in first["models"].values())
    assert all(len(result["samples"]["names"]) == 12 for result in first["models"].values())
    monkeypatch.setattr(finalization.cloud, "export", lambda *a: (_ for _ in ()).throw(AssertionError("no re-selection")))
    second = finalization.finalize(root)
    assert all(result["cached"] for result in second["models"].values())
    assert second["selection"] == first["selection"]
    assert (root / "artifacts/final/evolved_vs_baseline.diff").exists()
    output = report(root)
    assert "Individual training seeds" in output.read_text(encoding="utf-8")
    assert (root / "artifacts/comparison.json").exists()
