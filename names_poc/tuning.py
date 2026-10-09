"""Ordinary architecture tuning with the same frozen evaluator."""
from pathlib import Path

from filelock import FileLock

from .evaluation import evaluate, START, END
from .storage import verify_frozen, write_json


def architecture(seed: str, params: dict) -> str:
    before, rest = seed.split(START)
    _, after = rest.split(END)
    block = f'''
class TunedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding = {params['embedding']}
        width = {params['width']}
        self.embedding = nn.Embedding(config["vocab_size"], embedding)
        layers = []
        size = config["context_length"] * embedding
        for _ in range({params['depth']}):
            layers.extend([nn.Linear(size, width), nn.{params['activation']}(), nn.Dropout({params['dropout']})])
            size = width
        layers.append(nn.Linear(size, config["vocab_size"]))
        self.layers = nn.Sequential(*layers)
    def forward(self, context):
        return self.layers(self.embedding(context).flatten(1))

def build_model(config) -> nn.Module:
    return TunedMLP(config)
'''
    return before + START + block + END + after


def tune(root: Path, trials: int = 50) -> dict:
    import optuna
    if not 1 <= trials <= 50:
        raise ValueError("trials must be between 1 and 50")
    frozen = verify_frozen(root)
    if (root / "artifacts/final_selection.json").exists():
        raise ValueError("Final selection is frozen; tuning cannot continue")
    directory = root / "artifacts/tuning"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "study.lock"), timeout=0):
        study = optuna.create_study(study_name=frozen["id"], direction="maximize",
                                   storage=f"sqlite:///{(directory / 'study.db').as_posix()}", load_if_exists=True,
                                   sampler=optuna.samplers.TPESampler(seed=3407))
        # An interrupted running trial is retrained/resolved with the same params.
        def execute(trial):
            params = {
                "embedding": trial.suggest_categorical("embedding", [16, 32, 64]),
                "width": trial.suggest_categorical("width", [64, 128, 256]),
                "depth": trial.suggest_categorical("depth", [1, 2, 3]),
                "activation": trial.suggest_categorical("activation", ["Tanh", "ReLU", "GELU"]),
                "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
            }
            source = directory / f"trial_{trial.number:03d}.py"
            source.write_text(architecture((root / "experiment/initial_program.py").read_text(encoding="utf-8"), params), encoding="utf-8")
            result = evaluate(root, source, phase="tuning")
            trial.set_user_attr("source", str(source))
            trial.set_user_attr("evaluation_id", result["evaluation_id"])
            study.tell(trial, result["score"])
        for prior in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            execute(optuna.trial.Trial(study, prior._trial_id))
        for _ in range(max(0, trials-len(study.trials))):
            # Reconstruct the sampler deterministically for this trial number so
            # a restarted process follows the same proposal sequence.
            study.sampler = optuna.samplers.TPESampler(seed=3407+len(study.trials))
            execute(study.ask())
        best = study.best_trial
        if best.value <= 0:
            raise ValueError("All tuning trials failed")
        result = {"trial": best.number, "score": best.value, "params": best.params,
                  "source": best.user_attrs["source"], "trials": len(study.trials), "frozen_id": frozen["id"]}
        write_json(root / "artifacts/tuning_best.json", result)
        return result
