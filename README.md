# Names-model architecture evolution

Start with [steps.md](steps.md) to run the official `ae` commands individually.
The project is a CPU proof of concept inspired by
[karpathy/makemore](https://github.com/karpathy/makemore), using the official
[Google Cloud AlphaEvolve CLI](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud).
It does not reimplement the managed AlphaEvolve service.

## What is optimized

A small PyTorch model predicts the next character from eight preceding tokens.
AlphaEvolve proposes architecture changes; the local evaluator trains weights
with AdamW and returns `exp(-mean validation NLL)` over two training seeds.
The model never receives future characters or target labels. A local names file
can be imported, or the SSA 2025 corpus can be downloaded. Names are deduplicated
and split before any search. Generated names are not
added to the training corpus.

`config.json` contains the experiment constants. Once the baseline is trained,
model scaffold, package code, dependency lock, data, and configuration are
hashed in `artifacts/frozen.json`. Use a fresh project copy for a new protocol.

## Commands

```powershell
uv sync --locked
uv run python -m names_poc prepare --input-file ./data.txt
uv run python -m names_poc baseline
uv run python -m names_poc tune --trials 50
uv run python -m names_poc evolve --max-programs 50
uv run python -m names_poc finalize
uv run python -m names_poc report
```

Configure/authenticate the cloud profile first as described in `steps.md`.
The supplied `data.txt` produces 1,918 unique lowercase names. Its publication
year is not asserted. Use `prepare --ssa-year 2025` only for a separate study
using the full SSA archive.
`evolve` starts billable cloud generation. Local commands do not call model APIs.
The `--trials` argument is a total budget; repeated invocations do not add 50
more trials. `evolve` reuses the saved experiment when called again.

## Artifacts

- `data/manifest.json`: source ZIP/member hashes, normalization counts, split hashes.
- `artifacts/baseline.json`: score, latency cap, calibrated timeout.
- `artifacts/evaluations/<hash>/`: candidate source, request, logs, results, and per-seed weights; completed runs are cached.
- `artifacts/run.json`: canonical remote experiment identity; shared with manual runs through `attach`.
- `artifacts/candidates.json`: exported candidate identifiers, source hashes, and remote scores.
- `artifacts/final_selection.json`: validation-only selection, frozen before reading test data.
- `artifacts/final/`: candidate source, per-seed checkpoints, 1,000 samples per comparator, and evolved-vs-seed diff.
- `artifacts/report.md`: metrics, failures, runtime, sampling diagnostics, and acceptance outcome.

Checkpoints contain state_dict, model config, vocabulary, seed, and training
steps. To load a trusted exported checkpoint:

```python
from pathlib import Path
import torch
from names_poc.training import load_model

folder = Path("artifacts/final/evolved")
checkpoint = torch.load(folder / "seed_101.pt", map_location="cpu", weights_only=True)
model, _ = load_model(folder / "initial_program.py", checkpoint["config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

## Evaluation and limits

The final selection uses only validation. Then a locked evolved finalist,
baseline, and Optuna comparator are measured on test across five fresh seeds.
Passing means at least 1% lower mean test NLL and improvement in at least four
of five seeds, with all model limits satisfied. The report distinguishes
beating the seed from beating ordinary tuning; no improvement is guaranteed.

SSA frequency counts are intentionally ignored: this measures generalization
to unseen spellings, not modeling population frequencies. The frozen alphabet
is derived from all normalized names, without using test targets for fitting.
Latency is a local CPU microbenchmark, not a production SLA measurement.

Local subprocess execution and syntax checks are **not an adversarial sandbox**.
Only public data is used here. The code/score separation does not establish the
PDF's production VPC privacy guarantees. A production deployment needs an
isolated evaluator and its own access controls. This is also not evidence of
ETA-model improvement.

Exact dependency versions are in `uv.lock`; upstream commits are in
`upstream.json`. Run tests with `uv run pytest`. No credentials are stored in
the project; `ae` uses its profile and Google application-default credentials.
