# makemore + AlphaEvolve: Names Model Proof of Concept

Start with [steps.md](steps.md) to run the official `ae` commands individually.
The project is a CPU proof of concept inspired by
[karpathy/makemore](https://github.com/karpathy/makemore), using the official
[Google Cloud AlphaEvolve CLI](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud).
It does not reimplement the managed AlphaEvolve service.

The goal is to find a small name-generation model that predicts unseen names
better than the starting model and a conventional Optuna tuning baseline.
Model training and evaluation run on this PC's CPU; candidate code generation
runs through Google Cloud AlphaEvolve.

## Dataset

The primary dataset comes from [combined_names.txt](combined_names.txt), which combines
the ranked male/female names in [data.txt](data.txt) (`2,000` entries) with
Andrej Karpathy's [`makemore` `names.txt`](https://github.com/karpathy/makemore)
(`32,033` entries). Preparation lowercases all names, filters to alphabetic
spellings (`a`–`z`), and deduplicates globally before creating deterministic
80/10/10 splits. Both [combined_names.txt](combined_names.txt) and the smaller
[data.txt](data.txt) (`1,918` unique names, used by unit tests) are preserved in
the repository.

| Item | `combined_names.txt` (Primary) | `data.txt` (Small / Unit Tests) |
|---|---:|---:|
| Raw name entries before deduplication | 34,033 | 2,000 |
| Unique alphabetic names in `names.txt` | **29,495** | 1,918 |
| Training names (80%) | **23,596** | 1,534 |
| Validation names (10%) | **2,949** | 191 |
| Test names (10%) | **2,950** | 193 |

The alphabet contains `a` through `z`, plus one special token (`0`) for padding/start
and end-of-name prediction. The model uses a fixed eight-character causal context.
Splits use seed `3407` and have no overlapping names. Source and output checksums
are recorded in `data/manifest.json`. No external dataset download is needed for
this corpus.

## Requirements

- Linux/macOS (`bash`) or Windows (`PowerShell`), Git, and [uv](https://docs.astral.sh/uv/).
- Python 3.13, selected through `.python-version` and managed by uv.
- CPU PyTorch and the official `ae` CLI, installed together by `uv sync --locked`.
- For evolution: Google Cloud CLI, a billing-enabled project, provisioned
  AlphaEvolve/Gemini Enterprise access, an engine ID, and working authentication.

Data preparation, baseline training, Optuna tuning, and local reporting can run
without cloud credentials. Follow [steps.md](steps.md) for provisioning,
authentication, and each official `ae` command in order.

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

Run from the project root. Install dependencies and prepare the combined dataset:

```bash
uv sync --locked
uv run python -m names_poc prepare --input-file ./combined_names.txt
uv run python -m names_poc baseline
```

`baseline` loads `experiment/initial_program.py`, trains the starting network
from scratch for both search seeds, saves its weights, and records the reference
validation score. Running `initial_program.py` directly does not train a model.
Evolution reuses the starting architecture's code and score; each candidate
gets fresh weights and the same training allowance. No external pretrained
checkpoint or inheritance of baseline weights is used.

After configuring and testing cloud access using [steps.md](steps.md), run the
comparison and automated search sequentially:

```powershell
uv run python -m names_poc tune --trials 50
uv run python -m names_poc evolve --max-programs 50
uv run python -m names_poc finalize
uv run python -m names_poc report
```

The combined corpus (`combined_names.txt`) produces 29,495 unique lowercase names
(while `data.txt` produces 1,918 unique names for fast smoke testing). Use
`prepare --ssa-year 2025` only for a separate study using the full SSA archive.
`evolve` starts billable cloud generation. Local commands do not call model APIs.
The `--trials` argument is a total budget; repeated invocations do not add 50
more trials. `evolve` reuses the saved experiment when called again.

For the first run, use the manual workflow in `steps.md` instead of `evolve`.
Record the manually created experiment so both workflows share its identity:

```powershell
uv run python -m names_poc attach --experiment "YOUR_EXPERIMENT_NAME"
```

After the manual search completes, archive its programs and finalize it:

```powershell
uv run python -m names_poc export --experiment "YOUR_EXPERIMENT_NAME"
uv run python -m names_poc tune --trials 50
uv run python -m names_poc finalize --experiment "YOUR_EXPERIMENT_NAME"
uv run python -m names_poc report
```

Do not run tuning and evolutionary evaluations simultaneously: they share the
CPU, and contention would distort latency measurements. Completed evaluations
are cached; interrupted runs can reconnect to the same cloud experiment.

## Experiment settings

| Setting | Current value |
|---|---|
| Seed architecture | Embedding size 32, hidden size 128, `tanh` MLP (`37,243` parameters) |
| Evolved code | Model architecture inside one `EVOLVE-BLOCK` |
| Fixed optimization | AdamW; learning rate 0.001; weight decay 0.01 |
| Search training | 2,000 updates, batch size 128, seeds 3407 and 42 |
| Parameter limit | 200,000 |
| Inference limit | Device-aware batch-one p95 cap: `10x` baseline (`0.60 ms` floor) on CPU (`cpu_latency_multiplier: 10.0`, `cpu_min_latency_cap_ms: 0.6`) to accommodate per-operator Python dispatch overhead; `2x` baseline on GPU (`gpu_latency_multiplier: 2.0`) |
| Diagnostic insights | Reports `val_NLL`, `train_NLL`, `overfit_gap` (`val_NLL - train_NLL`), `params/max_parameters`, and `latency_p95` vs. cap to guide candidate mutations |
| Evaluation timeout | At least 180 seconds; calibrated from the baseline |
| Search allowance | Up to 50 generated programs, including failures |
| Final training | 5,000 updates across fresh seeds 101–105 |

Optimizer/scheduler evolution and a fixed 20-second budget were reviewed as
possible later extensions; they are not part of this experiment.

## Reference Run Results (50 Candidates on `combined_names.txt`)

Running a 50-candidate search with `gemini-3.5-flash` on the 29,495-name corpus (`23,596` train / `2,949` validation / `2,950` test):

- **Completion & Pass Rate:** `50 / 50` evaluated (`35 Succeeded`, `15 Failed` — **70% pass rate**; all 35 valid architectures outperformed the baseline seed).
- **Validation NLL Improvement:** Reduced validation NLL from **`2.1657`** (seed) to **`2.0969`** (`prog-spiked-zebu`, Generation 31) — a **`-3.18%` NLL reduction** (more than $3\times$ the `1%` target threshold), lifting fitness $\exp(-\text{NLL})$ from `0.114668` to **`0.122842`** (`+7.13%`).
- **Token Usage (`gemini-3.5-flash`):** `111,902` input tokens + `997,519` output tokens = **`1,109,421` total tokens** (~22K tokens/candidate).

| Rank | Candidate | Fitness ($\exp(-\text{Val NLL})$) | Val NLL | Train NLL | Overfit Gap | Params | CPU p95 Latency | Key Architectural Mutation |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| **1** | `prog-spiked-zebu` *(Gen 31)* | **0.122842** | **2.0969** *(-3.18%)* | 2.0035 | +0.0934 | 197,951 | 0.2035 ms | 3-block Pre-LN `SwiGLUResBlock` (`hidden=132`), post-flatten `LayerNorm`, fan-in + depth initialization ($\sigma = 1/\sqrt{d \cdot N}$) |
| **2** | `prog-sloppy-binturong` *(Gen 7)* | **0.122498** | **2.0997** *(-3.05%)* | 2.0281 | +0.0716 | 187,599 | 0.3491 ms | Per-token `LayerNorm(emb=64)` + 3-block `GeGLUResBlock` (`hidden=116`) |
| **3** | `prog-pearl-echidna` *(Gen 22)* | **0.122368** | **2.1007** *(-3.00%)* | 2.0210 | +0.0797 | 195,115 | 0.3180 ms | Gated residual MLP with scaled skip projections |
| **4** | `prog-lively-kudu` *(Gen 14)* | **0.122283** | **2.1014** *(-2.97%)* | 2.0183 | +0.0831 | 192,455 | 0.4603 ms | Custom `RMSNorm` + 3-block `SwiGLUResBlock` (`emb=48, hidden=124`) |
| **5** | `prog-agate-oarfish` *(Gen 23)* | **0.122204** | **2.1021** *(-2.94%)* | 2.0244 | +0.0777 | 196,420 | 0.2950 ms | Multi-scale context fusion + gated residual blocks |
| — | **Baseline Seed** (`aromatic-mouflon`) | `0.114668` | `2.1657` *(Baseline)* | `2.1281` | `+0.0376` | `37,243` | `0.0809 ms` | 2-layer `Tanh` MLP (`emb=32, hidden=128`) |

## Project layout

| Path | Purpose |
|---|---|
| [steps.md](steps.md) | Manual first-run instructions |
| [combined_names.txt](combined_names.txt) | Combined 29,495-name dataset (`data.txt` + Karpathy `makemore` `names.txt`) |
| [data.txt](data.txt) | Supplied 1,000-row ranked male/female names table (used by unit tests) |
| [experiment/initial_program.py](experiment/initial_program.py) | Starting model and mutable architecture block |
| [evaluator.py](evaluator.py) | Official `ae` evaluator entry point |
| [names_poc/](names_poc/) | Data preparation, training, caching, cloud orchestration, tuning, and reports |
| [config.json](config.json) | Fixed experiment settings |
| [problem_description.md](problem_description.md) | Instructions and constraints supplied to AlphaEvolve |
| [tests/](tests/) | Dataset, evaluator, recovery, CLI-contract, and finalization checks |
| `data/` | Prepared splits, vocabulary, and dataset manifest |
| `artifacts/` | Persistent evaluations, checkpoints, exports, and reports |

## Artifacts

- `data/manifest.json`: source-file checksum, normalization counts, and split hashes; ZIP/member hashes when using an SSA archive.
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

Rank values and any SSA frequency counts are intentionally ignored: this measures generalization
to unseen spellings, not modeling population frequencies. The frozen alphabet
is derived from all normalized names, without using test targets for fitting.
Latency is a local CPU microbenchmark, not a production SLA measurement.

Local subprocess execution and syntax checks are **not an adversarial sandbox**.
The code/score separation does not establish the
PDF's production VPC privacy guarantees. A production deployment needs an
isolated evaluator and its own access controls. This is also not evidence of
ETA-model improvement.

Exact dependency versions are in `uv.lock`; upstream commits are in
`upstream.json`. Run tests with `uv run pytest`. No credentials are stored in
the project; `ae` uses its profile and Google application-default credentials.
