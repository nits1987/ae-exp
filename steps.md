# First run: execute AlphaEvolve commands one at a time

Run these commands in **PowerShell from this project directory**. Each block is
one command. Stop on an error and resolve it before continuing. Commands below
use Google's official `ae` CLI installed with PyTorch in the same environment.
The project helper commands prepare data, train models, archive artifacts, and
compare results; they do not replace `ae`.

The CLI is pinned to **ae-cli 0.2.8**, commit
`674dd5e558e917320c251c085e47fae33e1bbcff`. Do not upgrade it mid-experiment.

## 1. Install and check the local environment

Prerequisites: Git, uv, and Google Cloud CLI on PATH. Python 3.13 will be selected
by uv using `.python-version`. CPU PyTorch comes from its CPU wheel index.

```powershell
uv sync --locked
```

Expected: a `.venv` and the exact dependencies in `uv.lock`. No GPU is required.

Keep matplotlib's CLI cache in this workspace (useful in restricted Windows
environments, and avoids rebuilding its font cache for every CLI invocation).

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) "artifacts\matplotlib"
```

```powershell
uv run ae version
```

Expected: ae-cli 0.2.8. Use `version`, not `--version`.

```powershell
uv run python -c "import torch; print(torch.__version__); print('CPU threads available:', torch.get_num_threads())"
```

The experiment explicitly sets two CPU threads regardless of this initial value.

```powershell
uv run pytest
```

Expected: all tests pass. Tests use synthetic names and mock cloud responses;
they never start a billable experiment.

## 2. Prepare Google Cloud access

Use a billing-enabled Google Cloud project with AlphaEvolve provisioned and a
Gemini Enterprise app/engine. The engine ID is the app's **ID**, not its display
name. Obtain it from the Gemini Enterprise console. A project alone is not an
AlphaEvolve entitlement. Follow the provisioning link in the
[official repository prerequisites](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud#quickstart-5-minutes)
if the app/service has not been set up. No GPU VM, GKE cluster, or Cloud Run
service is required for this experiment.

Set these two variables to your actual values; do not leave placeholder text.

```powershell
$ProjectId = "YOUR_PROJECT_ID"
```

```powershell
$EngineId = "YOUR_GEMINI_ENTERPRISE_ENGINE_ID"
```

Sign in to the CLI for project discovery and management.

```powershell
gcloud auth login
```

Enable the APIs if they are not already enabled. This requires the relevant
project permissions; ask the project administrator if your account lacks them.

```powershell
gcloud services enable discoveryengine.googleapis.com aiplatform.googleapis.com --project=$ProjectId
```

Set up application-default credentials used by `ae`.

```powershell
gcloud auth application-default login
```

```powershell
gcloud auth application-default set-quota-project $ProjectId
```

Discover available configuration. Discovery can query multiple projects; use
your known IDs in the next command if broad discovery is restricted.

```powershell
uv run ae --json config discover
```

```powershell
uv run ae config --name=names-poc --project=$ProjectId --engine=$EngineId --location=global --models=gemini-3.5-flash
```

Make the profile active explicitly (saving a named profile does not necessarily
switch the active profile).

```powershell
uv run ae config switch names-poc
```

```powershell
uv run ae --json config test
```

Expected: successful connectivity. Do not launch if this fails. The CLI may
print a human-readable success report here even with `--json`; do not pipe this
particular command into `ConvertFrom-Json`. The CLI may
resolve the project ID to its numeric project number through `gcloud`.
An existing `.env` can override the profile; inspect it if the wrong project
appears. Never paste authentication tokens into source files.

## 3. Prepare names and train initial_program.py as the baseline

Use the supplied `data.txt` for this project. It contains ranked male/female
columns; preparation combines both columns, lowercases names, removes duplicates,
and creates deterministic 80/10/10 splits. The source is recorded as your local
file, without assuming a publication year. This command requires no download.

```powershell
uv run python -m names_poc prepare --input-file ./data.txt
```

Expected for the supplied file: 2,000 entries, 1,918 unique names, 82 duplicates
removed; 1,534 training names, 191 validation names, and 193 test names. Rerunning
the same command verifies the prepared data. Do not switch sources afterward.

For a **separate study** using the full SSA corpus, `prepare --ssa-year 2025`
downloads the [SSA national ZIP](https://www.ssa.gov/oact/babynames/limits.html),
reads `yob2025.txt`, and writes `names.txt`, the vocabulary, and frozen splits.
It combines sex categories, ignores counts, and removes duplicate spellings
before splitting. An existing prepared dataset is verified rather than replaced.
For an already downloaded ZIP, use `prepare --ssa-year 2025 --archive C:\path\names.zip`
instead. A missing 2025 member is an error, not a silent fallback to another year.

Review `config.json` before the next command. Do not edit the protocol or code
after the baseline freezes their hashes.

**This is the first training run of the starting network.**
`experiment/initial_program.py` defines the model and `build_model(config)`;
running that file directly only defines those objects and does not train them.
The following command loads that file and runs the fixed training/evaluation
procedure around it:

```powershell
uv run python -m names_poc baseline
```

On the first run, this command:

1. Builds the unchanged network from `experiment/initial_program.py`.
2. Trains it from scratch for 2,000 updates with seed 3407, then independently
   repeats training with seed 42 on the same training split.
3. Measures validation NLL, parameter count, latency, and elapsed evaluation time.
4. Saves both trained checkpoints, `seed_3407.pt` and `seed_42.pt`, in the
   evaluation directory recorded by `artifacts/baseline.json`.
5. Records the reference fitness `exp(-mean validation NLL)` and freezes the
   dataset, code, and training settings for subsequent comparisons.

`artifacts/baseline.json` includes positive `score`, `validation_nll`,
`latency_cap_ms`, `timeout_seconds`, and `directory`. Repeating `baseline` with
unchanged inputs reuses its saved result. Baseline setup has a one-hour emergency timeout. Subsequent
evaluations use the measured timeout, with room for recording failed results.

```powershell
$Baseline = Get-Content -Raw ./artifacts/baseline.json | ConvertFrom-Json
```

Inspect the two trained baseline checkpoints:

```powershell
Get-ChildItem -LiteralPath $Baseline.directory -Filter "seed_*.pt"
```

```powershell
$TimeoutSeconds = [int]$Baseline.timeout_seconds
```

**What evolution reuses:** AlphaEvolve starts from the original architecture's
source code and this measured baseline score. Each proposed architecture is
trained from scratch using the same update budget, training seeds, and batch
sequences. Candidates do **not** inherit the baseline's trained weights. This
is architecture search, not continued fine-tuning of one checkpoint. The saved
weights preserve the baseline for inspection and reuse; no pretrained external
network is loaded by the current implementation.

At finalization, the original architecture and the selected alternatives are
all trained again for 5,000 updates across five fresh seeds. The longer-trained
finalists are therefore compared against an equally trained baseline, rather
than against its shorter initial training run.

Check the official CLI/evaluator connection. It copies the evaluator and seed
into a temporary directory; persistent training artifacts remain in this project.
This is a separate integration check after the baseline has been established;
it does not replace the first baseline training command.

```powershell
uv run ae --json program evaluate --program-dir ./experiment --evaluator ./evaluator.py --backend local --timeout $TimeoutSeconds
```

Expected: `success: true` and a positive `score` close to the baseline. This
evaluation does not start a cloud search. The fitness is **exp(-NLL)**; do not
pass raw NLL as the initial score. If latency fails because this PC is busy,
resolve the measurement issue before launching rather than changing caps during
a run. Run one local evaluation loop at a time; do not run tuning alongside it.

## 4. Create the cloud experiment

The generation cap includes failed candidates. It limits generated programs,
not just successful evaluations. The uploaded seed may appear separately in
program listings. Concurrency is one.

```powershell
uv run ae --json experiment create --max-programs 50 --concurrency 1 --problem-file ./problem_description.md --title "makemore names architecture search" --models gemini-3.5-flash
```

Copy the returned `name` (canonical resource path) or nickname. Prefer the full
`name` so recovery does not depend on local nickname caches.

```powershell
$Experiment = "PASTE_EXPERIMENT_NAME_OR_NICKNAME"
```

Record it locally. This allows `finalize` and the automated wrapper to reuse
your manually created experiment.

```powershell
uv run python -m names_poc attach --experiment $Experiment
```

Expected: `artifacts/run.json` with the canonical experiment name and frozen
protocol identity. Do not create another experiment when reopening a terminal.

## 5. Start cloud generation — billable

**The next command uploads the seed and activates cloud candidate generation.**
Gemini/AlphaEvolve usage is billed to your project. Training stays on this PC.
Only the candidate source is bundled from `experiment`; keep data and private
files outside that directory. Local subprocess execution is not a security
sandbox; this guide is for the public-data proof of concept.

```powershell
uv run ae --json experiment start $Experiment --program-dir ./experiment --score $Baseline.score
```

Expected: a start-operation response. This starts cloud generation; it does not
run candidate training on the PC. Do not repeat `start` on an active experiment.

## 6. Run the local evaluation loop

```powershell
uv run ae experiment run $Experiment --evaluator ./evaluator.py --backend local --timeout $TimeoutSeconds --dashboard ./artifacts/dashboard.md
```

The terminal shows acquired candidates, their scores/failures, and new best
results. The loop trains each new architecture with the fixed seeds, submits
scores and diagnostics, and continues until the remote experiment finishes.
The dashboard updates after evaluations. A candidate failure is expected search
feedback, not a reason to reset the whole experiment.

Persistent results live in `artifacts/evaluations/<hash>`. A result's key includes
candidate code and the frozen protocol. Re-acquiring identical code reuses a
completed result. Interrupted, incomplete work may need to be retrained.

### Reopening or recovering

Ctrl+C stops the local controller. **It does not cancel cloud generation.**
The server-side 50-program cap remains in effect. This pinned CLI has no
`experiment pause` or `experiment stop` command; do not invent those commands.

In a new terminal, restore the experiment variable from its saved manifest.

```powershell
$Experiment = (Get-Content -Raw ./artifacts/run.json | ConvertFrom-Json).experiment
```

```powershell
$TimeoutSeconds = [int](Get-Content -Raw ./artifacts/baseline.json | ConvertFrom-Json).timeout_seconds
```

Inspect the actual service state before deciding what to run next.

```powershell
uv run ae --json experiment describe $Experiment
```

For `ACTIVE`/`RUNNING`, repeat the `experiment run` command from step 6.
Only if the remote state is `PAUSED`, resume it first:

```powershell
uv run ae --json experiment resume $Experiment
```

Then repeat `experiment run`. For `COMPLETED`, go to export. For `FAILED` or
`CANCELLED`, export and inspect diagnostics; do not silently create a new run.
If creation succeeded but the terminal lost the returned ID, find it with:

```powershell
uv run ae --json experiment list
```

Attach that experiment instead of creating another one. If seed upload succeeded
but activation failed, `names_poc evolve --experiment $Experiment` checks the
existing seed and recovers activation without uploading another seed.

## 7. Inspect and export candidates

```powershell
uv run ae --json results best $Experiment --top 5
```

Set the best program resource name/nickname from that output.

```powershell
$BestProgram = "PASTE_PROGRAM_NAME_OR_NICKNAME"
```

```powershell
uv run ae program diff $BestProgram --experiment $Experiment
```

This is a **parent-relative** diff. Finalization also writes a diff against the
original seed, which may span multiple evolutionary steps.

Export a single program manually if desired:

```powershell
uv run ae program show $BestProgram --experiment $Experiment --code --output-file ./artifacts/inspected_program.py
```

Archive every program with identifiers, scores, diagnostics, and candidate code:

```powershell
uv run python -m names_poc export --experiment $Experiment
```

Expected: `artifacts/candidates.json`, `cloud_programs.json`, and a directory for
each exported candidate. Keep source/checkpoints together; candidate code must
match the weights when reloading.

## 8. Run the ordinary-tuning comparison locally

```powershell
uv run python -m names_poc tune --trials 50
```

This runs 50 Optuna trials under the same training/seeds/parameter/latency limits.
It does not call Gemini. Failed trials count toward 50. Repeating the command
resumes to **50 total**, not 50 additional trials. The best trial is selected on
validation only. Do not run it concurrently with CPU evolution measurements.

## 9. Freeze finalists, then evaluate test once

```powershell
uv run python -m names_poc finalize --experiment $Experiment
```

The search must be finished first. This retrains the top three distinct evolved
architectures plus both comparators for 5,000 updates across seeds 101–105.
It selects the evolved finalist on validation and writes `final_selection.json`
before loading test data. Test is then measured for that fixed finalist and
both comparators. Reruns reuse the locked selection and completed results.
If all evolved candidates fail, no improvement is claimed.

```powershell
uv run python -m names_poc report
```

Read `artifacts/report.md`. It reports NLL/perplexity, seed variation, parameter
counts, training time, latency, and sample diagnostics. Compare actual cloud
charges in Cloud Billing; the CLI does not expose a complete currency cost.

Success target: >=1% lower mean test NLL than the seed and improvement in >=4/5
seeds, while meeting resource limits. Superiority to tuning requires the same
result against Optuna. If the result fails, keep the stronger comparator and
report that outcome; do not retune against this test split.

The final export contains five checkpoints per comparator, model source,
vocabulary/configuration in each checkpoint, 1,000 samples at temperature 1.0,
and `evolved_vs_baseline.diff`. The generation checkpoint is always seed 101,
not a checkpoint selected by test performance.

## Later runs: optional automation

After setup/data/baseline, this wraps create/start/run and exports results:

```powershell
uv run python -m names_poc evolve --max-programs 50
```

It is billable when it starts/resumes cloud generation. With `run.json` present,
it reconnects to that experiment. To attach an existing manual run explicitly:

```powershell
uv run python -m names_poc evolve --experiment $Experiment
```

For a different corpus or protocol, use a fresh project copy and new artifacts.
Do not remove frozen manifests just to bypass a mismatch in an active study.

## Common failures

- **ImportError: torch/names_poc:** use `uv run ae`, not an unrelated globally installed `ae`; rerun `uv sync --locked`.
- **403 or missing engine:** verify the active profile, APIs, entitlement, engine ID, and account permissions. Credentials alone do not provision AlphaEvolve.
- **Frozen inputs changed:** restore the original files or start a separate study. Changing model limits mid-search invalidates comparisons.
- **Timeout:** diagnostics are persisted and the candidate fails. The baseline-derived limit stays fixed during search.
- **All candidates fail:** inspect `stderr.log` and `program.json` before spending on another search.
- **Copied project cannot find artifacts:** paths in run artifacts are local absolute paths. Run the study from its original location; use a fresh artifact directory for a relocated study.
- **Need only current progress:** `report` works before finalization and does not launch training or use the cloud.
