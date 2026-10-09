"""Human-readable evidence, including incomplete and unsuccessful runs."""
import math
import os
import statistics
from pathlib import Path

from .storage import read_json, write_json


def comparison(evolved: list[dict], other: list[dict]) -> dict:
    pairs = {r["seed"]: r["test_nll"] for r in other}
    if set(pairs) != {r["seed"] for r in evolved}:
        raise ValueError("Comparison requires the same training seeds")
    old = statistics.mean(pairs.values())
    new = statistics.mean(r["test_nll"] for r in evolved)
    gain = (old-new)/old
    wins = sum(r["test_nll"] < pairs[r["seed"]] for r in evolved)
    return {"relative_nll_improvement": gain, "winning_seeds": wins,
            "passes": gain >= .01 and wins >= 4 and len(evolved) == 5}


def report(root: Path) -> Path:
    directory = root / "artifacts"
    directory.mkdir(exist_ok=True)
    lines = ["# Names model experiment report", "", "Lower NLL is better. Fitness is exp(-validation NLL).", ""]
    rows = [read_json(path) for path in sorted((directory / "evaluations").glob("*/result.json"))]
    lines += [f"Persisted evaluations: {len(rows)}; failures: {sum(r['status'] != 'ok' for r in rows)}.",
              f"Recorded local evaluation time: {sum(r['elapsed_seconds'] for r in rows)/3600:.3f} hours (cached reuse adds no training).", ""]
    cloud_path = directory / "candidates.json"
    if cloud_path.exists():
        exported = read_json(cloud_path)
        lines += [f"Cloud experiment: `{exported['experiment']}`; state: {exported['state']}.",
                  f"Exported programs (including seed): {len(exported['programs'])}.", ""]
    lines += ["Cloud charges are not supplied by this CLI. Read actual AlphaEvolve/Gemini charges in Cloud Billing; no monetary estimate is inferred from trial counts.", ""]
    write_json(directory / "cloud_usage.json", {"status": "not_available_from_cli", "source": "Google Cloud Billing", "cost": None})
    final_path = directory / "final_results.json"
    if final_path.exists():
        final = read_json(final_path)
        summaries = {}
        lines += ["## Locked test comparison", "", "| Model | Test NLL mean ± SD | Perplexity | Parameters | Training seconds mean | Inference p95 ms mean |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, result in final["models"].items():
            seeds = result["seeds"]
            values = [r["test_nll"] for r in seeds]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values)>1 else 0
            summaries[name] = {"mean_test_nll": mean, "std_test_nll": sd}
            lines.append(f"| {name} | {mean:.6f} ± {sd:.6f} | {math.exp(mean):.3f} | {seeds[0]['parameters']} | {statistics.mean(r['train_seconds'] for r in seeds):.2f} | {statistics.mean(r['inference_p95_ms'] for r in seeds):.4f} |")
        evolved = final["models"]["evolved"]["seeds"]
        versus_seed = comparison(evolved, final["models"]["baseline"]["seeds"])
        versus_tuning = comparison(evolved, final["models"]["optuna"]["seeds"])
        accepted = versus_seed["passes"] and versus_tuning["passes"]
        lines += ["", f"Evolved vs seed: {versus_seed['relative_nll_improvement']:.2%} NLL reduction; better in {versus_seed['winning_seeds']}/5 seeds.",
                  f"Evolved vs Optuna: {versus_tuning['relative_nll_improvement']:.2%} NLL reduction; better in {versus_tuning['winning_seeds']}/5 seeds.",
                  f"Acceptance: {'passes both comparisons' if accepted else 'does not establish superiority to both baselines; retain the existing comparator'}.",
                  "These are descriptive repeatability checks, not a statistical significance claim.", "",
                  "## Individual training seeds", "", "| Model | Seed | Validation NLL | Test NLL |", "|---|---:|---:|---:|"]
        for name, result in final["models"].items():
            for row in result["seeds"]:
                lines.append(f"| {name} | {row['seed']} | {row['validation_nll']:.6f} | {row['test_nll']:.6f} |")
        lines += ["", "## Sampling diagnostics", "", "Temperature 1.0; checkpoint seed 101; sampling seed 2026. No test-driven checkpoint selection.", "",
                  "| Model | Duplicates | Training-name matches | Terminated | Empty |", "|---|---:|---:|---:|---:|"]
        for name, result in final["models"].items():
            sample = result["samples"]
            lines.append(f"| {name} | {sample['duplicate_rate']:.1%} | {sample['training_match_rate']:.1%} | {sample['termination_rate']:.1%} | {sample['empty_rate']:.1%} |")
        write_json(directory / "comparison.json", {"summaries": summaries, "vs_baseline": versus_seed, "vs_optuna": versus_tuning, "passes_both": accepted})
    else:
        lines += ["Final test evaluation has not been completed. No claim of improvement is supported yet.", ""]
    valid = [r for r in rows if r["status"] == "ok" and r["phase"] in {"search", "tuning"}]
    if valid:
        os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(figsize=(7, 4))
        for phase in ("search", "tuning"):
            scores = [r["validation_nll"] for r in valid if r["phase"] == phase]
            if scores:
                # Cache IDs are not chronological; use a sorted distribution plot.
                axes.plot(range(1, len(scores)+1), sorted(scores), marker=".", label=phase)
        axes.set(xlabel="Candidate rank", ylabel="Validation NLL", title="Candidate quality distribution")
        axes.legend()
        figure.tight_layout()
        figure.savefig(directory / "candidate_quality.png", dpi=160)
        plt.close(figure)
        lines += ["", "![Candidate quality](candidate_quality.png)", ""]
    lines += ["## Limits", "", "Architecture search can overfit validation. Test selection was frozen before test scoring. Equal trial counts are not equal CPU time or cloud cost. Names-model results do not establish ETA improvements. Local subprocesses are not an adversarial code sandbox.", ""]
    output = directory / "report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
