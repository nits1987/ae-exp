"""Fixed training and measurement logic. Never uploaded as candidate code."""
from __future__ import annotations

import importlib.util
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import encode_names, read_names
from .storage import read_json


def load_model(source: Path, cfg: dict):
    spec = importlib.util.spec_from_file_location("candidate_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model(dict(cfg))
    if not isinstance(model, torch.nn.Module):
        raise ValueError("build_model must return torch.nn.Module")
    params = sum(p.numel() for p in model.parameters())
    if not 0 < params <= cfg["max_parameters"]:
        raise ValueError(f"Parameter count {params} exceeds allowed range 1..{cfg['max_parameters']}")
    return model.cpu(), params


def check_logits(logits, batch: int, vocab_size: int):
    if not isinstance(logits, torch.Tensor) or logits.shape != (batch, vocab_size):
        raise ValueError(f"Expected logits [{batch}, {vocab_size}]")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")


@torch.inference_mode()
def nll(model, tensors, vocab_size: int, batch_size: int = 1024) -> float:
    model.eval()
    x, y = tensors
    total = 0.0
    for begin in range(0, len(y), batch_size):
        xb, yb = x[begin:begin+batch_size], y[begin:begin+batch_size]
        logits = model(xb)
        check_logits(logits, len(yb), vocab_size)
        # Accumulate in float64 to avoid dependence on evaluation batch boundaries.
        total += F.cross_entropy(logits.double(), yb, reduction="sum").item()
    value = total / len(y)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid validation NLL")
    return value


@torch.inference_mode()
def latency(model, cfg: dict) -> float:
    model.eval()
    x = torch.zeros((1, cfg["context_length"]), dtype=torch.long)
    for _ in range(cfg["latency_warmup"]):
        model(x)
    measurements = []
    for _ in range(cfg["latency_repeats"]):
        start = time.perf_counter_ns()
        logits = model(x)
        measurements.append((time.perf_counter_ns()-start) / 1e6)
    check_logits(logits, 1, cfg["vocab_size"])
    measurements.sort()
    return measurements[math.ceil(.95 * len(measurements))-1]


def train(request: dict, root: Path, directory: Path) -> dict:
    cfg = request["config"]
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    characters = read_json(root / "data/vocabulary.json")["characters"]
    cfg = dict(cfg, vocab_size=len(characters)+1)
    train_xy = encode_names(read_names(root, "train"), characters, cfg["context_length"])
    train_eval_xy = (train_xy[0][:20480], train_xy[1][:20480])
    validation_xy = encode_names(read_names(root, "validation"), characters, cfg["context_length"])
    rows = []
    for seed in request["seeds"]:
        random.seed(seed)
        torch.manual_seed(seed)
        model, params = load_model(directory / "initial_program.py", cfg)
        check_logits(model(train_xy[0][:2]), 2, cfg["vocab_size"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        # This RNG never depends on the model's initialization or dropout RNG.
        batches = torch.Generator().manual_seed(seed)
        started = time.perf_counter()
        model.train()
        for _ in range(request["steps"]):
            indices = torch.randint(len(train_xy[1]), (cfg["batch_size"],), generator=batches)
            logits = model(train_xy[0][indices])
            check_logits(logits, cfg["batch_size"], cfg["vocab_size"])
            loss = F.cross_entropy(logits, train_xy[1][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gradients")
            optimizer.step()
        train_seconds = time.perf_counter() - started
        tr_nll = nll(model, train_eval_xy, cfg["vocab_size"], cfg["eval_batch_size"])
        val = nll(model, validation_xy, cfg["vocab_size"], cfg["eval_batch_size"])
        inference_ms = latency(model, cfg)
        if request.get("latency_cap_ms") is not None and inference_ms > request["latency_cap_ms"]:
            raise ValueError(
                f"Inference p95 {inference_ms:.4f}ms exceeds {request['latency_cap_ms']:.4f}ms "
                f"(achieved val_NLL={val:.4f}, train_NLL={tr_nll:.4f}, overfit_gap={val-tr_nll:+.4f}, params={params})"
            )
        checkpoint = directory / f"seed_{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg, "characters": characters,
                    "seed": seed, "steps": request["steps"]}, checkpoint)
        rows.append({"seed": seed, "train_nll": tr_nll, "validation_nll": val, "parameters": params,
                     "train_seconds": train_seconds, "inference_p95_ms": inference_ms,
                     "checkpoint": checkpoint.name})
    mean = statistics.mean(row["validation_nll"] for row in rows)
    mean_train = statistics.mean(row["train_nll"] for row in rows)
    gap = mean - mean_train
    max_lat = max(row["inference_p95_ms"] for row in rows)
    cap_str = f"/{request['latency_cap_ms']:.4f}ms" if request.get("latency_cap_ms") is not None else ""
    return {"status": "ok", "score": math.exp(-mean), "validation_nll": mean, "train_nll": mean_train,
            "perplexity": math.exp(mean), "seeds": rows,
            "insights": [{"label": "validation", "text": (
                f"val_NLL={mean:.6f}; train_NLL={mean_train:.6f}; overfit_gap={gap:+.4f}; "
                f"params={rows[0]['parameters']}/{cfg['max_parameters']}; "
                f"latency_p95={max_lat:.4f}ms{cap_str}; seeds={request['seeds']}"
            )}]}


@torch.inference_mode()
def sample(model, cfg: dict, characters: list[str], training_names: list[str], seed: int) -> dict:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    count = cfg["sample_count"]
    context = torch.zeros((count, cfg["context_length"]), dtype=torch.long)
    names = [""] * count
    ended = [False] * count
    for _ in range(cfg["sample_max_length"]):
        logits = model(context)
        check_logits(logits, count, cfg["vocab_size"])
        tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).flatten()
        for i, token in enumerate(tokens.tolist()):
            if not ended[i]:
                if token == 0:
                    ended[i] = True
                else:
                    names[i] += characters[token-1]
        context = torch.cat((context[:, 1:], tokens[:, None]), dim=1)
        if all(ended):
            break
    known = set(training_names)
    return {"names": names, "temperature": 1.0, "seed": seed,
            "duplicate_rate": 1-len(set(names))/count,
            "training_match_rate": sum(name in known for name in names)/count,
            "termination_rate": sum(ended)/count,
            "empty_rate": sum(not name for name in names)/count}


def test_checkpoints(request: dict, root: Path, directory: Path) -> dict:
    """Called only after final selection has been durably frozen."""
    torch.set_num_threads(request["config"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_dir = Path(request["training_directory"])
    trained = read_json(source_dir / "result.json")
    rows = []
    samples = None
    for i, row in enumerate(trained["seeds"]):
        checkpoint = torch.load(source_dir / row["checkpoint"], map_location="cpu", weights_only=True)
        cfg, characters = checkpoint["config"], checkpoint["characters"]
        model, _ = load_model(source_dir / "initial_program.py", cfg)
        model.load_state_dict(checkpoint["state_dict"])
        test_xy = encode_names(read_names(root, "test"), characters, cfg["context_length"])
        rows.append(dict(row, test_nll=nll(model, test_xy, cfg["vocab_size"], cfg["eval_batch_size"])))
        if i == 0:  # Fixed seed 101, never selected using test performance.
            samples = sample(model, cfg, characters, read_names(root, "train"), seed=2026)
    return {"status": "ok", "seeds": rows, "samples": samples}
