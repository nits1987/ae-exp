"""SSA or local-name ingestion and deterministic, disjoint name splits."""
from __future__ import annotations

import csv
import hashlib
import io
import random
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from .storage import config, read_json, sha256, write_json

SSA_URL = "https://www.ssa.gov/oact/babynames/names.zip"


def prepare(root: Path, year: int = 2025, archive: Path | None = None,
            input_file: Path | None = None) -> dict:
    if input_file is not None and archive is not None:
        raise ValueError("Choose either a local input file or an SSA archive")
    if input_file is not None:
        input_file = input_file.resolve()
        if input_file == (root / "names.txt").resolve():
            raise ValueError("Input must have a different path from the generated names.txt")
    manifest_path = root / "data/manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if input_file is not None:
            if existing.get("source_type") != "local_file" or existing.get("source_sha256") != sha256(input_file):
                raise ValueError("Prepared data uses a different source; use a fresh project.")
        elif existing.get("source_type") == "local_file" or existing["year"] != year:
            raise ValueError("Prepared data uses a different source or year; use a fresh project.")
        validate_data(root)
        return existing
    if (root / "artifacts/frozen.json").exists():
        raise ValueError("Cannot prepare data after the experiment is frozen.")
    directory = root / "data"
    directory.mkdir(parents=True, exist_ok=True)
    if input_file is not None:
        lines = [line for line in input_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            raise ValueError("Input file is empty")
        header = [cell.strip().lower() for cell in lines[0].split("\t")]
        if header == ["rank", "male name", "female name"]:
            entries = []
            seen_ranks = set()
            for line_number, line in enumerate(lines[1:], 2):
                row = line.split("\t")
                if len(row) != 3 or not row[0].strip().isdigit():
                    raise ValueError(f"Invalid ranked-name row at line {line_number}")
                rank = int(row[0].strip())
                if rank < 1 or rank in seen_ranks:
                    raise ValueError(f"Invalid or duplicate rank at line {line_number}")
                seen_ranks.add(rank)
                entries.extend(row[1:])
            input_format = "ranked_male_female_tsv"
            source_rows = len(lines)-1
        else:
            entries = lines
            input_format = "one_name_per_line"
            source_rows = len(lines)
        return _write_dataset(root, entries, {
            "source": str(input_file), "source_type": "local_file", "year": None,
            "source_sha256": sha256(input_file), "input_format": input_format,
            "source_rows": source_rows,
        })
    if archive is None:
        archive = directory / "ssa-national.zip"
        if not archive.exists():
            request = urllib.request.Request(SSA_URL, headers={"User-Agent": "names-poc/0.1 public research"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    content = response.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                raise ValueError(
                    f"SSA download failed ({exc}). Download National data from "
                    "https://www.ssa.gov/oact/babynames/limits.html in your browser, then run "
                    f"prepare --ssa-year {year} --archive <path-to-names.zip>. No substitute data was used."
                ) from exc
            with zipfile.ZipFile(io.BytesIO(content)) as check:
                check.getinfo(f"yob{year}.txt")
            archive.write_bytes(content)
    try:
        with zipfile.ZipFile(archive) as zipped:
            raw = zipped.read(f"yob{year}.txt")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Archive must be the SSA national ZIP containing yob{year}.txt") from exc
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    return _write_dataset(root, [row[0] if row else "" for row in rows], {
        "source": SSA_URL, "source_type": "ssa_archive", "year": year,
        "archive_sha256": sha256(archive), "source_member": f"yob{year}.txt",
        "source_member_sha256": hashlib.sha256(raw).hexdigest(),
    })


def _write_dataset(root: Path, entries: list[str], provenance: dict) -> dict:
    directory = root / "data"
    normalized = [entry.strip().lower() for entry in entries]
    valid = [name for name in normalized if name.isalpha()]
    names = sorted(set(valid))
    if len(names) < 30:
        raise ValueError("Need at least 30 unique names for nonempty train/validation/test splits.")
    rng = random.Random(config(root)["split_seed"])
    shuffled = names.copy()
    rng.shuffle(shuffled)
    ntrain, nval = int(len(names) * .8), int(len(names) * .1)
    splits = {"train": shuffled[:ntrain], "validation": shuffled[ntrain:ntrain+nval], "test": shuffled[ntrain+nval:]}
    # Vocabulary is a frozen corpus alphabet, not a fitted statistical feature.
    chars = sorted(set("".join(names)))
    write_json(directory / "vocabulary.json", {"characters": chars, "special_id": 0})
    (root / "names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    for name, words in splits.items():
        (directory / f"{name}.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    manifest = {
        **provenance,
        "split_seed": config(root)["split_seed"], "rows": len(entries), "rejected_rows": len(entries)-len(valid),
        "duplicates_removed": len(valid)-len(names), "unique_names": len(names),
        "counts": {k: len(v) for k, v in splits.items()},
        "files": {p.relative_to(root).as_posix(): sha256(p) for p in
                  [root / "names.txt", directory / "vocabulary.json", *[directory / f"{k}.txt" for k in splits]]},
    }
    write_json(directory / "manifest.json", manifest)
    validate_data(root)
    return manifest


def validate_data(root: Path) -> None:
    manifest = read_json(root / "data/manifest.json")
    for name, expected in manifest["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    groups = [set(read_names(root, split)) for split in ("train", "validation", "test")]
    if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Empty or overlapping splits")


def read_names(root: Path, split: str) -> list[str]:
    return (root / "data" / f"{split}.txt").read_text(encoding="utf-8").splitlines()


def encode_names(words: list[str], characters: list[str], context_length: int):
    import torch
    stoi = {ch: i+1 for i, ch in enumerate(characters)}
    contexts, targets = [], []
    for word in words:
        context = [0] * context_length
        for token in [*[stoi[ch] for ch in word], 0]:
            contexts.append(context.copy())
            targets.append(token)
            context = context[1:] + [token]
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
