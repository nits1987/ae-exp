import shutil
from pathlib import Path

import pytest

from names_poc.data import prepare, validate_data, read_names


def test_supplied_ranked_names_import(tmp_path):
    original = Path(__file__).resolve().parents[1]
    shutil.copy2(original / "config.json", tmp_path / "config.json")
    source = original / "data.txt"
    before = source.read_bytes()
    result = prepare(tmp_path, input_file=source)
    assert result["source_type"] == "local_file"
    assert result["year"] is None
    assert result["source_rows"] == 1000
    assert result["rows"] == 2000
    assert result["unique_names"] == 1918
    assert result["duplicates_removed"] == 82
    assert result["rejected_rows"] == 0
    assert result["counts"] == {"train": 1534, "validation": 191, "test": 193}
    assert source.read_bytes() == before
    assert prepare(tmp_path, input_file=source) == result
    validate_data(tmp_path)
    assert sum("avery" in read_names(tmp_path, split) for split in ("train", "validation", "test")) == 1
    assert (tmp_path / "names.txt").read_text().splitlines() == sorted(set((tmp_path / "names.txt").read_text().splitlines()))
    with pytest.raises(ValueError, match="different source"):
        prepare(tmp_path)


def test_input_changes_cannot_replace_prepared_data(tmp_path):
    original = Path(__file__).resolve().parents[1]
    shutil.copy2(original / "config.json", tmp_path / "config.json")
    source = tmp_path / "input.txt"
    source.write_text("\n".join(f"name{chr(97+i)}{chr(97+j)}" for i in range(4) for j in range(10)))
    prepare(tmp_path, input_file=source)
    source.write_text(source.read_text()+"\nextra")
    with pytest.raises(ValueError, match="different source"):
        prepare(tmp_path, input_file=source)
