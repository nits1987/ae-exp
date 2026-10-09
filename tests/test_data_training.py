import math

import pytest
import torch

from names_poc.data import encode_names, read_names, validate_data
from names_poc.storage import read_json, verify_frozen
from names_poc.training import check_logits, nll


def test_data_deduplicated_and_disjoint(project):
    manifest = read_json(project / "data/manifest.json")
    assert manifest["counts"] == {"train": 80, "validation": 10, "test": 10}
    assert manifest["duplicates_removed"] == 1
    assert manifest["rejected_rows"] == 1
    train, val, test = (set(read_names(project, s)) for s in ("train", "validation", "test"))
    assert not train & val and not train & test and not val & test
    validate_data(project)


def test_context_has_only_past_tokens_and_eos():
    x, y = encode_names(["ab", "a"], ["a", "b"], 2)
    assert x.tolist() == [[0, 0], [0, 1], [1, 2], [0, 0], [0, 1]]
    assert y.tolist() == [1, 2, 0, 1, 0]


def test_nll_weights_tokens_not_batches():
    class Lookup(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2., 0.], [0., 4.], [-1., 2.]])[x[:, 0]]
    data = (torch.tensor([[0], [1], [2], [0], [1]]), torch.tensor([0, 1, 0, 1, 1]))
    first = nll(Lookup(), data, 2, batch_size=1)
    assert first == pytest.approx(nll(Lookup(), data, 2, batch_size=3), abs=1e-12)
    assert first == pytest.approx(nll(Lookup(), data, 2, batch_size=100), abs=1e-12)


def test_invalid_logits():
    with pytest.raises(ValueError, match="Expected logits"):
        check_logits(torch.zeros(2, 3, 4), 2, 4)
    with pytest.raises(ValueError, match="Nonfinite"):
        check_logits(torch.tensor([[math.nan]]), 1, 1)


def test_frozen_protocol_detects_change(frozen_project):
    verify_frozen(frozen_project)
    (frozen_project / "data/train.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen inputs changed"):
        verify_frozen(frozen_project)
