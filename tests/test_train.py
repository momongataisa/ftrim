"""The parts of the training loop that do not need a model."""
import pytest

from ftrim import train


def test_one_pass_uses_every_question_at_most_once():
    ids = [f"q{i}" for i in range(100)]
    steps = train.step_order(ids, seed=17, batch=32)
    assert len(steps) == 3 and all(len(s) == 32 for s in steps)        # the incomplete last batch is not used
    flat = [i for s in steps for i in s]
    assert len(set(flat)) == 96
    assert steps == train.step_order(list(reversed(ids)), seed=17, batch=32)   # the order depends on the seed only
    assert steps != train.step_order(ids, seed=42, batch=32)


def test_learning_rate_decays_linearly_to_zero():
    assert train.lr_factor(0, 300) == 1.0
    assert train.lr_factor(100, 300) == pytest.approx(2 / 3)           # after the one trained pass of three
    assert train.lr_factor(300, 300) == 0.0 and train.lr_factor(400, 300) == 0.0


def test_micro_batches_keep_the_padded_size_within_the_limit():
    items = [{"id": f"q{i}", "input_ids": [0] * n} for i, n in enumerate([900, 100, 400, 50, 3000, 120])]
    groups = train.micro_batches(items, max_tokens=1000)
    assert sorted(i["id"] for g in groups for i in g) == sorted(i["id"] for i in items)
    assert [len(g[0]["input_ids"]) for g in groups] == sorted((len(g[0]["input_ids"]) for g in groups), reverse=True)
    for g in groups:
        assert len(g) == 1 or max(len(i["input_ids"]) for i in g) * len(g) <= 1000
    assert [i["id"] for i in groups[0]] == ["q4"]                      # a pair over the limit is alone
