"""The method itself (ftrim.core): line scores, the one-bit rule and the comparison edits."""
import math
import random

import numpy as np
import pytest

from ftrim import core


def test_lines_end_after_tokens_with_a_line_break():
    assert core.split_lines([False, True, False, False, True, False]) == [(0, 2), (2, 5), (5, 6)]
    assert core.split_lines([False, False, True]) == [(0, 3)]          # the last line ends with the last token
    assert core.split_lines([]) == []


def test_scores_are_log_likelihood_ratios_in_bits():
    lp_full, lp_bare = np.log([0.5, 0.25, 0.8]), np.log([0.25, 0.25, 0.1])
    assert core.token_scores(lp_full, lp_bare) == pytest.approx([1.0, 0.0, 3.0])
    assert core.line_scores(lp_full, lp_bare, [(0, 2), (2, 3)]) == pytest.approx([0.5, 3.0])
    # the line score is the line's log-likelihood ratio per token (Eq. 1, right)
    ratio = (lp_full[:2].sum() - lp_bare[:2].sum()) / math.log(2) / 2
    assert core.line_scores(lp_full, lp_bare, [(0, 2)])[0] == pytest.approx(ratio)


def test_ftrim_deletes_lines_twice_as_likely_per_token():
    bare = np.log(np.full(6, 0.1))
    full = bare + np.log([1, 1, 2, 2, 4, 1])            # ratios per token: lines score 0, 1 and 1 bit
    lines = [(0, 2), (2, 4), (4, 6)]
    assert core.format_talk_lines(full, bare, lines) == {1, 2}
    assert core.format_talk_lines(full, bare, lines, threshold_bits=0.5) == {1, 2}
    assert core.format_talk_lines(full, bare, lines, threshold_bits=2.0) == set()
    assert core.format_talk_lines(full, bare, lines, threshold_bits=0.0, strict=True) == {1, 2}
    assert core.format_talk_lines(full, bare, lines, threshold_bits=0.0) == {0, 1, 2}   # m = 0 reaches a threshold of 0


def test_the_threshold_is_inclusive_and_exact_in_nats():
    lines = [(0, 1)]
    at = np.array([math.log(2)])
    assert core.format_talk_lines(at, np.zeros(1), lines) == {0}
    assert core.format_talk_lines(at, np.zeros(1), lines, strict=True) == set()
    assert core.format_talk_lines(np.nextafter(at, 0), np.zeros(1), lines) == set()
    assert core.format_talk_lines(2 * at, np.zeros(1), lines, threshold_bits=2.0) == {0}


def test_a_long_line_needs_the_ratio_on_average():
    bare = np.zeros(10)
    full = np.zeros(10)
    full[0] = 8 * math.log(2)                           # one token with 8 bits in a line of ten: 0.8 bits per token
    assert core.format_talk_lines(full, bare, [(0, 10)]) == set()
    assert core.format_talk_lines(full, bare, [(0, 5), (5, 10)]) == {0}


def test_random_deletion_walks_towards_the_token_count():
    sizes = [10, 40, 5, 5]
    assert core.random_lines([1, 0, 2, 3], sizes, 18) == ({0, 2, 3}, 20)   # 40 overshoots; 20 is closer to 18 than 15
    assert core.random_lines([0, 1, 2, 3], sizes, 0) == (set(), 0)
    rng = random.Random(11)
    for _ in range(200):
        sizes = [rng.randint(1, 60) for _ in range(rng.randint(1, 25))]
        order = list(range(len(sizes)))
        rng.shuffle(order)
        goal = rng.randint(0, sum(sizes))
        chosen, n = core.random_lines(order, sizes, goal)
        assert n == sum(sizes[k] for k in chosen)
        assert abs(n - goal) <= goal                       # never further from the goal than no deletion
        assert all(abs(n + sizes[k] - goal) >= abs(n - goal) for k in order if k not in chosen)


def test_stm_masks_reasoning_tokens_with_high_perplexity():
    lp = np.log([0.9, 0.3, 0.5, 0.39, 0.2])
    assert core.stm_mask(lp, n_reason=4, tau=2.5) == [1, 3]                # p < 1 / 2.5; never in the answer line
    assert core.stm_mask(lp, n_reason=5, tau=2.5) == [1, 3, 4]


def test_delete_removes_whole_lines():
    tokens, lines = list(range(10, 20)), [(0, 3), (3, 4), (4, 9), (9, 10)]
    assert core.delete(tokens, lines, {1, 3}) == [10, 11, 12, 14, 15, 16, 17, 18]
    assert core.delete(tokens, lines, set()) == tokens


def test_save_npz_is_atomic_and_loads(tmp_path):
    from ftrim.utils import save_npz
    path = tmp_path / "parts" / "0000.npz"
    save_npz(path, ids=np.array(["a", "b"]), lp=np.arange(3, dtype=np.float32))
    with np.load(path) as z:
        assert list(z["ids"]) == ["a", "b"] and z["lp"].dtype == np.float32 and z["lp"].tolist() == [0.0, 1.0, 2.0]
    assert [p.name for p in path.parent.iterdir()] == ["0000.npz"]          # no temporary file is left
    assert list(path.parent.glob("*.npz")) == [path]
