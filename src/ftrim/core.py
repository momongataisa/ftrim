"""FTrim in a few functions of numpy (Sect. 2 of the paper). Nothing here depends on a model, a tokenizer or a dataset.

Given one self-generated solution, tokenized as the assistant turn y_1..y_T, and the untrained model's log-probability
of each token under two prompts — x with the format instructions and the counterfactual prompt x~ with only the
answer tags —

    c_t  = log2 p0(y_t | x, y_<t) - log2 p0(y_t | x~, y_<t)        token score, in bits
    m(l) = mean of c_t over the tokens of the reasoning line l      line score, in bits per token

FTrim deletes every reasoning line with m(l) >= 1: the format instructions make the line at least twice as likely
per token. The functions take natural-log probabilities, as language-model libraries return them.

    >>> import numpy as np
    >>> lp_full = np.log([0.5, 0.5, 0.8, 0.8])      # p0(y_t | x, y_<t)
    >>> lp_bare = np.log([0.5, 0.5, 0.1, 0.2])      # p0(y_t | x~, y_<t)
    >>> lines = [(0, 2), (2, 4)]                    # token ranges of the two reasoning lines
    >>> [round(m, 2) for m in line_scores(lp_full, lp_bare, lines)]
    [0.0, 2.5]
    >>> sorted(format_talk_lines(lp_full, lp_bare, lines))
    [1]
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

LN2 = math.log(2)
Lines = Sequence[tuple[int, int]]          # [start, end) token ranges of the reasoning lines


def split_lines(ends_line: Sequence[bool]) -> list[tuple[int, int]]:
    """The lines of a reasoning of len(ends_line) tokens: a line ends after each token whose text contains a line
    break (`ends_line[t]`), and the last line ends with the last token."""
    lines, start, n = [], 0, len(ends_line)
    for t in range(n):
        if ends_line[t] or t == n - 1:
            lines.append((start, t + 1))
            start = t + 1
    return lines


def token_scores(lp_full: np.ndarray, lp_bare: np.ndarray) -> np.ndarray:
    """c_t in bits: how much more likely the format instructions make each token (Eq. 1)."""
    return (np.asarray(lp_full, np.float64) - np.asarray(lp_bare, np.float64)) / LN2


def _line_means_nats(lp_full: np.ndarray, lp_bare: np.ndarray, lines: Lines) -> list[float]:
    c = np.asarray(lp_full, np.float64) - np.asarray(lp_bare, np.float64)
    return [float(c[a:b].mean()) for a, b in lines]


def line_scores(lp_full: np.ndarray, lp_bare: np.ndarray, lines: Lines) -> list[float]:
    """m(l) in bits per token for every line: the line's log-likelihood ratio between the two prompts, per token."""
    return [m / LN2 for m in _line_means_nats(lp_full, lp_bare, lines)]


def format_talk_lines(lp_full: np.ndarray, lp_bare: np.ndarray, lines: Lines, *, threshold_bits: float = 1.0,
                      strict: bool = False) -> set[int]:
    """The lines that FTrim deletes: m(l) >= threshold (`strict`: m(l) > threshold). Indices into `lines`."""
    theta = threshold_bits * LN2            # compared in nats, the unit of the stored log-probabilities
    means = _line_means_nats(lp_full, lp_bare, lines)
    return {k for k, m in enumerate(means) if m > theta or (not strict and m == theta)}


def random_lines(order: Sequence[int], sizes: Sequence[int], goal: int) -> tuple[set[int], int]:
    """Random deletion of a given number of tokens: visit the lines in `order` and delete a line whenever the number
    of deleted tokens then comes closer to `goal`. Returns the lines and their token count."""
    chosen, n = set(), 0
    for k in order:
        if abs(n + sizes[k] - goal) < abs(n - goal):
            chosen.add(k)
            n += sizes[k]
    return chosen, n


def stm_mask(lp_full: np.ndarray, n_reason: int, tau: float) -> list[int]:
    """Selective Token Masking (Wu et al., 2025): the reasoning tokens whose perplexity under the untrained model
    exceeds tau, i.e. p0(y_t | x, y_<t) < 1 / tau. They stay in the input and leave the loss."""
    return [int(i) for i in np.flatnonzero(np.asarray(lp_full, np.float64)[:n_reason] < -math.log(tau))]


def delete(tokens: Sequence[int], lines: Lines, deleted: set[int]) -> list[int]:
    """The reasoning tokens without the deleted lines (the edited teacher is these tokens, then the answer line)."""
    gone = {t for k in deleted for t in range(*lines[k])}
    return [tok for t, tok in enumerate(tokens) if t not in gone]
