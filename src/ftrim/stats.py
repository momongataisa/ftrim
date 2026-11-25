"""The statistics of the paper (Sect. 3, "Statistics"), in numpy.

A *run* is one fine-tuned model (one seed) evaluated on one benchmark; a method is a matrix seeds x questions of
scores in percent. The paper reports the mean over the seeds in each setting, and pooled differences: the mean over
the four settings, with a 95 % interval from a two-level bootstrap that resamples the seeds of each method and, on
the target task, whole documents (tables or passages) or, on the unseen benchmarks, the questions within each
benchmark. The differences of two methods with the same seed are in results/tables.json.

The settings share their test sets: both models are evaluated on the same documents of a target task, and all four
settings on the same questions of the unseen benchmarks. Each bootstrap sample therefore draws the documents or
questions of a test set once, for every setting evaluated on it (one random stream per test set), and the seeds of
each setting from a random stream of their own. A stream is named by a tuple of integers (`numpy.random.default_rng`).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def seed_weights(rng: np.random.Generator, n: int, size: int) -> np.ndarray:
    """size x n: the share of each of n seeds in a draw of n seeds with replacement (one run: always 1)."""
    return rng.multinomial(n, np.full(n, 1 / n), size=size) / n


# The random weights depend only on their stream and on the sizes, so that every pair of methods, and every setting
# on the same test set, shares them. They are cached: generating them is most of the cost of a comparison.
@lru_cache(maxsize=4)
def _document_weights(stream: tuple[int, ...], resamples: int, documents: int) -> np.ndarray:
    """resamples x documents: how often each document is drawn in each bootstrap sample."""
    rng = np.random.default_rng(stream)
    return rng.multinomial(documents, np.full(documents, 1 / documents), size=resamples).astype(float)


@lru_cache(maxsize=1)
def _question_weights(stream: tuple[int, ...], resamples: int, questions: tuple[int, ...]) -> list[np.ndarray]:
    """For each benchmark, resamples x questions: how often each question is drawn in each bootstrap sample."""
    rng = np.random.default_rng(stream)
    counts = []
    for q in questions:
        w = rng.multinomial(q, np.full(q, 1 / q), size=resamples)
        if w.max() > 255:
            raise OverflowError("a question was drawn more than 255 times")
        counts.append(w.astype(np.uint8))       # kept small: resamples x questions per benchmark
    return counts


@lru_cache(maxsize=64)
def _seed_weights(stream: tuple[int, ...], resamples: int, seeds_a: int, seeds_b: int):
    rng = np.random.default_rng(stream)
    return seed_weights(rng, seeds_a, resamples), seed_weights(rng, seeds_b, resamples)


def two_level_draws(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, *, documents: tuple[int, ...],
                    seeds: tuple[int, ...], resamples: int) -> np.ndarray:
    """Bootstrap draws of the mean difference a - b on one test set.

    a, b: seeds x questions (percent); the two methods may have different numbers of seeds. `clusters` gives the
    document of each question. Each draw resamples whole documents with replacement (stream `documents`) and,
    independently for the two methods, their seeds with replacement (stream `seeds`).
    """
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    nk = len(keys)
    w = _document_weights(tuple(documents), resamples, nk)                                       # draws x documents
    wa, wb = _seed_weights(tuple(seeds), resamples, len(a), len(b))
    sums = [np.stack([np.bincount(inv, weights=x, minlength=nk) for x in m]) for m in (a, b)]   # seeds x documents
    da, db = wa @ sums[0], wb @ sums[1]
    return ((da - db) * w).sum(axis=1) / (w @ np.bincount(inv, minlength=nk).astype(float))


def two_level_draws_benchmarks(pairs: list[tuple[np.ndarray, np.ndarray]], *, questions: tuple[int, ...],
                               seeds: tuple[int, ...], resamples: int) -> np.ndarray:
    """Bootstrap draws of the mean over several benchmarks of the differences a - b.

    pairs: one (a, b) per benchmark, seeds x questions each. One seed is one model, so each draw resamples the seeds
    of a method once for all benchmarks (stream `seeds`), and the questions within each benchmark (stream
    `questions`).
    """
    sizes = tuple(a.shape[1] for a, _ in pairs)
    counts = _question_weights(tuple(questions), resamples, sizes)
    wa, wb = _seed_weights(tuple(seeds), resamples, len(pairs[0][0]), len(pairs[0][1]))
    draws = []
    for (a, b), wq in zip(pairs, counts):                                                      # wq: draws x questions
        wq = wq.astype(float)
        draws.append(((wa * (wq @ a.T)).sum(axis=1) - (wb * (wq @ b.T)).sum(axis=1)) / a.shape[1])
    return np.mean(draws, axis=0)


def same_seed_differences(a: np.ndarray, b: np.ndarray) -> list[float]:
    """The differences of the per-seed scores a - b with the same seed; a single run is compared with each seed."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return [float(x) for x in a - b]


def interval(draws: np.ndarray) -> list[float]:
    """The 95 % percentile interval of bootstrap draws."""
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def pooled(differences: list[float], draws: list[np.ndarray] | None = None,
           same_seed: list[list[float]] | None = None) -> dict:
    """The mean over the settings of a difference; with the settings' draws, its interval; with their same-seed
    differences, how many of them are positive."""
    out: dict = {"diff": float(np.mean(differences)), "settings_positive": int(sum(d > 0 for d in differences))}
    if draws is not None:
        out["ci95"] = interval(np.mean(draws, axis=0))
    if same_seed is not None:
        flat = [d for per_setting in same_seed for d in per_setting]
        out.update(seeds_positive=int(sum(d > 0 for d in flat)), seed_pairs=len(flat))
    return out
