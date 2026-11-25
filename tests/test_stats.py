"""The two-level bootstrap of the pooled intervals and the same-seed differences."""
import numpy as np
import pytest

from ftrim import stats

RESAMPLES = 4000


def scores(rng, n_seeds, n_questions, p, spread):
    """seeds x questions of 0/100 scores; each seed shifts the success probability by up to `spread`."""
    shift = np.linspace(-spread, spread, n_seeds)[:, None] if n_seeds > 1 else np.zeros((1, 1))
    return 100.0 * (rng.random((n_seeds, n_questions)) < p + shift)


def document_draws(diff, clusters, seed):
    """A plain cluster bootstrap of a per-question difference (documents only)."""
    keys, inv = np.unique(np.array(clusters), return_inverse=True)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=RESAMPLES).astype(float)
    return (w @ np.bincount(inv, weights=diff)) / (w @ np.bincount(inv).astype(float))


def test_without_seed_variance_the_interval_is_the_document_interval():
    rng = np.random.default_rng(0)
    a = np.repeat(scores(rng, 1, 600, 0.7, 0.0), 3, axis=0)            # three equal seeds
    b = np.repeat(scores(rng, 1, 600, 0.6, 0.0), 3, axis=0)
    clusters = [f"doc{i // 5}" for i in range(600)]
    two = stats.two_level_draws(a, b, clusters, documents=(100,), seeds=(100, 1), resamples=RESAMPLES)
    docs = document_draws(a.mean(axis=0) - b.mean(axis=0), clusters, 100)
    assert two.mean() == pytest.approx(a.mean() - b.mean(), abs=0.15)
    assert np.std(two) == pytest.approx(np.std(docs), rel=0.05)


def test_seed_variance_widens_the_interval():
    rng = np.random.default_rng(1)
    a, b = scores(rng, 3, 600, 0.7, 0.08), scores(rng, 3, 600, 0.6, 0.08)
    clusters = [f"doc{i // 5}" for i in range(600)]
    two = stats.two_level_draws(a, b, clusters, documents=(100,), seeds=(100, 1), resamples=RESAMPLES)
    docs = document_draws(a.mean(axis=0) - b.mean(axis=0), clusters, 100)
    assert np.std(two) > 1.2 * np.std(docs)


def test_a_single_run_is_compared_with_each_seed():
    rng = np.random.default_rng(2)
    a, u = scores(rng, 3, 200, 0.7, 0.05), scores(rng, 1, 200, 0.6, 0.0)
    assert stats.same_seed_differences(a.mean(axis=1), u.mean(axis=1)) == pytest.approx(list(a.mean(axis=1) - u.mean()))
    draws = stats.two_level_draws_benchmarks([(a, u), (a[:, ::-1].copy(), u)], questions=(300,), seeds=(300, 1),
                                              resamples=RESAMPLES)
    assert draws.shape == (RESAMPLES,)
    assert draws.mean() == pytest.approx(a.mean() - u.mean(), abs=0.4)


def test_streams_are_reproducible_and_distinct():
    rng = np.random.default_rng(3)
    a, b = scores(rng, 3, 300, 0.7, 0.05), scores(rng, 3, 300, 0.6, 0.05)
    clusters = [f"doc{i // 3}" for i in range(300)]

    def draws(documents, seeds):
        return stats.two_level_draws(a, b, clusters, documents=documents, seeds=seeds, resamples=RESAMPLES)

    one = draws((100, 0), (100, 1))
    assert np.array_equal(one, draws((100, 0), (100, 1)))
    assert not np.array_equal(one, draws((101, 0), (100, 1)))      # other documents
    assert not np.array_equal(one, draws((100, 0), (101, 1)))      # other seeds
    pairs = [(a, b), (b, a)]
    first = stats.two_level_draws_benchmarks(pairs, questions=(7,), seeds=(7, 1), resamples=RESAMPLES)
    assert np.array_equal(first, stats.two_level_draws_benchmarks(pairs, questions=(7,), seeds=(7, 1),
                                                                  resamples=RESAMPLES))


def test_cached_weights_do_not_change_the_draws():
    rng = np.random.default_rng(4)
    a, b, c = (scores(rng, 3, 120, p, 0.05) for p in (0.7, 0.6, 0.5))
    clusters = [f"doc{i // 4}" for i in range(120)]
    first = stats.two_level_draws(a, b, clusters, documents=(11, 0), seeds=(11, 1), resamples=500)
    stats.two_level_draws(a, c, clusters, documents=(11, 0), seeds=(11, 1), resamples=500)  # another pair, same streams
    stats._document_weights.cache_clear()
    stats._seed_weights.cache_clear()
    assert np.array_equal(first, stats.two_level_draws(a, b, clusters, documents=(11, 0), seeds=(11, 1),
                                                       resamples=500))


def test_settings_on_one_test_set_share_its_documents():
    rng = np.random.default_rng(5)
    a = np.repeat(scores(rng, 1, 600, 0.7, 0.0), 3, axis=0)            # three equal seeds: no seed variance
    b = np.repeat(scores(rng, 1, 600, 0.6, 0.0), 3, axis=0)
    clusters = [f"doc{i // 5}" for i in range(600)]
    first, second = (stats.two_level_draws(a, b, clusters, documents=(1, 0), seeds=(1, 1, k), resamples=RESAMPLES)
                     for k in (0, 1))
    # the same differences in two settings on one test set: their mean is as uncertain as each of them
    assert np.std((first + second) / 2) == pytest.approx(np.std(first), rel=1e-9)
    other = stats.two_level_draws(a, b, clusters, documents=(2, 0), seeds=(1, 1, 1), resamples=RESAMPLES)
    assert np.std((first + other) / 2) < 0.8 * np.std(first)          # two test sets: independent draws


def test_pooled_counts_the_same_seed_pairs():
    diffs = [1.0, 0.1, 0.5, 0.2]
    same = [[1.0, 0.5, 1.5], [0.2, -0.1, 0.2], [0.4, 0.5, 0.6], [0.0, 0.3, 0.3]]
    p = stats.pooled(diffs, [np.full(10, d) for d in diffs], same)
    assert p["diff"] == pytest.approx(0.45) and p["settings_positive"] == 4
    assert p["ci95"] == pytest.approx([0.45, 0.45])
    assert (p["seeds_positive"], p["seed_pairs"]) == (10, 12)          # a tie (0.0) is not above
    assert "ci95" not in stats.pooled(diffs)
