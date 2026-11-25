"""The compact results (ftrim.results) and the report (ftrim.report)."""
import hashlib

import numpy as np
import pytest

from ftrim import config, report, results


def test_npz_files_depend_only_on_their_arrays(tmp_path):
    arrays = {"b/x": np.arange(5, dtype=np.int32), "a/y": np.array(["q1", "q2"])}
    results.save_npz(tmp_path / "one.npz", arrays)
    results.save_npz(tmp_path / "two.npz", dict(reversed(arrays.items())))
    digest = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()  # noqa: E731
    assert digest("one.npz") == digest("two.npz")
    with np.load(tmp_path / "one.npz") as z:
        assert sorted(z.files) == ["a/y", "b/x"] and z["b/x"].tolist() == [0, 1, 2, 3, 4]


def test_deleted_lines_and_format_words():
    before = "Find the revenue.\nRound to two decimals:\n10 - 8 = 2\n\nThe unit is thousand.\n<answer>2</answer>"
    after = "Find the revenue.\n10 - 8 = 2\n\n<answer>2</answer>"
    removed = results.removed_lines(before, after)
    assert removed == ["Round to two decimals:", "The unit is thousand."]
    assert all(results.FORMAT_WORDS.search(x) for x in removed)
    assert not results.FORMAT_WORDS.search("10 - 8 = 2")
    for content in ("More information is around page 4.", "The United States community", "contract assets"):
        assert not results.FORMAT_WORDS.search(content)                # whole words only
    assert all(results.FORMAT_WORDS.search(x) for x in ("Units: thousand", "formatted as JSON", "rounded to 2"))
    assert results.normalize_line("**Step 3:** Round 12.345 to `2` decimals") == "round N to N decimals"
    assert results.normalize_line("- Revenue was 1,200") == "revenue was N"


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A small results directory with random runs of every method, in the layout of results/per_question/."""
    exp = config.experiment()
    rng = np.random.default_rng(0)
    sizes = {"tatqa": 60, "drop": 80, "finance_reasoning": 40, "gsm8k": 30, "ifeval": 20, "arc_challenge": 30}
    questions = {}
    for bench, n in sizes.items():
        questions[f"{bench}/ids"] = np.array([f"{bench}:{i:03d}" for i in range(n)])
        questions[f"{bench}/documents"] = (np.arange(n) // 4).astype(np.int32)
    results.save_npz(tmp_path / "per_question" / "questions.npz", questions)

    def run(bench: str, p: float, target: bool) -> dict:
        n = sizes[bench]
        out = {"correct": (rng.random(n) < p).astype(np.uint8), "format_valid": np.ones(n, np.uint8),
               "tokens": rng.integers(50, 400, n).astype(np.int32), "answer_tag": np.ones(n, np.uint8),
               "no_reasoning": np.zeros(n, np.uint8), "truncated": np.zeros(n, np.uint8)}
        if target:
            out["lenient"] = out["correct"]
        return out

    skill = {"ftrim": 0.8, "random_deletion": 0.7}
    for m in exp["models"]:
        arrays = {f"{method}/{b}/{k}": v for method in config.UNTRAINED for b in sizes
                  for k, v in run(b, 0.5, b in exp["datasets"]).items()}
        results.save_npz(tmp_path / "per_question" / f"{m}_untrained.npz", arrays)
    for m, ds in config.SETTINGS:
        arrays = {}
        for method, seeds in config.method_seeds().items():
            for seed in seeds:
                for b in config.eval_benchmarks(ds):
                    got = run(b, skill.get(method, 0.6), b == ds)
                    arrays.update({f"{method}/s{seed}/{b}/{k}": v for k, v in got.items()})
        results.save_npz(tmp_path / "per_question" / f"{m}_{ds}.npz", arrays)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setitem(exp["statistics"], "resamples", 300)
    return tmp_path


def test_the_report_on_synthetic_results(synthetic):
    res = results.Results()
    assert res.seeds("qwen", "tatqa", "ftrim") == [17, 42, 89] and res.seeds("qwen", "tatqa", "ftrim_zero") == [17]
    assert res.matrix("qwen", "tatqa", "ftrim", "tatqa").shape == (3, 60)
    assert res.matrix("qwen", "tatqa", "untrained", "gsm8k").shape == (1, 30)
    assert res.matrix("qwen", "tatqa", "ftrim_zero", "tatqa").shape == (1, 60)
    assert res.matrix("qwen", "tatqa", "ftrim", "tatqa", seeds=[17, 23]) is None        # a run that does not exist

    tables = report._without_draws(report.collect())
    row = tables["methods"]["ftrim"][0]
    assert row["runs"] == 3 and row["test"] == pytest.approx(np.mean(row["test_seeds"]))
    assert row["unseen_mean"] == pytest.approx(np.mean(list(row["unseen"].values())))
    d = tables["differences"]["ftrim-random_deletion"]
    assert d["test"]["diff"] == pytest.approx(np.mean([c["test"]["diff"] for c in d["settings"]]))
    assert d["test"]["ci95"][0] < d["test"]["diff"] < d["test"]["ci95"][1]
    assert d["test"]["seed_pairs"] == 12 and len(d["settings"][0]["test"]["seed_diffs"]) == 3
    assert "ci95" not in tables["threshold_differences"]["ftrim_half_bit"]["test"]     # one run: no interval
    text = report.markdown(tables)
    for heading in ("## Target accuracy", "## Generalization accuracy", "## Gains of FTrim", "## Threshold",
                    "## Normalization only"):
        assert heading in text
    assert "## What FTrim deletes" not in text and "skip.png" not in text              # no teachers.json here


def test_the_report_for_a_subset_of_seeds(synthetic):
    tables = report._without_draws(report.collect(seeds=[17]))
    assert tables["methods"]["ftrim"][0]["runs"] == 1 and tables["methods"]["ftrim"][0]["test_sd"] is None
    assert "ci95" not in tables["differences"]["ftrim-random_deletion"]["test"]
    assert "one run with seed 17" in report.markdown(tables)


def test_the_committed_results_hold_the_paper_values():
    """Table 1 of the paper from results/per_question/ (without the bootstrap; the full comparison is
    `python -m ftrim report --check`)."""
    if not (results.per_question_dir() / "questions.npz").exists():
        pytest.skip("results/per_question/ is not present")
    res = results.Results()
    rows = [report.method_row(res, m, ds, "ftrim") for m, ds in config.SETTINGS]
    assert [round(r["test"], 2) for r in rows] == [76.09, 86.71, 69.91, 84.70]
    base = [report.method_row(res, m, ds, "untrained") for m, ds in config.SETTINGS]
    assert [round(r["test"], 2) for r in base] == [62.06, 84.38, 57.37, 82.30]
    assert [round(r["unseen_mean"] - b["unseen_mean"], 2) for r, b in zip(rows, base)] == [0.48, -0.21, -0.21, -0.82]
    assert round(float(np.mean([r["tokens"] for r in rows]))) == 326
