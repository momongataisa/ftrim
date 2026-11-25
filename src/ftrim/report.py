"""The tables of the results from results/: Table 1 of the paper, the gains, the ablations and the analysis (CPU).

    python -m ftrim report            # -> results/RESULTS.md and results/tables.json
    python -m ftrim report --check    # also compare with the numbers of the paper (ftrim.paper)
    python -m ftrim report --seeds 17 # only the runs with these seeds (a partial reproduction)

The report reads only results/per_question/, results/teachers.json and results/mechanism.json, so it runs without
the run directory and without a GPU. A value whose runs are missing is shown as "–".
"""
from __future__ import annotations

import os

import numpy as np

from . import config, stats
from .results import Results
from .utils import read_json, write_json

HEADS = {("qwen", "tatqa"): "Qwen-TAT", ("qwen", "drop"): "Qwen-DROP", ("gemma", "tatqa"): "Gemma-TAT",
         ("gemma", "drop"): "Gemma-DROP"}
NAMES = {"untrained": "Untrained", "thinking": "+ Thinking mode", "sft": "SFT on gold answers",
         "rejection": "Rejection sampling", "rationalization": "+ Rationalization",
         "random_deletion": "Random deletion", "stm": "STM", "ftrim": "FTrim (ours)",
         "normalized": "Normalization only",
         "ftrim_half_bit": "FTrim, threshold ½ bit", "ftrim_two_bits": "FTrim, threshold 2 bits",
         "ftrim_zero": "FTrim, threshold 0 bits (m > 0)"}
SHORT = {"untrained": "untrained", "thinking": "thinking mode", "sft": "SFT", "rejection": "rejection sampling",
         "rationalization": "rationalization", "random_deletion": "random deletion", "stm": "STM", "ftrim": "FTrim",
         "normalized": "normalization only", "ftrim_half_bit": "½ bit", "ftrim_two_bits": "2 bits",
         "ftrim_zero": "0 bits"}
BENCH = {"tatqa": "TAT-QA", "drop": "DROP", "finance_reasoning": "FinanceReasoning", "gsm8k": "GSM8K",
         "ifeval": "IFEval", "arc_challenge": "ARC-Challenge"}
SIMPLE = {"math": "MATH", "gsm8k_strict": "GSM8K", "medcalc": "MedCalc-Bench", "medqa": "MedQA"}
TABLE = ["untrained", "thinking", "sft", "rejection", "rationalization", "random_deletion", "stm", "ftrim"]
ROWS = [*TABLE, "normalized"]                    # every method with three seeds
SELF_TRAINING = ["rejection", "rationalization", "random_deletion", "stm", "ftrim"]
THRESHOLDS = ["ftrim_half_bit", "ftrim", "ftrim_two_bits", "ftrim_zero"]      # by threshold; one run with seed 17
THRESHOLD_SEED = 17
# (a, b): the difference a - b, with the two-level bootstrap of ftrim.stats for the pooled value
COMPARISONS = [("ftrim", "rationalization"), ("ftrim", "random_deletion"), ("ftrim", "stm"), ("ftrim", "rejection"),
               ("ftrim", "sft"), ("ftrim", "untrained"), ("ftrim", "thinking"), ("random_deletion", "rationalization"),
               ("stm", "rationalization"), ("ftrim", "normalized"), ("normalized", "rationalization")]
# random streams of the bootstrap: (statistics seed, kind, 0, test set) for the documents or questions, shared by
# the settings on that test set; (statistics seed, kind, 1, setting) for the seeds of the methods
STREAM = {"test": 100, "lenient": 200, "unseen": 300}
GAINS = [("ftrim", "rationalization"), ("ftrim", "random_deletion"), ("ftrim", "stm")]       # the gains of Sect. 4
THRESHOLD_LABELS = {"ftrim_zero": "0 bits (m > 0)", "ftrim_half_bit": "½ bit", "ftrim": "**1 bit (FTrim)**",
                    "ftrim_two_bits": "2 bits"}                                             # in the order of the tables


# --------------------------------------------------------------------------------------------------------- collecting
def _sd(x: np.ndarray) -> float | None:
    """The sample standard deviation over the seeds (None for one run)."""
    return float(np.std(x, ddof=1)) if len(x) > 1 else None


def method_row(res: Results, m: str, ds: str, method: str, seeds: list[int] | None = None) -> dict | None:
    """The values of one method in one setting (means over its seeds), or None while a run is missing."""
    unseen_benches = config.experiment()["unseen"]

    def get(bench: str, field: str = "correct"):
        return res.matrix(m, ds, method, bench, field, seeds=seeds)

    test = get(ds)
    unseen = {b: get(b) for b in unseen_benches}
    if test is None or any(v is None for v in unseen.values()):
        return None
    unseen_seeds = np.mean([v.mean(axis=1) for v in unseen.values()], axis=0)
    tagged = get(ds, "answer_tag").sum()
    units = get(ds, "correct_units")                    # TAT-QA only ("Scoring of the TAT-QA units")
    mapped = {} if units is None else {"test_units": float(units.mean()),
                                       "lenient_units": float(get(ds, "lenient_units").mean())}
    return {**mapped, "runs": int(test.shape[0]), "test": float(test.mean()),
            "test_seeds": [float(x) for x in test.mean(axis=1)], "test_sd": _sd(test.mean(axis=1)),
            "lenient": float(get(ds, "lenient").mean()),
            "unseen": {b: float(v.mean()) for b, v in unseen.items()},
            "unseen_mean": float(np.mean([v.mean() for v in unseen.values()])),
            "unseen_seeds": [float(x) for x in unseen_seeds], "unseen_sd": _sd(unseen_seeds),
            "tokens": float(get(ds, "tokens").mean()), "format_valid": float(get(ds, "format_valid").mean()),
            "no_reasoning": float(100 * get(ds, "no_reasoning").sum() / tagged) if tagged else None,
            "truncated": float(get(ds, "truncated").mean())}


def compare(res: Results, k: int, m: str, ds: str, a: str, b: str, *, seeds: list[int] | None = None,
            units: bool = False) -> dict | None:
    """The difference a - b in setting k on the target task (strict and lenient) and on the unseen benchmarks: the mean,
    the differences with the same seed and, for the pooled interval, the bootstrap draws (key `draws`). `units`:
    TAT-QA scored with the units mapped (`scoring.MAPPED_UNITS`), with the same bootstrap samples, and no unseen
    benchmarks."""
    exp = config.experiment()
    st, unseen_benches = exp["statistics"], exp["unseen"]

    def mat(method: str, bench: str, field: str = "correct"):
        return res.matrix(m, ds, method, bench, field, seeds=None if method in config.UNTRAINED else seeds)

    def run_seeds(method: str) -> list[int]:
        return [0] if method in config.UNTRAINED else list(config.method_seeds()[method] if seeds is None else seeds)

    if len(run_seeds(a)) > 1 and len(run_seeds(b)) > 1 and run_seeds(a) != run_seeds(b):
        raise ValueError(f"{a} and {b} have other seeds ({run_seeds(a)}, {run_seeds(b)}): no same-seed differences")
    strict, lenient = ("correct_units", "lenient_units") if units and ds == "tatqa" else ("correct", "lenient")
    A, B = mat(a, ds, strict), mat(b, ds, strict)
    pairs = [] if units else [(mat(a, bench), mat(b, bench)) for bench in unseen_benches]
    if A is None or B is None or any(x is None or y is None for x, y in pairs):
        return None
    resample = A.shape[0] > 1 or B.shape[0] > 1        # a single run of both methods: nothing to resample over seeds
    test_set = exp["datasets"].index(ds)                # the settings on the same test set share its documents
    out = {}
    for kind, (x, y) in (("test", (A, B)), ("lenient", (mat(a, ds, lenient), mat(b, ds, lenient)))):
        out[kind] = {"diff": float(x.mean() - y.mean()),
                     "seed_diffs": stats.same_seed_differences(x.mean(axis=1), y.mean(axis=1))}
        if resample:
            out[kind]["draws"] = stats.two_level_draws(
                x, y, res.documents(ds), documents=(st["seed"], STREAM[kind], 0, test_set),
                seeds=(st["seed"], STREAM[kind], 1, k), resamples=st["resamples"])
    if units:
        return out
    per_seed = [np.mean([side.mean(axis=1) for side in sides], axis=0) for sides in zip(*pairs)]
    out["unseen"] = {"diff": float(np.mean([x.mean() - y.mean() for x, y in pairs])),
                     "seed_diffs": stats.same_seed_differences(*per_seed)}
    if resample:
        out["unseen"]["draws"] = stats.two_level_draws_benchmarks(
            pairs, questions=(st["seed"], STREAM["unseen"], 0), seeds=(st["seed"], STREAM["unseen"], 1, k),
            resamples=st["resamples"])
    return out


def pool(cells: list[dict | None], kind: str) -> dict | None:
    """The mean over the settings, its interval and the count of positive same-seed differences."""
    if any(c is None for c in cells):
        return None
    items = [c[kind] for c in cells]
    draws = [x["draws"] for x in items] if all("draws" in x for x in items) else None
    return stats.pooled([x["diff"] for x in items], draws, [x["seed_diffs"] for x in items])


def collect(seeds: list[int] | None = None) -> dict:
    """Every number of the report. `seeds`: use only the runs with these seeds (default: the seeds of each method)."""
    res = Results()
    exp = config.experiment()
    out: dict = {"settings": [f"{m}/{ds}" for m, ds in config.SETTINGS], "seeds": seeds or exp["seeds"],
                 "questions": {b: int(len(res.questions[f"{b}/ids"])) for b in [*exp["datasets"], *exp["unseen"]]},
                 "methods": {}, "seed17": {}, "differences": {}, "units_differences": {}, "threshold_differences": {}}
    for method in ROWS:
        out["methods"][method] = [method_row(res, m, ds, method, seeds) for m, ds in config.SETTINGS]
    for method in THRESHOLDS:
        out["seed17"][method] = [method_row(res, m, ds, method, seeds=[THRESHOLD_SEED]) for m, ds in config.SETTINGS]
    cells = {pair: [None] * len(config.SETTINGS) for pair in COMPARISONS}
    mapped = {pair: [None] * len(config.SETTINGS) for pair in COMPARISONS}
    for k, (m, ds) in sorted(enumerate(config.SETTINGS), key=lambda x: x[1][1]):   # by test set: weights are cached
        for a, b in COMPARISONS:
            cells[a, b][k] = compare(res, k, m, ds, a, b, seeds=seeds)
            mapped[a, b][k] = compare(res, k, m, ds, a, b, seeds=seeds, units=True)
    for a, b in COMPARISONS:
        entry = {"settings": [None if c is None else {kind: {x: v for x, v in c[kind].items() if x != "draws"}
                                                      for kind in c} for c in cells[a, b]]}
        entry.update({kind: pool(cells[a, b], kind) for kind in ("test", "lenient", "unseen")})
        out["differences"][f"{a}-{b}"] = entry
        out["units_differences"][f"{a}-{b}"] = {kind: pool(mapped[a, b], kind) for kind in ("test", "lenient")}
    for method in THRESHOLDS:
        if method == "ftrim":
            continue
        one_run = [compare(res, k, m, ds, method, "ftrim", seeds=[THRESHOLD_SEED])
                   for k, (m, ds) in enumerate(config.SETTINGS)]
        out["threshold_differences"][method] = {
            "settings": one_run, **{kind: pool(one_run, kind) for kind in ("test", "unseen")}}
    for name in ("teachers", "mechanism"):
        path = config.RESULTS / f"{name}.json"
        out[name] = read_json(path) if path.exists() else None
    return out


# --------------------------------------------------------------------------------------------------------- formatting
def num(x: float | None, nd: int = 2, sign: bool = False) -> str:
    return "–" if x is None else f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(d: dict | None, nd: int = 2) -> str:
    """A difference with its interval, when it has one."""
    if d is None:
        return "–"
    return f"{d['diff']:+.{nd}f}" + (f" [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}]" if "ci95" in d else "")


def table(head: list[str], rows: list[list[str]], align: str | None = None) -> list[str]:
    """A markdown table; `align` holds one of l, r, c per column (default: first left, others right)."""
    marks = {"l": ":--", "r": "--:", "c": ":-:"}
    align = align or "l" + "r" * (len(head) - 1)
    return ["| " + " | ".join(head) + " |", "|" + "|".join(marks[a] for a in align) + "|",
            *("| " + " | ".join(r) + " |" for r in rows), ""]


def difference_name(a: str, b: str) -> str:
    name = f"{SHORT[a]} − {SHORT[b]}"
    return name[0].upper() + name[1:]


def _mean(values: list) -> float | None:
    return None if any(v is None for v in values) else float(np.mean(values))


def _field(rows: list[dict | None], field: str) -> list:
    return [None if r is None else r[field] for r in rows]


def _heads() -> list[str]:
    return [HEADS[s] for s in config.SETTINGS]


def _bold(text: str, on: bool) -> str:
    return f"**{text}**" if on else text


def _best(values: list, pick) -> float | None:
    """The highest or lowest of the values that exist."""
    values = [v for v in values if v is not None]
    return pick(values) if values else None


def _with_sd(value: float, sd: float | None, sign: bool = False) -> str:
    return num(value, 2, sign) + (f" ±{sd:.2f}" if sd is not None else "")


def _edit(t: dict, setting: tuple[str, str], name: str) -> dict | None:
    teachers = (t["teachers"] or {}).get("settings", {})
    return teachers.get("/".join(setting), {}).get("edits", {}).get(name)


# -------------------------------------------------------------------------------------------------------------- tables
# The README shows the first four tables (scripts/update_docs.py); RESULTS.md shows all of them.
def target_table(t: dict) -> list[str]:
    """Table 1, target tasks: strict EM per setting (mean ± sd over the seeds), its mean and the output length."""
    methods = t["methods"]
    means = {x: _mean(_field(methods[x], "test")) for x in TABLE}
    tokens = {x: _mean(_field(methods[x], "tokens")) for x in TABLE}
    best = [_best([_field(methods[x], "test")[k] for x in SELF_TRAINING], max) for k in range(len(config.SETTINGS))]
    best_mean = _best([means[x] for x in SELF_TRAINING], max)
    shortest = _best([tokens[x] for x in SELF_TRAINING], min)
    rows = []
    for name in TABLE:
        mine = name in SELF_TRAINING
        acc = ["–" if c is None else _bold(_with_sd(c["test"], c["test_sd"]), mine and c["test"] == best[k])
               for k, c in enumerate(methods[name])]
        rows.append([_bold(NAMES[name], name == "ftrim"), *acc,
                     _bold(num(means[name]), mine and means[name] is not None and means[name] == best_mean),
                     _bold(num(tokens[name], 0), mine and tokens[name] is not None and tokens[name] == shortest)])
    return table(["Method", *_heads(), "Mean", "Tokens"], rows)


def generalization_table(t: dict) -> list[str]:
    """Table 1, generalization benchmarks: the accuracy of the untrained model, then the change from it."""
    base = t["methods"]["untrained"]
    rows = [[NAMES["untrained"] + " (accuracy)", *[num(x) for x in _field(base, "unseen_mean")]]]
    for name in TABLE[1:]:
        changes = ["–" if c is None or b is None
                   else _with_sd(c["unseen_mean"] - b["unseen_mean"], c["unseen_sd"], True)
                   for c, b in zip(t["methods"][name], base)]
        rows.append([_bold(NAMES[name], name == "ftrim"), *changes])
    return table(["Method", *_heads()], rows)


def gains_table(t: dict, pairs: list[tuple[str, str]] = GAINS, *, full: bool = False) -> list[str]:
    """Differences in target accuracy per setting and pooled; `full` adds the last-tag EM and the generalization."""
    rows = []
    for a, b in pairs:
        d = t["differences"][f"{a}-{b}"]
        row = [difference_name(a, b), *[num(c and c["test"]["diff"], 2, True) for c in d["settings"]], ci(d["test"])]
        rows.append(row + ([ci(d["lenient"]), ci(d["unseen"])] if full else []))
    head = ["Difference (points)", *_heads(), "Pooled"]
    return table(head + (["Last answer tag only", "Generalization accuracy"] if full else []), rows)


def _threshold_values(t: dict, name: str) -> list[tuple[str, list, int]]:
    """(title, value per setting, decimals) of one threshold: what it deletes and the accuracy of its run."""
    cells = t["seed17"][name]
    edits = [_edit(t, s, name) for s in config.SETTINGS]
    return [("Reasoning tokens deleted (%)", [e and 100 * e["share_of_reasoning"] for e in edits], 1),
            ("Deleted lines with format words (%)", [e and e["format_words_pct"] for e in edits], 0),
            ("Target accuracy", _field(cells, "test"), 2),
            ("Generalization accuracy", _field(cells, "unseen_mean"), 2),
            ("Tokens", _field(cells, "tokens"), 0)]


def threshold_table(t: dict) -> list[str]:
    """The thresholds of FTrim, means of the settings: the deleted share and the accuracy of the run."""
    rows = [[label, *[num(_mean(values), nd) for _, values, nd in _threshold_values(t, name)]]
            for name, label in THRESHOLD_LABELS.items()]
    return table(["Threshold θ", *[title for title, _, _ in _threshold_values(t, "ftrim")]], rows)


def threshold_by_setting(t: dict) -> list[str]:
    rows = []
    for name, label in THRESHOLD_LABELS.items():
        for i, (title, values, nd) in enumerate(_threshold_values(t, name)[:4]):
            rows.append([label if i == 0 else "", title, *[num(x, nd) for x in values], num(_mean(values), nd)])
    return table(["Threshold θ", "", *_heads(), "Mean"], rows, "ll" + "r" * (len(config.SETTINGS) + 1))


def normalization_table(t: dict) -> list[str]:
    """Rationalization, then the answer normalization alone, then FTrim (normalization and format-talk deletion)."""
    rows = []
    for name, label in (("rationalization", NAMES["rationalization"]), ("normalized", NAMES["normalized"]),
                        ("ftrim", "**FTrim**: normalization and format-talk deletion")):
        cells = t["methods"][name]
        rows.append([label, *[num(x) for x in _field(cells, "test")], num(_mean(_field(cells, "test"))),
                     num(_mean(_field(cells, "tokens")), 0)])
    return table(["Fine-tuned on", *_heads(), "Mean", "Tokens"], rows)


def deleted_table(t: dict) -> list[str]:
    pools = [t["teachers"]["settings"].get("/".join(s)) for s in config.SETTINGS]
    rows = []
    for name, label in (("ftrim", "FTrim"), ("random_deletion", "Random deletion")):
        edits = [_edit(t, s, name) for s in config.SETTINGS]
        rows.append([f"{label}: reasoning tokens deleted (%)",
                     *[num(e and 100 * e["share_of_reasoning"], 1) for e in edits]])
        rows.append([f"{label}: deleted lines with format words (%)",
                     *[num(e and e["format_words_pct"], 0) for e in edits]])
    rows.append(["All reasoning lines with format words (%)",
                 *[num(x and x["all_reasoning_lines"]["format_words_pct"], 0) for x in pools]])
    return table(["", *_heads()], rows)


def format_talk_table(t: dict) -> list[str]:
    talk = [t["mechanism"]["format_talk"].get("/".join(s)) for s in config.SETTINGS]
    return table(["Test outputs of the untrained model", *_heads()], [
        ["Outputs with format talk (%)", *[num(x and x["with_format_talk_pct"], 0) for x in talk]],
        ["Reasoning tokens in format talk (%)", *[num(x and x["token_share_pct"], 1) for x in talk]]])


MEASURES = (("format_talk_line_dll", "Log-likelihood of a format-talk line"),
            ("other_line_dll", "Log-likelihood of another line"),
            ("skip_format_talk", "Skip of format talk"), ("skip_random", "Skip of random lines"))


def mechanism_table(t: dict) -> list[str]:
    """What the models learn: means of the settings for the first seed of the analysis (Fig. 2 of the paper)."""
    entry = next(iter(t["mechanism"]["probes"].values()))
    pooled = entry["pooled"]["methods"]
    rows = [[_bold(NAMES[name], name == "ftrim"), *[num(pooled[name][key], 1, True) for key, _ in MEASURES]]
            for name in config.experiment()["mechanism"]["methods"] if name in pooled]
    return table(["Fine-tuned with", *[title for _, title in MEASURES]], rows)


def simple_format_rows(t: dict) -> list[list[str]]:
    """The datasets with simple answer formats, and TAT-QA with the same model for comparison."""
    simple = (t["teachers"] or {}).get("simple_formats") or {}
    settings = (t["teachers"] or {}).get("settings", {})
    rows = []
    for m in dict.fromkeys(key.split("/")[0] for key in simple):
        for key, v in simple.items():
            if key.startswith(f"{m}/"):
                rows.append([config.model(m)["name"], SIMPLE[key.split("/")[1]], f"{v['teachers']:,}",
                             f"{100 * v['ftrim_share']:.2f}",
                             f"{100 * v['teachers_with_format_talk'] / v['teachers']:.1f}"])
        tat = settings.get(f"{m}/tatqa")
        if tat:
            e, n = tat["edits"]["ftrim"], tat["pools"]["normalized"]
            rows.append([config.model(m)["name"], "TAT-QA (Table 1)", f"{n:,}", f"{100 * e['share_before_check']:.2f}",
                         f"{100 * (e['edited'] + e['kept_by_check']) / n:.1f}"])
    return rows


def simple_formats_table(t: dict) -> list[str]:
    return table(["Model", "Dataset", "Teachers", "Reasoning tokens that FTrim deletes (%)",
                  "Teachers with a deleted line (%)"], simple_format_rows(t), "llrrr")


def units_table(t: dict) -> list[str]:
    """The pooled differences with TAT-QA scored as in the paper's runs and with the units mapped."""
    rows = []
    for a, b in GAINS:
        d, u = t["differences"][f"{a}-{b}"], t["units_differences"][f"{a}-{b}"]
        rows.append([difference_name(a, b), ci(d["test"]), ci(u["test"]), ci(d["lenient"]), ci(u["lenient"])])
    return table(["Difference (points)", "Target accuracy", "Target accuracy, units mapped", "Last answer tag only",
                  "Last answer tag only, units mapped"], rows)


# ------------------------------------------------------------------------------------------------------------ sections
def _intro(t: dict) -> list[str]:
    seeds = t["seeds"]
    runs = (f"means over the seeds {', '.join(map(str, seeds[:-1]))} and {seeds[-1]}" if len(seeds) > 1
            else f"one run with seed {seeds[0]}")
    try:                                          # the results directory, relative to the repository when inside
        where = (config.RESULTS / "per_question").resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        where = str(config.RESULTS / "per_question")
    unseen = [BENCH[b] for b in config.experiment()["unseen"]]
    return ["# Results", "",
            f"Generated by `python -m ftrim report` from the per-question results in `{where}/`. "
            "Do not edit by hand.", "",
            "A *setting* is one model fine-tuned on one target task: Qwen3.5-4B or Gemma-4-E4B-it on TAT-QA or DROP "
            "(Qwen-TAT, Qwen-DROP, Gemma-TAT, Gemma-DROP). *Target accuracy* is the strict exact match on the test "
            f"set of the target task. *Generalization accuracy* is the mean accuracy on {', '.join(unseen[:-1])} and "
            f"{unseen[-1]}, which are never used for training or selection. docs/DATA.md defines both. All values "
            f"are percentages unless noted. Fine-tuned methods are {runs}; the untrained rows are one run. Evaluation "
            "uses greedy decoding, except the thinking mode: one sample per question with the sampling settings of "
            "the model card.", "",
            "`tables.json` holds these numbers and more: every seed, each generalization benchmark and the "
            "statistics of the outputs.", ""]


def _target(t: dict) -> list[str]:
    return ["## Target accuracy", "",
            "Table 1 of the paper, target tasks. Mean ± sample standard deviation over the seeds. *Mean*: mean of the "
            "four settings. *Tokens*: mean output length on the target test set, mean of the four settings (thinking "
            "mode: the thinking included). Bold: best self-training row.", "", *target_table(t)]


def _generalization(t: dict) -> list[str]:
    return ["## Generalization accuracy", "",
            "Table 1 of the paper, generalization benchmarks. The first row is the accuracy of the untrained model. "
            "The other rows are the change from it, ± the sample standard deviation over the seeds.", "",
            *generalization_table(t)]


def _gains(t: dict) -> list[str]:
    resamples = config.experiment()["statistics"]["resamples"]
    return ["## Gains of FTrim", "",
            "Differences in target accuracy, in points. *Pooled*: the mean of the four settings, with the 95 % "
            f"interval of {resamples:,} bootstrap samples over the seeds and over the tables or passages of the test "
            "set (`src/ftrim/stats.py`). *Last answer tag only*: the pooled difference when the output is cut at its "
            "last answer tag before scoring (last-tag EM in docs/DATA.md). *Generalization accuracy*: the pooled "
            "difference on the generalization benchmarks.", "",
            *gains_table(t, [*GAINS, ("random_deletion", "rationalization")], full=True)]


def _threshold(t: dict) -> list[str]:
    return ["## Threshold", "",
            "FTrim deletes a reasoning line when its score m reaches the threshold θ. The paper uses 1 bit. Each "
            f"threshold has one run with seed {THRESHOLD_SEED}, so the row of 1 bit is the run of FTrim with this "
            "seed. *Reasoning tokens deleted*: share of the reasoning tokens of the teachers. *Deleted lines with "
            "format words*: see \"What FTrim deletes\".", "", *threshold_by_setting(t)]


def _normalization(t: dict) -> list[str]:
    return ["## Normalization only", "",
            "FTrim normalizes the answer and deletes format talk. *Normalization only* fine-tunes on the normalized "
            "teachers without any deletion, with the same seeds. Random deletion and STM edit the rationalization "
            "teachers, without the normalization.", "", *normalization_table(t),
            *gains_table(t, [("ftrim", "normalized"), ("normalized", "rationalization")])]


def _deleted(t: dict) -> list[str]:
    settings = (t["teachers"] or {}).get("settings", {})
    if not settings:
        return []
    out = ["## What FTrim deletes", "",
           "A *format word* is format, unit, scale, digit, JSON, array, instruction, required, rule, verbatim, "
           "decimal, round or tag as a whole word, also in plural or verb forms, or `<answer>`, \"answer tag\", "
           "\"word for word\" or \"day month year\" (`FORMAT_WORDS` in `src/ftrim/results.py`). FTrim does not use "
           "these words. They only describe the lines that its rule selects.", "", *deleted_table(t),
           "The lines that FTrim deletes most often, with numbers replaced by N and the share of the edited "
           "teachers that contain the line:", ""]
    for s in config.SETTINGS:
        e = _edit(t, s, "ftrim")
        if e:
            out.append(f"* {HEADS[s]}: " + "; ".join(f"`{line[:70]}` {pct:.1f} %" for line, pct in e["top_lines"][:3]))
    kept = sum(_edit(t, s, "ftrim")["kept_by_check"] for s in config.SETTINGS if _edit(t, s, "ftrim"))
    total = sum(x["pools"]["normalized"] for x in settings.values())
    return out + ["", "FTrim keeps a teacher unedited when the edited teacher fails the check of line 8 of "
                  f"Algorithm 1 (exactly one answer marker, and the verifier accepts): {kept} of {total:,} teachers.",
                  ""]


def _figure() -> list[str]:
    """Fig. 2 of the paper; it shows the paper's runs, so only the report on results/ includes it."""
    if config.RESULTS.resolve() != (config.ROOT / "results").resolve():
        return []
    return ["<p align=\"center\">", "  <img src=\"../assets/skip.png\" width=\"640\"",
            "       alt=\"Fig. 2 of the paper: the change of the log-odds of skipping a line against the untrained "
            "model, at the first format-talk line and at random other lines. The table below has the values.\">",
            "</p>", ""]


def _mechanism(t: dict) -> list[str]:
    mech = t["mechanism"]
    if not mech or not mech.get("format_talk"):
        return []
    out = ["## What the models learn", "",
           "Sect. 5 and Fig. 2 of the paper. Format talk in the test outputs of the untrained models, found with "
           "FTrim's rule:", "", *format_talk_table(t)]
    probes = mech.get("probes") or {}
    if not probes:
        return out
    seed, entry = next(iter(probes.items()))
    outputs = ", ".join(f"{HEADS[s]} {entry['settings']['/'.join(s)]['outputs']:,}" for s in config.SETTINGS
                        if "/".join(s) in entry["settings"])
    out += ["The fine-tuned models score these outputs three times: as written, without their format-talk lines, "
            "and without random other lines of the same length. All such outputs are used on TAT-QA, and a seeded "
            f"random sample on DROP ({outputs}). The table gives the change against the untrained model for the "
            f"models with seed {seed[1:]}, as the mean of the settings, in nats. *Log-likelihood*: of a whole line. "
            "*Skip*: the log-odds that the model writes the token after a line instead of the line's first token.",
            "", *_figure(), *mechanism_table(t)]
    others = [e["pooled"]["methods"] for e in list(probes.values())[1:]]
    if others:
        first = entry["pooled"]["methods"]
        gap = max(abs(first[x][k] - other[x][k]) for other in others for x in first if x != "untrained" and x in other
                  for k in ("format_talk_line_dll", "skip_format_talk", "skip_random"))
        names = " and ".join(s[1:] for s in list(probes)[1:])
        out += [f"The models with seed {names} agree with these values within {np.ceil(gap * 10) / 10:.1f} nats.", ""]
    return out


def _simple_formats(t: dict) -> list[str]:
    if not simple_format_rows(t):
        return []
    return ["## Simple answer formats", "",
            "Format talk is rarer when the answer is a single number, expression or option. This repository scores "
            "the teachers of such datasets only, with the threshold of 1 bit, and TAT-QA with the same model is "
            "given for comparison.", "", *simple_formats_table(t)]


def _units(t: dict) -> list[str]:
    tat = [k for k, (_, ds) in enumerate(config.SETTINGS) if ds == "tatqa"]
    if not tat or not any(c and "test_units" in c for c in t["methods"]["ftrim"]):
        return []
    best = []
    for k, c in enumerate(t["methods"]["ftrim"]):
        field = "test_units" if k in tat else "test"
        if all(t["methods"][x][k] for x in SELF_TRAINING):
            best.append(all(c[field] >= t["methods"][x][k][field] for x in SELF_TRAINING))
    return ["## Scoring of the TAT-QA units", "",
            "The paper's runs count an answer as wrong if its unit is one of the five allowed units without a TAT-QA "
            "scale: `ratio`, `percentage_point`, `basis_point`, `hundred` and `trillion` (docs/DATA.md, "
            "\"Evaluators\"). *Units mapped* scores the same test outputs with `ratio` read as no scale and "
            "`percentage_point` as percent, as the official evaluator reads these names. Pooled differences:", "",
            *units_table(t),
            *([f"With the units mapped, FTrim has the highest target accuracy of the self-training rows in "
               f"{sum(best)} of {len(best)} settings.", ""] if best else [])]


def markdown(t: dict) -> str:
    sections = (_intro, _target, _generalization, _gains, _threshold, _normalization, _deleted, _mechanism,
                _simple_formats, _units)
    return "\n".join(line for section in sections for line in section(t)).rstrip("\n") + "\n"


def _without_draws(x):
    """The tables without the bootstrap draws (they are not stored)."""
    if isinstance(x, dict):
        return {k: _without_draws(v) for k, v in x.items() if k != "draws"}
    return [_without_draws(v) for v in x] if isinstance(x, list) else x


def load_tables() -> dict:
    """The tables of the last `python -m ftrim report` (results/tables.json), with the teacher and mechanism results."""
    tables = read_json(config.RESULTS / "tables.json")
    for name in ("teachers", "mechanism"):
        path = config.RESULTS / f"{name}.json"
        tables[name] = read_json(path) if path.exists() else None
    return tables


def write(*, check: bool = False, seeds: list[int] | None = None) -> dict:
    tables = _without_draws(collect(seeds))
    write_json(config.RESULTS / "tables.json", {k: v for k, v in tables.items() if k not in ("teachers", "mechanism")})
    (config.RESULTS / "RESULTS.md").write_text(markdown(tables), encoding="utf-8")
    result: dict = {"written": [os.path.relpath(config.RESULTS / name) for name in ("RESULTS.md", "tables.json")]}
    if check:
        from . import paper
        try:
            different = paper.check(tables)
        except (KeyError, TypeError) as e:
            raise SystemExit(f"--check needs the complete results of the paper ({e!r} is missing)") from e
        n = paper.counts()
        result.update(paper_values_checked=n["printed"], statement_values_checked=n["backing"], different=different)
        if different:
            raise SystemExit("the results differ from the paper:\n  " + "\n  ".join(different))
    return result
