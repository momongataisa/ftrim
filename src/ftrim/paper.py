"""The numbers of the paper, checked against the tables of `python -m ftrim report` and configs/experiment.yaml.

`python -m ftrim report --check` fails if a value of the repository, rounded as in the paper, differs from the
paper. An entry is (where, value of the repository, value in the paper, decimals). A list holds one value per setting
in the order Qwen-TAT, Qwen-DROP, Gemma-TAT, Gemma-DROP, unless its label says otherwise.

* `printed`: every number of the six-page paper that follows from results/ or from configs/experiment.yaml.
* `backing`: values of this repository behind statements that the paper makes in words. A comparison is checked
  as true (1) or false (0).

Not checked: the share of the GPU time that FTrim's scoring takes (Sect. 3). It comes from the job logs of the
paper's runs, which are not released (docs/REPRODUCTION.md#compute).
"""
from __future__ import annotations

import numpy as np

from . import config

SELF_TRAINING = ("rejection", "rationalization", "random_deletion", "stm", "ftrim")
TAT, DROP = (0, 2), (1, 3)                       # the settings of each target task
UNSEEN = ("finance_reasoning", "gsm8k", "ifeval", "arc_challenge")


def _field(rows: list[dict], field: str) -> list[float]:
    return [r[field] for r in rows]


def _change(t: dict, method: str) -> list[float]:
    """The change of the mean accuracy on the unseen benchmarks from the untrained model, per setting."""
    return [r["unseen_mean"] - b["unseen_mean"] for r, b in zip(t["methods"][method], t["methods"]["untrained"])]


def _pooled(t: dict, pair: str, kind: str = "test") -> list[float]:
    d = t["differences"][pair][kind]
    return [d["diff"], *d["ci95"]]


def _low_high(values) -> list[float]:
    values = list(values)
    return [min(values), max(values)]


def _settings_where(t: dict, better) -> int:
    """The number of settings in which FTrim beats every other self-training row under `better(ftrim, other)`."""
    m = t["methods"]
    return sum(all(better(m["ftrim"][k], m[x][k]) for x in SELF_TRAINING if x != "ftrim") for k in range(4))


def printed(t: dict) -> list[tuple[str, object, object, int]]:
    m, d = t["methods"], t["differences"]
    exp = config.experiment()
    edits = [t["teachers"]["settings"][s]["edits"] for s in t["settings"]]
    teachers = [t["teachers"]["settings"][s] for s in t["settings"]]
    gd_base, gd_ftrim = m["untrained"][3]["unseen"], m["ftrim"][3]["unseen"]       # Gemma-DROP
    out = [
        # ---- Table 1: strict EM, standard deviations, generalization benchmarks, output length
        ("Table 1, Untrained, EM", _field(m["untrained"], "test"), [62.06, 84.38, 57.37, 82.30], 2),
        ("Table 1, + Thinking mode, EM", _field(m["thinking"], "test"), [72.16, 86.87, 61.15, 80.58], 2),
        ("Table 1, SFT, EM", _field(m["sft"], "test"), [74.40, 80.62, 63.32, 77.22], 2),
        ("Table 1, rejection sampling, EM", _field(m["rejection"], "test"), [70.35, 86.40, 61.35, 82.96], 2),
        ("Table 1, + Rationalization, EM", _field(m["rationalization"], "test"), [72.34, 86.44, 66.93, 84.05], 2),
        ("Table 1, random deletion, EM", _field(m["random_deletion"], "test"), [72.60, 86.43, 67.13, 84.46], 2),
        ("Table 1, STM, EM", _field(m["stm"], "test"), [71.50, 86.55, 67.11, 83.37], 2),
        ("Table 1, FTrim, EM", _field(m["ftrim"], "test"), [76.09, 86.71, 69.91, 84.70], 2),
        ("Table 1, SFT, EM sd", _field(m["sft"], "test_sd"), [0.40, 0.13, 0.06, 0.30], 2),
        ("Table 1, rejection sampling, EM sd", _field(m["rejection"], "test_sd"), [0.52, 0.23, 0.38, 0.08], 2),
        ("Table 1, + Rationalization, EM sd", _field(m["rationalization"], "test_sd"), [0.16, 0.05, 0.74, 0.20], 2),
        ("Table 1, random deletion, EM sd", _field(m["random_deletion"], "test_sd"), [0.51, 0.12, 0.28, 0.06], 2),
        ("Table 1, STM, EM sd", _field(m["stm"], "test_sd"), [0.26, 0.12, 0.47, 0.17], 2),
        ("Table 1, FTrim, EM sd", _field(m["ftrim"], "test_sd"), [0.43, 0.26, 0.47, 0.05], 2),
        ("Table 1, Untrained, generalization", _field(m["untrained"], "unseen_mean"), [87.26, 87.26, 87.76, 87.76], 2),
        ("Table 1, + Thinking mode, generalization change", _change(t, "thinking"), [3.26, 3.26, 0.19, 0.19], 2),
        ("Table 1, SFT, generalization change", _change(t, "sft"), [-27.82, -30.11, -15.61, -33.37], 2),
        ("Table 1, rejection sampling, generalization change", _change(t, "rejection"), [0.09, -0.58, -0.21, -0.27], 2),
        ("Table 1, + Rationalization, generalization change", _change(t, "rationalization"),
         [-0.23, -0.49, -0.48, -0.34], 2),
        ("Table 1, random deletion, generalization change", _change(t, "random_deletion"),
         [-0.27, -0.59, -0.61, -1.02], 2),
        ("Table 1, STM, generalization change", _change(t, "stm"), [-0.26, -0.29, -0.44, -0.56], 2),
        ("Table 1, FTrim, generalization change", _change(t, "ftrim"), [0.48, -0.21, -0.21, -0.82], 2),
        ("Table 1, SFT, generalization sd", _field(m["sft"], "unseen_sd"), [0.24, 0.27, 0.35, 0.28], 2),
        ("Table 1, rejection sampling, generalization sd", _field(m["rejection"], "unseen_sd"),
         [0.21, 0.12, 0.19, 0.31], 2),
        ("Table 1, + Rationalization, generalization sd", _field(m["rationalization"], "unseen_sd"),
         [0.08, 0.27, 0.31, 0.24], 2),
        ("Table 1, random deletion, generalization sd", _field(m["random_deletion"], "unseen_sd"),
         [0.28, 0.31, 0.10, 0.34], 2),
        ("Table 1, STM, generalization sd", _field(m["stm"], "unseen_sd"), [0.19, 0.32, 0.25, 0.19], 2),
        ("Table 1, FTrim, generalization sd", _field(m["ftrim"], "unseen_sd"), [0.14, 0.19, 0.26, 0.08], 2),
        ("Table 1, Tok. of the eight rows (mean of the settings)",
         [float(np.mean(_field(m[x], "tokens"))) for x in ("untrained", "thinking", "sft", "rejection",
                                                           "rationalization", "random_deletion", "stm", "ftrim")],
         [426, 2459, 17, 391, 364, 341, 380, 326], 0),
        # ---- Abstract, Sect. 1, 4 and 6
        ("Abstract, Sect. 1, 4 and 6, settings where FTrim has the highest strict EM of the self-training rows",
         _settings_where(t, lambda a, b: a["test"] > b["test"]), 4, 0),
        ("Abstract, Sect. 1, 4 and 6, settings where FTrim has the shortest outputs of the self-training rows",
         _settings_where(t, lambda a, b: a["tokens"] < b["tokens"]), 4, 0),
        ("Abstract and Sect. 1, FTrim's generalization changes by less than one point in every setting",
         max(abs(x) for x in _change(t, "ftrim")) < 1, True, 0),
        # ---- Sect. 1
        ("Sect. 1, untrained test outputs with format talk, lowest and highest (%)",
         _low_high(t["mechanism"]["format_talk"][s]["with_format_talk_pct"] for s in t["settings"]), [30, 67], 0),
        ("Sect. 1, pooled strict EM, FTrim - rationalization and FTrim - random deletion",
         [d["ftrim-rationalization"]["test"]["diff"], d["ftrim-random_deletion"]["test"]["diff"]], [1.9, 1.7], 1),
        # ---- Sect. 3, models and data
        ("Sect. 3, training questions", _field(teachers, "questions"), [13215, 10006, 13215, 10006], 0),
        ("Sect. 3, test questions of TAT-QA and DROP", [t["questions"]["tatqa"], t["questions"]["drop"]],
         [1663, 9536], 0),
        # ---- Sect. 3, format instructions and FTrim
        ("Sect. 3, sampling: K, temperature, top-p, token limit",
         [exp["sampling"][k] for k in ("k", "temperature", "top_p", "max_new_tokens")], [4, 0.7, 0.9, 8192], 1),
        ("Sect. 3, thresholds in bits: FTrim, then 0.5, 2 and zero",
         [exp["edits"][x]["threshold_bits"] for x in ("ftrim", "ftrim_half_bit", "ftrim_two_bits", "ftrim_zero")],
         [1, 0.5, 2, 0], 1),
        ("Sect. 3, reasoning tokens deleted by FTrim (%)", [100 * e["ftrim"]["share_of_reasoning"] for e in edits],
         [13.0, 2.1, 7.1, 6.0], 1),
        # ---- Sect. 3, training, comparisons, statistics
        ("Sect. 3, LoRA rank and alpha", [exp["training"]["lora"]["r"], exp["training"]["lora"]["alpha"]], [16, 32], 0),
        ("Sect. 3, weight decay, gradient clipping norm, batch size",
         [exp["training"][k] for k in ("weight_decay", "grad_clip", "batch")], [0, 1, 32], 0),
        ("Sect. 3, learning rate", exp["training"]["lr"], 2e-5, 7),
        ("Sect. 3, passes of the learning-rate schedule", exp["training"]["horizon_factor"], 3, 0),
        ("Sect. 3, seeds", exp["seeds"], [17, 42, 89], 0),
        ("Sect. 3, reasoning tokens deleted by random deletion (%)",
         [100 * e["random_deletion"]["share_of_reasoning"] for e in edits], [13.4, 2.3, 7.8, 6.6], 1),
        ("Sect. 3, perplexity limit of STM", exp["edits"]["stm"]["tau"], 2.5, 1),
        ("Sect. 3, bootstrap samples", exp["statistics"]["resamples"], 20000, 0),
        # ---- Sect. 4, accuracy
        ("Sect. 4, FTrim - rationalization per setting",
         [c["test"]["diff"] for c in d["ftrim-rationalization"]["settings"]], [3.75, 0.27, 2.99, 0.65], 2),
        ("Sect. 4, FTrim - rationalization, pooled", _pooled(t, "ftrim-rationalization"), [1.91, 1.36, 2.48], 2),
        ("Sect. 4, untrained outputs that follow the format on TAT-QA, lowest and highest (%)",
         _low_high(m["untrained"][k]["format_valid"] for k in TAT), [86, 88], 0),
        ("Sect. 4, untrained outputs that follow the format on DROP, lowest and highest (%)",
         _low_high(m["untrained"][k]["format_valid"] for k in DROP), [98.6, 99.9], 1),
        ("Sect. 4, random deletion - rationalization, pooled", _pooled(t, "random_deletion-rationalization"),
         [0.21, -0.26, 0.68], 2),
        ("Sect. 4, FTrim - random deletion, pooled", _pooled(t, "ftrim-random_deletion"), [1.70, 1.13, 2.28], 2),
        ("Sect. 4, FTrim - STM, pooled", _pooled(t, "ftrim-stm"), [2.22, 1.66, 2.80], 2),
        ("Sect. 4, last answer tag only, FTrim - random deletion, pooled",
         _pooled(t, "ftrim-random_deletion", "lenient"), [1.59, 1.01, 2.17], 2),
        ("Sect. 4, last answer tag only, FTrim - STM, pooled", _pooled(t, "ftrim-stm", "lenient"), [1.82, 1.25, 2.40], 2),
        # ---- Sect. 4, general abilities in each setting
        ("Sect. 4, largest change of a self-training row on the generalization benchmarks",
         max(abs(c) for x in SELF_TRAINING for c in _change(t, x)), 1.02, 2),
        ("Sect. 4, SFT's loss on the generalization benchmarks, lowest and highest",
         _low_high(-c for c in _change(t, "sft")), [15.61, 33.37], 2),
        ("Sect. 4, FTrim's largest loss on the generalization benchmarks, and its setting (index)",
         [max(-c for c in _change(t, "ftrim")), int(np.argmin(_change(t, "ftrim")))], [0.82, 3], 2),
        ("Sect. 4, settings where every seed of FTrim loses on the generalization benchmarks (index)",
         [k for k, (r, b) in enumerate(zip(m["ftrim"], m["untrained"]))
          if all(s < b["unseen_mean"] for s in r["unseen_seeds"])], [3], 0),
        ("Sect. 4, IFEval on Gemma-DROP, untrained and FTrim", [gd_base["ifeval"], gd_ftrim["ifeval"]], [86.1, 84.7], 1),
        ("Sect. 4, the fall of the other three benchmarks on Gemma-DROP, lowest and highest",
         _low_high(gd_base[b] - gd_ftrim[b] for b in UNSEEN if b != "ifeval"), [0.5, 0.7], 1),
        # ---- Sect. 4, thinking and length
        ("Sect. 4, tokens of the thinking mode as a multiple of FTrim's, lowest and highest",
         _low_high(r["tokens"] / x["tokens"] for r, x in zip(m["thinking"], m["ftrim"])), [2.3, 12.8], 1),
        ("Sect. 4, FTrim - thinking mode in Qwen-TAT, Gemma-TAT and Gemma-DROP, lowest and highest",
         _low_high(d["ftrim-thinking"]["settings"][k]["test"]["diff"] for k in (0, 2, 3)), [3.9, 8.8], 1),
        ("Sect. 4, Qwen-DROP, the thinking mode, then FTrim's lowest and highest seed",
         [m["thinking"][1]["test"], *_low_high(m["ftrim"][1]["test_seeds"])], [86.87, 86.44, 86.97], 2),
        # ---- Sect. 5, what FTrim deletes
        ("Sect. 5, deleted lines with format words, FTrim (%)", [e["ftrim"]["format_words_pct"] for e in edits],
         [90, 55, 79, 77], 0),
        ("Sect. 5, deleted lines with format words, random deletion (%)",
         [e["random_deletion"]["format_words_pct"] for e in edits], [20, 3, 6, 1], 0),
        # ---- Sect. 5, where FTrim helps less
        ("Sect. 5, FTrim - random deletion per setting",
         [c["test"]["diff"] for c in d["ftrim-random_deletion"]["settings"]], [3.49, 0.28, 2.79, 0.24], 2),
        ("Sect. 5, settings with simple answer formats", len(t["teachers"]["simple_formats"]), 5, 0),
    ]
    # ---- Sect. 5 and Fig. 2, what the models learn
    mech = t["mechanism"]
    s17, s42 = (mech["probes"][s]["pooled"] for s in ("s17", "s42"))
    keys = ("format_talk_line_dll", "skip_format_talk", "skip_random")
    gap = max(abs(s17["methods"][x][k] - s42["methods"][x][k]) for x in s17["methods"] if x != "untrained"
              for k in keys)
    out += [
        ("Sect. 5, seeds of the analysis", exp["mechanism"]["seeds"], [17, 42], 0),
        ("Sect. 5, analysed outputs per DROP setting",
         [mech["probes"]["s17"]["settings"][t["settings"][k]]["outputs"] for k in DROP], [2000, 2000], 0),
        ("Sect. 5, seed 42 agrees with seed 17 within (nats)", float(np.ceil(gap * 10) / 10), 0.3, 1),
        ("Sect. 5, log-likelihood of a format-talk line: rationalization, FTrim, STM (nats)",
         [s17["methods"][x]["format_talk_line_dll"] for x in ("rationalization", "ftrim", "stm")], [2.5, -1.2, 4.3], 1),
        ("Sect. 5, settings where FTrim makes format talk less likely",
         sum(c["methods"]["ftrim"]["format_talk_line_dll"] < 0 for c in mech["probes"]["s17"]["settings"].values()),
         4, 0),
        ("Fig. 2, skip of format talk: rationalization, random deletion, STM, FTrim",
         [s17["methods"][x]["skip_format_talk"] for x in ("rationalization", "random_deletion", "stm", "ftrim")],
         [-3.1, 5.8, -8.5, 9.2], 1),
        ("Fig. 2, skip of random other lines: rationalization, random deletion, STM, FTrim",
         [s17["methods"][x]["skip_random"] for x in ("rationalization", "random_deletion", "stm", "ftrim")],
         [-3.8, 7.9, -10.8, 1.7], 1),
    ]
    return out


def backing(t: dict) -> list[tuple[str, object, object, int]]:
    m = t["methods"]
    edits = [t["teachers"]["settings"][s]["edits"] for s in t["settings"]]
    teachers = [t["teachers"]["settings"][s] for s in t["settings"]]
    mech = t["mechanism"]

    def without_words(e: dict) -> float:                # deleted lines without format words
        return e["deleted_lines"] * (100 - e["format_words_pct"]) / 100

    gd_fall = {b: m["untrained"][3]["unseen"][b] - m["ftrim"][3]["unseen"][b] for b in UNSEEN}
    simple = t["teachers"]["simple_formats"]
    return [
        ("Sect. 2, 'Line 8 then keeps y': teachers that line 8 keeps unedited, and all teachers",
         [sum(e["ftrim"]["kept_by_check"] for e in edits), sum(x["pools"]["normalized"] for x in teachers)],
         [1, 41374], 0),
        ("Sect. 3, '0.5 and 2 bits give a similar strict accuracy': strict EM minus that of 1 bit, seed 17",
         [t["threshold_differences"][x]["test"]["diff"] for x in ("ftrim_half_bit", "ftrim_two_bits")],
         [0.16, -0.38], 2),
        ("Sect. 3, '0.5 bits also deletes many lines without format-related words': such lines as a multiple of "
         "those of 1 bit, lowest and highest", _low_high(without_words(e["ftrim_half_bit"]) / without_words(e["ftrim"])
                                                         for e in edits), [2.9, 3.7], 1),
        ("Sect. 3, 'a threshold of zero removes most of the reasoning': reasoning tokens deleted, lowest and "
         "highest (%)", _low_high(100 * e["ftrim_zero"]["share_of_reasoning"] for e in edits), [78, 82], 0),
        ("Sect. 3, the same: test answers without reasoning, mean of the settings (%)",
         float(np.mean(_field(t["seed17"]["ftrim_zero"], "no_reasoning"))), 54, 0),
        ("Sect. 4, 'IFEval falls most' on Gemma-DROP", max(gd_fall, key=gd_fall.get) == "ifeval", True, 0),
        ("Sect. 5, 'all such outputs on TAT-QA': the analysed outputs are all outputs with format talk",
         all(mech["probes"]["s17"]["settings"][t["settings"][k]]["outputs"]
             == round(mech["format_talk"][t["settings"][k]]["outputs"]
                      * mech["format_talk"][t["settings"][k]]["with_format_talk_pct"] / 100) for k in TAT), True, 0),
        ("Sect. 5, 'FTrim deletes much less of the reasoning than on TAT-QA': largest share of the reasoning tokens "
         "on the simple formats, Qwen3.5-4B and Gemma-4-E4B-it (%)",
         [max(100 * v["ftrim_share"] for k, v in simple.items() if k.startswith(f"{model}/")) for model in ("qwen", "gemma")],
         [1.0, 3.0], 1),
    ]


def _same(got, expected, decimals: int) -> bool:
    return abs(round(float(got), decimals) - float(expected)) < 0.5 * 10 ** -decimals


def check(tables: dict) -> list[str]:
    """The entries whose value differs from the paper, as readable lines (empty: everything is reproduced)."""
    failures = []
    for group in (printed, backing):
        n = 0
        for where, got, expected, decimals in group(tables):
            got_list, exp_list = (list(x) if isinstance(x, (list, tuple)) else [x] for x in (got, expected))
            n += len(exp_list)
            if len(got_list) != len(exp_list) or not all(_same(g, e, decimals) for g, e in zip(got_list, exp_list)):
                failures.append(f"{where}: repository {[round(float(g), decimals) for g in got_list]}, "
                                f"paper {exp_list}")
        _COUNTS[group.__name__] = n
    return failures


_COUNTS: dict[str, int] = {}


def counts() -> dict[str, int]:
    """The number of values compared by the last `check`: printed in the paper, and behind statements in words."""
    return dict(_COUNTS)
