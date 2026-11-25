"""Compact results of the finished runs (CPU): all that `python -m ftrim report` needs, without the run directory.

    results/per_question/questions.npz           <benchmark>/ids, <benchmark>/documents
    results/per_question/<model>_untrained.npz   <method>/<benchmark>/<field>            method: untrained, thinking
    results/per_question/<model>_<dataset>.npz   <method>/s<seed>/<benchmark>/<field>    fine-tuned on <dataset>
    results/teachers.json                        the teacher pools and what the edits delete
    results/mechanism.json                       `python -m ftrim mechanism summary`

Every field holds one value per question of the benchmark, in the order of its `ids`:

    correct       1 if the output is correct: strict EM on TAT-QA and DROP, the rule of the benchmark on the others
    lenient       target task only: last-tag EM, 1 if the output cut at its last answer tag is correct
    correct_units, lenient_units
                  TAT-QA only: correct and lenient with the units ratio and percentage_point read as TAT-QA scales
                  (`scoring.MAPPED_UNITS`; the paper's runs count them as wrong)
    format_valid  1 if the output follows the answer format (TAT-QA, DROP, FinanceReasoning; 0 on the other benchmarks)
    tokens        generated tokens
    answer_tag    1 if the output contains an answer tag
    no_reasoning  1 if the output starts with its answer tag
    truncated     1 if the generation stopped at the token limit

`documents` is the index of the table or passage of each question (the unit that the bootstrap resamples).
"""
from __future__ import annotations

import difflib
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, core, data, edits, scoring
from .utils import read_json, read_jsonl, write_json

DELETION_POOLS = ("ftrim", "random_deletion", "ftrim_half_bit", "ftrim_two_bits", "ftrim_zero")
PLAIN_POOLS = ("random_deletion",)              # made on the rationalization teachers, without answer normalization
# Words that mark a line as talk about the answer format (descriptive only: FTrim does not use them). Whole words
# only, so that "information", "around" or "community" do not count.
FORMAT_WORDS = re.compile(r"\b(?:format(?:s|ted|ting)?|units?|scales?|digits?|json|arrays?|instructions?|required|"
                          r"rules?|verbatim|decimals?|round(?:s|ed|ing)?|tags?)\b|<answer>|answer tag|word for word|"
                          r"day month year", re.I)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


# ------------------------------------------------------------------------------------------------------------ npz files
def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """A compressed .npz whose bytes depend only on its arrays (no timestamps), so that an unchanged export leaves the
    repository unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())
    tmp.replace(path)


def per_question_dir() -> Path:
    return config.RESULTS / "per_question"


# ------------------------------------------------------------------------------------------------------ per question
def run_arrays(path: Path, ids: list[str], examples: dict[str, dict] | None) -> dict[str, np.ndarray]:
    """The fields of one evaluation file; `examples` (target task only) adds the lenient score, and on TAT-QA the
    scores with the units mapped."""
    rows = {r["id"]: r for r in read_jsonl(path)}
    if sorted(rows) != ids:
        raise ValueError(f"{path} does not cover the questions of the benchmark")
    recs = [rows[i] for i in ids]
    out = {
        "correct": np.array([bool(r["score"]["primary"]) for r in recs], np.uint8),
        "format_valid": np.array([bool(r["score"].get("format_valid")) for r in recs], np.uint8),
        "tokens": np.array([r["tokens"] for r in recs], np.int32),
        "answer_tag": np.array(["<answer>" in r["text"] for r in recs], np.uint8),
        "no_reasoning": np.array([r["text"].lstrip().startswith("<answer>") for r in recs], np.uint8),
        "truncated": np.array([r["finish"] == "length" for r in recs], np.uint8),
    }
    if examples is not None:
        out["lenient"] = np.array([bool(scoring.lenient(examples[r["id"]], r.get("final", r["text"]))) for r in recs],
                                  np.uint8)
        if all(examples[i]["dataset"] == "tatqa" for i in ids):
            def mapped(text: str, gold: dict) -> bool:
                return scoring.score_tatqa(text, gold, map_units=True)["primary"] == 1

            finals = [(r.get("final", r["text"]), examples[r["id"]]["gold"]) for r in recs]
            out["correct_units"] = np.array([mapped(text, gold) for text, gold in finals], np.uint8)
            out["lenient_units"] = np.array([mapped(text[text.rfind("<answer>"):], gold) if "<answer>" in text
                                             else False for text, gold in finals], np.uint8)
            if (out["correct_units"] < out["correct"]).any() or (out["lenient_units"] < out["lenient"]).any():
                raise ValueError(f"{path}: mapping the units lowers a score")
    return out


def _finished(directory: Path, benches: list[str]) -> bool:
    return (directory / "done.json").exists() and all((directory / f"{b}-test.jsonl").exists() for b in benches)


def export(*, only_finished: bool = False) -> dict:
    """Write results/per_question/ and results/teachers.json from the run directory. Without `only_finished`, every
    run of configs/experiment.yaml must be evaluated."""
    exp = config.experiment()
    benches = [*exp["datasets"], *exp["unseen"]]
    examples = {b: data.load_index(b, "test") for b in benches}
    ids = {b: sorted(examples[b]) for b in benches}
    questions = {}
    for b in benches:
        questions[f"{b}/ids"] = np.array(ids[b])
        questions[f"{b}/documents"] = np.unique(np.array([examples[b][i]["document_id"] for i in ids[b]]),
                                                return_inverse=True)[1].astype(np.int32)
    save_npz(per_question_dir() / "questions.npz", questions)

    missing, written = [], {}
    for m in exp["models"]:
        arrays = {}
        for method in config.UNTRAINED:
            d = config.eval_dir(m, method)
            if not _finished(d, benches):
                missing.append(f"{m}/{method}")
                continue
            for b in benches:
                got = run_arrays(d / f"{b}-test.jsonl", ids[b], examples[b] if b in exp["datasets"] else None)
                arrays.update({f"{method}/{b}/{field}": v for field, v in got.items()})
        if arrays:
            save_npz(per_question_dir() / f"{m}_untrained.npz", arrays)
            written[f"{m}_untrained"] = len(arrays)
    for m, ds in config.SETTINGS:
        arrays = {}
        for method, seeds in config.method_seeds().items():
            for seed in seeds:
                d = config.eval_dir(m, method, ds=ds, seed=seed)
                if not _finished(d, config.eval_benchmarks(ds)):
                    missing.append(f"{m}/{ds}/{method}/s{seed}")
                    continue
                for b in config.eval_benchmarks(ds):
                    got = run_arrays(d / f"{b}-test.jsonl", ids[b], examples[b] if b == ds else None)
                    arrays.update({f"{method}/s{seed}/{b}/{field}": v for field, v in got.items()})
        if arrays:
            save_npz(per_question_dir() / f"{m}_{ds}.npz", arrays)
            written[f"{m}_{ds}"] = len(arrays)
        print(f"exported {m}/{ds}", flush=True)
    if missing and not only_finished:
        raise SystemExit(f"{len(missing)} runs are not evaluated (use --only-finished to skip them): "
                         + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))

    teachers = {"settings": {}, "simple_formats": {}}
    for m, ds in config.SETTINGS:
        if (config.setting_dir(m, ds) / "teachers" / "edits.json").exists():
            teachers["settings"][f"{m}/{ds}"] = teacher_statistics(m, ds)
    for m, datasets in exp.get("simple_formats", {}).items():
        for ds in datasets:
            if config.scores_file(m, ds).exists():
                teachers["simple_formats"][f"{m}/{ds}"] = simple_format_statistics(m, ds)
    write_json(config.RESULTS / "teachers.json", teachers)
    return {"written": written, "missing": missing, "teachers": sorted(teachers["settings"]),
            "simple_formats": sorted(teachers["simple_formats"])}


# ---------------------------------------------------------------------------------------------------------- teachers
def normalize_line(line: str) -> str:
    """A line in a form that groups its variants: lower case, numbers -> N, list markers and emphasis removed."""
    s = _NUMBER.sub("N", line.strip().lower().replace("**", "").replace("`", ""))
    s = re.sub(r"^(?:[-*•]\s*|N[.)]\s*|step N\s*[:.]\s*)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def removed_lines(before: str, after: str) -> list[str]:
    """The non-empty lines of `before` that are not in `after`."""
    a, b = before.splitlines(), after.splitlines()
    out = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            out += [x for x in a[i1:i2] if x.strip()]
    return out


def _format_word_share(lines: list[str]) -> float:
    return 100 * sum(bool(FORMAT_WORDS.search(x)) for x in lines) / max(len(lines), 1)


def teacher_statistics(m: str, ds: str) -> dict:
    """The teacher pools of one setting, the share of the reasoning that each edit removes, and how many of the deleted
    lines contain format words."""
    tdir = config.setting_dir(m, ds) / "teachers"
    summary, edited = read_json(tdir / "summary.json"), read_json(tdir / "edits.json")
    normalized = {r["id"]: r["target"] for r in read_jsonl(tdir / "normalized.jsonl")}
    plain = {r["id"]: r["target"] for r in read_jsonl(tdir / "rationalization.jsonl")}
    every = [x for t in normalized.values() for x in edits.reasoning_text(t).splitlines() if x.strip()]
    out = {"questions": summary["questions"], "solved_by_sampling": summary["solved_by_sampling"],
           "unsolved": summary["unsolved"], "recovered_by_rationalization": summary["recovered_by_rationalization"],
           "pools": {name: pool["teachers"] for name, pool in summary["pools"].items()},
           "answer_lines_normalized": summary["pools"]["normalized"]["answer_line_rewritten"],
           "reasoning_tokens": edited["reasoning_tokens"],
           "all_reasoning_lines": {"lines": len(every), "format_words_pct": _format_word_share(every)},
           "edits": {}}
    keep = ("teachers", "edited", "tokens", "share_of_reasoning", "share_before_check", "kept_by_check", "no_reasoning")
    for name, info in edited["pools"].items():
        entry = {k: info[k] for k in keep if k in info}
        if name in DELETION_POOLS:
            before = plain if name in PLAIN_POOLS else normalized
            by_teacher = [removed_lines(before[p["id"]], p["target"])
                          for p in read_jsonl(tdir / f"{name}.jsonl") if p.get("edited")]
            removed = [x for lines in by_teacher for x in lines]
            top = Counter(n for lines in by_teacher for n in {normalize_line(x) for x in lines}).most_common(5)
            entry.update(deleted_lines=len(removed), format_words_pct=_format_word_share(removed),
                         top_lines=[[line, 100 * k / max(len(by_teacher), 1)] for line, k in top])
        out["edits"][name] = entry
    return out


def simple_format_statistics(m: str, ds: str) -> dict:
    """A dataset with a simple answer format: the share of the reasoning tokens of the normalized teachers that FTrim's
    rule selects (before the check of the edited teacher)."""
    rule = config.experiment()["edits"]["ftrim"]
    teachers = total = deleted = with_talk = 0
    for _, tt, full, bare in edits.scored_teachers(m, ds):
        teachers += 1
        if tt is None:
            continue
        lines = core.format_talk_lines(full, bare, tt.lines, threshold_bits=rule["threshold_bits"],
                                       strict=rule.get("strict", False))
        total += tt.n_reason
        deleted += sum(tt.lines[k][1] - tt.lines[k][0] for k in lines)
        with_talk += bool(lines)
    return {"teachers": teachers, "reasoning_tokens": total, "ftrim_share": deleted / total,
            "teachers_with_format_talk": with_talk}


# ------------------------------------------------------------------------------------------------------------ loading
class Results:
    """Read access to results/per_question/ for the report."""

    def __init__(self, directory: Path | None = None):
        self.dir = directory or per_question_dir()
        if not (self.dir / "questions.npz").exists():
            raise SystemExit(f"{self.dir} has no results; run `python -m ftrim export-results` first")
        self._files: dict[str, dict[str, np.ndarray]] = {}
        self.questions = self._load("questions")

    def _load(self, name: str) -> dict[str, np.ndarray]:
        if name not in self._files:
            path = self.dir / f"{name}.npz"
            if path.exists():
                with np.load(path) as z:
                    self._files[name] = {k: z[k] for k in z.files}
            else:
                self._files[name] = {}
        return self._files[name]

    def documents(self, bench: str) -> np.ndarray:
        return self.questions[f"{bench}/documents"]

    def seeds(self, m: str, ds: str, method: str) -> list[int]:
        """The evaluated seeds of a method in a setting (an untrained row: one run, given as seed 0)."""
        if method in config.UNTRAINED:
            return [0] if f"{method}/{ds}/correct" in self._load(f"{m}_untrained") else []
        found = {int(k.split("/")[1][1:]) for k in self._load(f"{m}_{ds}") if k.startswith(f"{method}/s")}
        return sorted(found)

    def matrix(self, m: str, ds: str, method: str, bench: str, field: str = "correct",
               seeds: list[int] | None = None) -> np.ndarray | None:
        """seeds x questions of one field (correct, lenient, format_valid, ...: percent; tokens: count), or None if
        a run is missing. `seeds`: default, the seeds of configs/experiment.yaml."""
        scale = 1.0 if field == "tokens" else 100.0
        if method in config.UNTRAINED:
            got = self._load(f"{m}_untrained").get(f"{method}/{bench}/{field}")
            return None if got is None else scale * got[None].astype(float)
        wanted = config.method_seeds()[method] if seeds is None else seeds
        rows = [self._load(f"{m}_{ds}").get(f"{method}/s{seed}/{bench}/{field}") for seed in wanted]
        if not rows or any(r is None for r in rows):
            return None
        return scale * np.stack(rows).astype(float)
