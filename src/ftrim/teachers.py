"""Teacher pools (Sect. 2, "Sampling and selection"; lines 3 to 6 of Algorithm 1).

* rejection        the first usable sampled solution of every question that has one (rejection sampling).
* rationalization  + for a question without one, the first usable solution written with the gold answer as a hint.
* normalized       the rationalization pool with every answer line replaced by the gold answer in the form of the
                   format instructions (answer normalization, line 6). FTrim starts from it; random deletion and
                   STM keep the answer lines of the rationalization pool (see edits.py).
* sft              every training question with the gold answer line as the whole target (no reasoning).

A teacher file holds one JSON line per teacher: id (question), source (sample | rationalize | gold), draw, target
[, edited, deleted_tokens, deleted_lines, loss_mask]. `loss_mask` lists response-token positions left out of the loss
(STM); it is aligned with the tokenization of chat.pair(model, prompt, target) and checked again at training time.
"""
from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal

from . import chat, config, data, scoring
from .prompts import build_prompt
from .utils import provenance, read_json, read_jsonl, write_json, write_jsonl

# A usable solution does not mention the hint of rationalization, in either of two ways: the hint itself ...
# (Both patterns are those of the paper's runs and decide which solutions become teachers. They are broader than the
# hint prompt of this release needs, and they also reject a few correct solutions that use one of these words.)
HINT_MENTION = re.compile(r"guidance|hint|skill card|\bcard\b|correct final answer is|blueprint|flowchart", re.I)
# ... or an answer that it was given rather than derived.
GIVEN_ANSWER = re.compile(
    r"(expected|provided|given|target|stated|reference|correct)\s+(final\s+)?answer(?!\s+format)|answer\s+key|"
    r"match(es|ing)? the (expected|given|target|provided|stated)|reverse[- ]engineer|work(ing)? backwards?|"
    r"back[- ]solv|check(ing)? value|earlier attempt|previous attempt|the notes|\bdraft\b", re.I)
CONTROL = re.compile(r"</?think>|<\|[a-z_]+\|>|<(?:start|end)_of_turn>")


def no_reasoning(text: str) -> bool:
    """True if nothing precedes the last answer tag."""
    at = text.rfind("<answer>")
    return not (text[:at] if at >= 0 else text).strip()


def draw_flags(ex: dict, text: str) -> dict:
    """The verifier's verdict and the three exclusions that make a sampled solution usable as a teacher."""
    correct = scoring.score(ex, text)["primary"] == 1
    flags = {"correct": correct, "mentions_hint": bool(HINT_MENTION.search(text) or GIVEN_ANSWER.search(text)),
             "control_tokens": bool(CONTROL.search(text)), "no_reasoning": no_reasoning(text)}
    flags["usable"] = correct and not any(flags[k] for k in ("mentions_hint", "control_tokens", "no_reasoning"))
    return flags


def first_usable(draws: list[dict]) -> dict | None:
    """The lowest-index usable draw (identical texts keep the lowest index, so this is also the deduplicated choice)."""
    return next((d for d in sorted(draws, key=lambda d: d["draw"]) if d["usable"]), None)


# -------------------------------------------------------------------------------------------- standard answer lines
def _plain_number(value) -> Decimal:
    text = format(Decimal(str(value)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return Decimal(text if text not in {"", "-0"} else "0")


def gold_line(ex: dict, text: str | None = None) -> str:
    """nu(g): the gold answer in the standard form of the format instructions. TAT-QA: a number without trailing zeros,
    a JSON string or a JSON array, with the gold scale as unit. DROP: the released number, `day month year`, the span,
    or a JSON array of spans. A MedCalc decimal has no standard form (any value within the limits is correct): there
    the answer line of the solution `text` is kept."""
    gold = ex["gold"]
    if ex["dataset"] == "tatqa":
        unit = gold.get("scale") or "none"
        value = gold["answer"]
        if isinstance(value, bool):
            return scoring.render_answer_unit("boolean", value, unit)
        if isinstance(value, str) and value.lower() in {"yes", "no"}:
            return scoring.render_answer_unit("boolean", value.lower() == "yes", unit)
        if isinstance(value, (list, tuple)):
            if len(value) == 1 and gold["answer_type"] == "span":
                return scoring.render_answer_unit("span", str(value[0]), unit)
            return scoring.render_answer_unit("multi_span", tuple(value), unit)
        if gold["answer_type"] in {"arithmetic", "count"}:
            return scoring.render_answer_unit("number", _plain_number(value), unit)
        return scoring.render_answer_unit("span", str(value), unit)
    if ex["dataset"] == "drop":
        return f"<answer>{str(gold['answer']).strip()}</answer>"
    if ex["dataset"] == "math":
        return f"<answer>{gold['answer']}</answer>"
    if ex["dataset"] == "medqa":
        return f"<answer>{str(gold['answer']).strip()}</answer>"
    if ex["dataset"] == "gsm8k_strict":
        return f"<answer>{str(gold['answer']).replace(',', '')}</answer>"
    if ex["dataset"] == "medcalc":
        kind = scoring.medcalc_kind(gold)
        if kind == "integer":
            return f"<answer>{int(Decimal(str(gold['answer'])))}</answer>"
        if kind == "date":
            return f"<answer>{gold['answer']}</answer>"
        if kind == "weeks_days":
            weeks, days = scoring._WEEKS_DAYS.match(str(gold["answer"]).strip()).groups()
            return f"<answer>({int(weeks)} weeks, {int(days)} days)</answer>"
        content, error = scoring.parse_answer_only(text) if text is not None else (None, "no solution")
        if error:
            raise ValueError(
                f"{ex['id']}: a MedCalc decimal keeps the solution's answer, which is not strict ({error})")
        return f"<answer>{content}</answer>"
    raise ValueError(ex["dataset"])


def normalize(ex: dict, text: str) -> str:
    """Answer normalization: the reasoning y_1..y_n unchanged, then nu(g) in place of the answer line."""
    return text[:text.rfind("<answer>")] + gold_line(ex, text)


def normalized_target(ex: dict, text: str) -> str | None:
    """The normalized teacher, or None if it would fail the verifier, V(q, y_1..n || nu(g)) = 0, or if nu(g) cannot be
    written in the standard form (a TAT-QA gold span with < or >, an empty span, an arithmetic gold that is no number):
    then the teacher keeps its own answer line."""
    try:
        target = normalize(ex, text)
    except (ValueError, ArithmeticError):
        return None
    return target if scoring.score(ex, target)["primary"] == 1 else None


# ------------------------------------------------------------------------------------------------------------ the pools
def build(model_key: str, ds: str) -> dict:
    out = config.setting_dir(model_key, ds) / "teachers"
    if (out / "summary.json").exists():
        return read_json(out / "summary.json")
    limit = config.experiment()["sampling"]["max_pair_tokens"]
    batch = config.experiment()["training"]["batch"]
    examples = data.load_index(ds, data.TRAIN_SPLIT[ds])
    sampled = {r["id"]: r["draws"] for r in read_jsonl(config.draw_file(model_key, ds, "sample"))}
    hinted = {r["id"]: r["draws"] for r in read_jsonl(config.draw_file(model_key, ds, "rationalize"))}
    if set(sampled) != set(examples):
        raise SystemExit("the sampled solutions do not cover the training questions")
    chosen = {"rejection": [], "rationalization": []}
    for qid in sorted(examples):
        d = first_usable(sampled[qid])
        if d is not None:
            chosen["rejection"].append((qid, "sample", d))
            chosen["rationalization"].append((qid, "sample", d))
            continue
        if qid not in hinted:
            raise SystemExit(f"question {qid} has no usable sampled solution and no rationalization draws")
        d = first_usable(hinted[qid])
        if d is not None:
            chosen["rationalization"].append((qid, "rationalize", d))
    summary = {"model": model_key, "dataset": ds, "questions": len(examples),
               "solved_by_sampling": len(chosen["rejection"]), "unsolved": len(hinted),
               "recovered_by_rationalization": len(chosen["rationalization"]) - len(chosen["rejection"]), "pools": {}}
    for method in ("rejection", "rationalization", "normalized"):
        rows, over, rewritten, failed = [], [], 0, []
        for qid, source, d in chosen["rationalization" if method == "normalized" else method]:
            ex, target = examples[qid], d["text"]
            if method == "normalized":
                target = normalized_target(ex, d["text"])
                if target is None:                                  # keep the teacher's own answer line
                    failed.append(qid)
                    target = d["text"]
                rewritten += target != d["text"]
            try:
                p = chat.pair(model_key, build_prompt(ex), target, max_tokens=limit)
            except chat.PairTooLong:
                over.append(qid)
                continue
            rows.append({"id": qid, "source": source, "draw": d["draw"], "target": target,
                         "tokens": len(p.input_ids), "response_tokens": len(p.response)})
        write_jsonl(config.teacher_file(model_key, ds, method), rows)
        summary["pools"][method] = {
            "teachers": len(rows), "steps": len(rows) // batch, "by_source": dict(Counter(r["source"] for r in rows)),
            "over_length": over, "response_tokens": sum(r["response_tokens"] for r in rows),
            **({"answer_line_rewritten": rewritten, "normalization_failed": failed} if method == "normalized" else {})}
    summary["pools"]["sft"] = _build_sft(model_key, ds, examples, limit, batch)
    summary.update(provenance())
    write_json(out / "summary.json", summary)
    return summary


def _build_sft(model_key: str, ds: str, examples: dict, limit: int, batch: int) -> dict:
    """SFT on gold answers: every training question with nu(g) as the whole target."""
    pool, not_correct, over = [], [], []
    for qid in sorted(examples):
        ex = examples[qid]
        try:
            target = gold_line(ex)
        except (ValueError, ArithmeticError):
            not_correct.append(qid)
            continue
        if scoring.score(ex, target)["primary"] != 1:
            not_correct.append(qid)
            continue
        try:
            p = chat.pair(model_key, build_prompt(ex), target, max_tokens=limit)
        except chat.PairTooLong:
            over.append(qid)
            continue
        pool.append({"id": qid, "source": "gold", "draw": None, "target": target, "tokens": len(p.input_ids),
                     "response_tokens": len(p.response)})
    write_jsonl(config.teacher_file(model_key, ds, "sft"), pool)
    return {"teachers": len(pool), "steps": len(pool) // batch, "by_source": {"gold": len(pool)},
            "not_strict_correct": not_correct, "over_length": over,
            "response_tokens": sum(p["response_tokens"] for p in pool)}


def load(model_key: str, ds: str, method: str) -> list[dict]:
    return read_jsonl(config.teacher_file(model_key, ds, method))
