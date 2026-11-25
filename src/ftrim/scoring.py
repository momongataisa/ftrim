"""Scorers: the verifier V(q, y) of the paper. An output that breaks the format scores 0 (it stays in the denominator).

* TAT-QA / FinanceReasoning: the answer + unit grammar of the format instructions, then the official TAT-QA EM / F1
  (unit as scale) or the FinanceReasoning numeric tolerance.
* DROP: exactly one <answer> tag at the very end, then the official DROP evaluator (max over the validated answers).
* GSM8K / ARC-Challenge: the tag content, else the last number / option letter. IFEval: lm-evaluation-harness.
* `primary` is the strict score that every table reports; `lenient` rescores the last answer tag of an output.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .prompts import UNITS

# ----------------------------------------------------------------------------------------------- answer + unit grammar
_TERMINATORS = re.compile(r"(?:<\|im_end\|>|<\|eot_id\|>|</s>|<end_of_turn>)\s*$")
_FINAL = re.compile(r"<answer>(.*?)</answer>\s*<unit>(.*?)</unit>\s*\Z", re.DOTALL)
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


@dataclass(frozen=True)
class Parsed:
    kind: str | None              # number | boolean | span | multi_span (None when nothing parsed)
    value: object = None
    unit: str | None = None
    error: str | None = None      # None = format valid

    @property
    def valid(self) -> bool:
        return self.error is None and self.kind is not None


def parse_answer_unit(text: str) -> Parsed:
    """The strict final-answer grammar of the finance format instructions; uses no gold, question or expected type."""
    if not isinstance(text, str):
        return Parsed(None, error="not_text")
    text = _TERMINATORS.sub("", text.rstrip())
    start = text.find("<answer>")
    if start >= 0 and text[:start].count("```") % 2:
        return Parsed(None, error="terminal_tags_in_code_fence")
    if (text.count("<answer>") == text.count("</answer>") == 1 and "<unit" not in text and "</unit" not in text
            and re.search(r"<answer>(.*?)</answer>\s*$", text, re.DOTALL)):
        trial = parse_answer_unit(text + "<unit>none</unit>")
        return Parsed(trial.kind, trial.value, trial.unit, "missing_unit")
    tags = re.findall(r"<\s*/?\s*(?:answer|unit)\b[^>]*>", text, re.IGNORECASE)
    starts = re.findall(r"<\s*/?\s*(?:answer|unit)\b", text, re.IGNORECASE)
    if len(starts) != 4 or tags != ["<answer>", "</answer>", "<unit>", "</unit>"]:
        return Parsed(None, error="missing_duplicate_or_malformed_tags")
    match = _FINAL.search(text)
    if not match:
        return Parsed(None, error="unclosed_tags_or_trailing_text")
    raw, unit = (part.strip() for part in match.groups())
    if not raw or not unit:
        return Parsed(None, error="empty_answer_or_unit")
    if "<" in raw or ">" in raw or "<" in unit or ">" in unit:
        return Parsed(None, error="nested_tag")
    if unit not in UNITS:
        return Parsed(None, error="unknown_unit")
    if len(raw) > 100_000:
        return Parsed(None, error="answer_too_large")
    if _NUMBER.fullmatch(raw):
        try:
            number = Decimal(raw)
        except InvalidOperation:
            return Parsed(None, error="invalid_number")
        if not number.is_finite() or abs(number.adjusted()) > 1000:
            return Parsed(None, error="nonfinite_or_oversized_number")
        return Parsed("number", number, unit)
    if raw in {"yes", "no"}:
        if unit != "none":
            return Parsed(None, error="boolean_with_unit")
        return Parsed("boolean", raw == "yes", unit)
    if raw.casefold() in {"yes", "no", "true", "false"}:
        return Parsed(None, error="boolean_must_be_lowercase_yes_no")
    if raw.startswith(('"', "[")):
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            return Parsed(None, error="invalid_span_json")
        if isinstance(value, str) and value.strip():
            return Parsed("span", value, unit)
        if isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value):
            return Parsed("multi_span", tuple(value), unit)
        return Parsed(None, error="invalid_span_type_or_empty_span")
    if (raw.casefold() in {"nan", "inf", "infinity", "+inf", "-inf", "null", "none"}
            or re.match(r"[+\-.$€£¥0-9{]", raw)):
        return Parsed(None, error="invalid_number_or_nonfinite")
    return Parsed("span", raw, unit)


def render_answer_unit(kind: str, value, unit: str) -> str:
    """The standard answer line of the finance grammar (used by the answer normalization)."""
    if kind == "number":
        text = str(value)
    elif kind == "boolean":
        text = "yes" if value else "no"
    elif kind in ("span", "multi_span"):
        text = json.dumps(value if kind == "span" else list(value), ensure_ascii=False)
    else:
        raise ValueError(kind)
    line = f"<answer>{text}</answer><unit>{unit}</unit>"
    if not parse_answer_unit(line).valid:
        raise ValueError(f"cannot render a valid answer line for {value!r}")
    return line


# ------------------------------------------------------------------------------------------------ answer-only grammar
_TERMINATORS_ANSWER_ONLY = re.compile(r"(?:<\|im_end\|>|<\|eot_id\|>|</s>|<end_of_turn>|<eos>)\s*$")
_ANSWER_TAGS = re.compile(r"<\s*/?\s*answer\b[^>]*>", re.IGNORECASE)


def parse_answer_only(text: str) -> tuple[str | None, str | None]:
    """(content, None) for exactly one <answer>...</answer> at the very end, else (None, error)."""
    if not isinstance(text, str):
        return None, "not_text"
    text = _TERMINATORS_ANSWER_ONLY.sub("", text.rstrip())
    start = text.find("<answer>")
    if start >= 0 and text[:start].count("```") % 2:
        return None, "terminal_tags_in_code_fence"
    if _ANSWER_TAGS.findall(text) != ["<answer>", "</answer>"]:
        return None, "missing_duplicate_or_malformed_tags"
    match = re.search(r"<answer>(.*?)</answer>\s*\Z", text, re.DOTALL)
    if not match:
        return None, "unclosed_tags_or_trailing_text"
    content = match.group(1).strip()
    if not content:
        return None, "empty_answer"
    if len(content) > 100_000:
        return None, "answer_too_large"
    return content, None


# ---------------------------------------------------------------------------------------------------------- TAT-QA
# TAT-QA has five scales. The verifier reads `none` and the currency, count and time labels of the format instructions
# as no scale. Five units of the instructions have no TAT-QA scale (ratio, percentage_point, basis_point, hundred,
# trillion), and the verifier of the paper counts an answer with one of them as wrong, for the teachers and in the
# evaluation. With `map_units`, ratio counts as no scale and percentage_point as percent, as the official evaluator
# reads these two names; the report gives the test results scored so next to the paper's (results/RESULTS.md).
_DIMENSION_LABELS = frozenset({"count", "years", "months", "days", "shares", "USD_per_share",
                               "USD", "EUR", "GBP", "JPY", "CNY", "INR", "CAD", "AUD"})
_OFFICIAL_SCALES = frozenset({"", "thousand", "million", "billion", "percent"})
MAPPED_UNITS = {"ratio": "", "percentage_point": "percent"}


def _prediction_strings(p: Parsed) -> list[str]:
    if p.kind == "multi_span":
        return list(p.value)
    if p.kind == "boolean":
        return ["yes" if p.value else "no"]
    if p.kind == "number":
        return [format(p.value, "f")]   # the official regex does not read scientific notation
    return [str(p.value)]


def official_tatqa(prediction: list[str], pred_scale: str, gold: dict) -> tuple[float, float]:
    import warnings
    with warnings.catch_warnings():     # the vendored file is unchanged; its regexes warn once at compile time
        warnings.simplefilter("ignore", SyntaxWarning)
        from .vendor.tatqa_metric import (
            add_percent_pred,
            extract_gold_answers,
            get_answer_str,
            get_metrics,
            metric_max_over_ground_truths,
        )
    if not prediction:
        return 0.0, 0.0
    answer, kind = gold["answer"], gold["answer_type"]
    if kind in ("span", "multi-span"):
        answer = list(answer) if isinstance(answer, (list, tuple)) else [str(answer)]
    gold_scale = "" if gold.get("scale") in (None, "none") else gold["scale"]
    kind, answers, scale = extract_gold_answers({"answer": answer, "answer_type": kind, "scale": gold_scale})
    strings = add_percent_pred(get_answer_str(prediction, pred_scale), pred_scale, prediction)
    em, f1 = metric_max_over_ground_truths(get_metrics, strings, get_answer_str(answers, scale))
    return float(em), float(em if kind in ("arithmetic", "count") else f1)


def score_tatqa(text: str, gold: dict, *, map_units: bool = False) -> dict:
    p = parse_answer_unit(text)
    if not p.valid:
        return {"primary": 0.0, "em": 0.0, "f1": 0.0, "format_valid": False, "error": p.error}
    gold_scale = "" if gold.get("scale") in (None, "none") else gold["scale"]
    if gold_scale not in _OFFICIAL_SCALES:
        raise ValueError(f"unsupported TAT-QA gold scale {gold_scale}")
    pred_scale = "" if p.unit == "none" or p.unit in _DIMENSION_LABELS else p.unit
    if map_units:
        pred_scale = MAPPED_UNITS.get(pred_scale, pred_scale)
    if pred_scale not in _OFFICIAL_SCALES:
        return {"primary": 0.0, "em": 0.0, "f1": 0.0, "format_valid": True, "error": "unsupported_prediction_scale"}
    em, f1 = official_tatqa(_prediction_strings(p), pred_scale, gold)
    return {"primary": em, "em": em, "f1": f1, "format_valid": True, "error": None}


# ------------------------------------------------------------------------------------------------------------- DROP
def drop_prediction(content: str):
    """The tag content as the official evaluator's prediction: a JSON array of strings / numbers is a list of answer
    strings; anything else (a malformed array included) is the answer string itself."""
    text = content.strip()
    if text.startswith("["):
        try:
            value = json.loads(text)
        except ValueError:
            return text
        def plain(x) -> bool:
            return isinstance(x, str) or (isinstance(x, (int, float)) and not isinstance(x, bool))

        if isinstance(value, list) and all(plain(x) for x in value):
            return tuple(str(x) for x in value)
    return text


def official_drop(content: str, gold: dict) -> tuple[float, float]:
    from .vendor.drop_eval import answer_json_to_strings, get_metrics
    predicted = drop_prediction(content)
    em = f1 = 0.0
    for answer in gold["answers"]:
        gold_answer, _ = answer_json_to_strings(answer)
        if gold_answer[0].strip() == "":      # as the official evaluate_json
            continue
        e, f = get_metrics(predicted, gold_answer)
        em, f1 = max(em, float(e)), max(f1, float(f))
    return em, f1


def score_drop(text: str, gold: dict) -> dict:
    content, error = parse_answer_only(text)
    if error:
        return {"primary": 0.0, "em": 0.0, "f1": 0.0, "format_valid": False, "error": error}
    em, f1 = official_drop(content, gold)
    return {"primary": em, "em": em, "f1": f1, "format_valid": True, "error": None}


# ------------------------------------------------------------------------------------------------ FinanceReasoning
def score_finance_reasoning(text: str, gold: dict) -> dict:
    p = parse_answer_unit(text)
    if not p.valid:
        return {"primary": 0.0, "format_valid": False, "error": p.error}
    answer = gold["answer"]
    if gold["answer_type"] == "boolean":
        reference = answer if isinstance(answer, bool) else str(answer).lower() in {"yes", "true"}
        ok = p.kind == "boolean" and p.value is reference
        return {"primary": float(ok), "format_valid": True, "error": None}
    if p.kind != "number":
        return {"primary": 0.0, "format_valid": True, "error": "answer_type_mismatch"}
    reference = Decimal(str(answer))
    ok = abs(p.value - reference) <= abs(reference) * Decimal("0.002")   # the benchmark's 0.2 % tolerance
    return {"primary": float(ok), "format_valid": True, "error": None}


# -------------------------------------------------------------------------------------- simple answer formats (Sect. 5)
SIMPLE_FORMAT_DATASETS = frozenset({"math", "gsm8k_strict", "medcalc", "medqa"})
_PLAIN_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
_MM_DD_YYYY = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})\Z")
_WEEKS_DAYS = re.compile(r"\(?\s*[\"']?\s*([0-9]+)\s*(?:weeks?)?\s*[\"']?\s*,"
                         r"\s*[\"']?\s*([0-9]+)\s*(?:days?)?\s*[\"']?\s*\)?\Z")


def math_equivalent(content: str, gold: str) -> bool:
    """`is_equiv` of hendrycks/math (vendored unchanged; its invalid escape sequence warns once at compile time)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        from .vendor.math_equivalence import is_equiv
    return bool(is_equiv(content, gold))


def medcalc_kind(gold: dict) -> str:
    """integer | decimal | date | weeks_days: the official rule that applies to the gold."""
    kind, answer = gold["answer_type"], str(gold["answer"]).strip()
    if kind == "date":
        return "weeks_days" if _WEEKS_DAYS.match(answer) and "/" not in answer else "date"
    if kind not in {"decimal", "integer"}:
        raise ValueError(f"unknown MedCalc output type {kind}")
    return kind


def medcalc_match(content: str, gold: dict) -> tuple[bool, str | None]:
    """MedCalc-Bench's rules: a decimal within [lower, upper], an integer equal after rounding, a date or a
    (weeks, days) pair equal."""
    content, answer = content.strip(), str(gold["answer"]).strip()
    kind = medcalc_kind(gold)
    if kind == "weeks_days":
        g, p = _WEEKS_DAYS.match(answer), _WEEKS_DAYS.match(content)
        if not p:
            return False, "not_weeks_days"
        return (int(p.group(1)), int(p.group(2))) == (int(g.group(1)), int(g.group(2))), None
    if kind == "date":
        g, p = _MM_DD_YYYY.match(answer), _MM_DD_YYYY.match(content)
        if not p:
            return False, "not_a_date"
        return tuple(map(int, p.groups())) == tuple(map(int, g.groups())), None
    if not _PLAIN_NUMBER.match(content):
        return False, "not_a_plain_number"
    try:
        value = Decimal(content)
    except InvalidOperation:
        return False, "invalid_number"
    if not value.is_finite():
        return False, "nonfinite_number"
    if kind == "integer":
        return round(float(value)) == int(Decimal(answer)), None
    return Decimal(gold["lower_limit"]) <= value <= Decimal(gold["upper_limit"]), None


def simple_correct(ds: str, content: str, gold: dict) -> tuple[bool, str | None]:
    """(correct, error) of an answer content under the dataset's own rule."""
    if ds == "math":
        return math_equivalent(content, str(gold["answer"])), None
    if ds == "gsm8k_strict":
        content = content.strip()
        if not _PLAIN_NUMBER.match(content):
            return False, "not_a_plain_number"
        return Decimal(content) == Decimal(str(gold["answer"]).replace(",", "")), None
    if ds == "medcalc":
        return medcalc_match(content, gold)
    if ds == "medqa":
        content = content.strip()
        if not re.fullmatch(r"[A-Z]", content):
            return False, "not_an_option_letter"
        return content == str(gold["answer"]).strip(), None
    raise ValueError(ds)


def score_simple(ds: str, text: str, gold: dict) -> dict:
    """Strict (primary): exactly one <answer> tag at the very end, then the dataset's rule; `last_tag`: the same rule on
    the last tag anywhere in the output."""
    content, error = parse_answer_only(text)
    last = _last_tag(text) if isinstance(text, str) else None
    last_ok = float(last is not None and simple_correct(ds, last, gold)[0])
    if error:
        return {"primary": 0.0, "format_valid": False, "error": error, "last_tag": last_ok}
    ok, why = simple_correct(ds, content, gold)
    return {"primary": float(ok), "format_valid": True, "error": why, "last_tag": last_ok}


# --------------------------------------------------------------------------------------------- GSM8K / ARC / IFEval
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_tag(text: str) -> str | None:
    found = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return found[-1].strip() if found else None


def _last_number(s: str | None) -> float | None:
    if s is None:
        return None
    found = _NUM.findall(s.replace("$", ""))
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def score_gsm8k(text: str, gold: dict) -> dict:
    tag = _last_tag(text)
    value = _last_number(tag) if tag is not None else None
    lenient = value if value is not None else _last_number(text)
    return {"primary": float(lenient is not None and abs(lenient - gold["answer"]) < 1e-6),
            "strict": float(value is not None and abs(value - gold["answer"]) < 1e-6)}


def score_arc(text: str, gold: dict) -> dict:
    tag = _last_tag(text)
    pick = None
    if tag is not None:
        m = re.search(r"\b([A-E1-5])\b", tag)
        pick = m.group(1) if m else None
    if pick is None:
        found = re.findall(r"\b([A-E1-5])\b", text[-200:])
        pick = found[-1] if found else None
    return {"primary": float(pick == gold["answer"]), "strict": float(tag is not None and pick == gold["answer"])}


def score_ifeval(text: str, gold: dict) -> dict:
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0                 # language checks must not vary between runs
    from lm_eval.tasks.ifeval import utils
    doc = {"key": gold["key"], "prompt": None, "instruction_id_list": gold["instruction_id_list"],
           "kwargs": gold["kwargs"]}
    random.seed(gold["key"])                 # lm_eval fills absent kwargs at random: fix them per question
    return_value = utils.process_results({**doc, "prompt": gold.get("prompt", "")}, [text])
    return {"primary": float(return_value["prompt_level_strict_acc"]),
            "loose": float(return_value["prompt_level_loose_acc"])}


def score(example: dict, text: str) -> dict:
    """Score one model answer against the example's gold."""
    ds, gold = example["dataset"], example["gold"]
    if ds == "tatqa":
        return score_tatqa(text, gold)
    if ds == "drop":
        return score_drop(text, gold)
    if ds == "finance_reasoning":
        return score_finance_reasoning(text, gold)
    if ds == "gsm8k":
        return score_gsm8k(text, gold)
    if ds == "arc_challenge":
        return score_arc(text, gold)
    if ds == "ifeval":
        return score_ifeval(text, {**gold, "prompt": example["prompt"]})
    if ds in SIMPLE_FORMAT_DATASETS:
        return score_simple(ds, text, gold)
    raise ValueError(f"no scorer for {ds}")


def lenient(example: dict, text: str) -> float:
    """The strict score of the output cut at its last answer tag: what an output scores when everything before its
    final answer is ignored (0 without a tag)."""
    cut = text.rfind("<answer>")
    return 0.0 if cut < 0 else float(score(example, text[cut:])["primary"])


def thinking_final_answer(text: str, model_id: str) -> str:
    """The answer part of a thinking-mode output: after </think> (Qwen), after the thought channel (Gemma); empty if the
    thinking block never closes."""
    if model_id.startswith("Qwen/"):
        return text.split("</think>", 1)[1] if "</think>" in text else ""
    if model_id.startswith("google/gemma-"):
        if "<|channel>thought" in text:
            return text.split("<channel|>", 1)[1] if "<channel|>" in text else ""
        return text
    raise ValueError(f"no thinking protocol for {model_id}")
