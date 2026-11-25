"""Strict scoring with the official evaluators, and the answer lines of the teachers."""
from decimal import Decimal

from ftrim import scoring
from ftrim.teachers import draw_flags, first_usable, gold_line, normalize, normalized_target


def tatqa(answer, kind, scale=""):
    gold = {"answer": answer, "answer_type": kind, "scale": scale or "none"}
    return {"id": "tatqa:x", "dataset": "tatqa", "gold": gold}


DROP_DATE = {"id": "drop:x", "dataset": "drop", "gold": {"answer": "1941", "answer_type": "date", "answers": [
    {"number": "", "spans": [], "date": {"day": "7", "month": "December", "year": "1941"}},
    {"number": "1941", "spans": [], "date": {"day": "", "month": "", "year": ""}}]}}


def test_answer_and_unit_tags():
    p = scoring.parse_answer_unit("Revenue grew.\n<answer>7.25</answer><unit>percent</unit>")
    assert p.valid and p.kind == "number" and p.value == Decimal("7.25") and p.unit == "percent"
    assert scoring.parse_answer_unit("<answer>7.25</answer>").error == "missing_unit"
    assert scoring.parse_answer_unit("<answer>1</answer><unit>none</unit> <answer>2</answer><unit>none</unit>").error
    assert scoring.parse_answer_unit("<answer>yes</answer><unit>none</unit>").kind == "boolean"
    assert scoring.parse_answer_unit('<answer>["a", "b"]</answer><unit>none</unit>').kind == "multi_span"
    assert scoring.parse_answer_unit("<answer>5%</answer><unit>percent</unit>").error == "invalid_number_or_nonfinite"
    assert scoring.parse_answer_unit("<answer>1</answer><unit>dozen</unit>").error == "unknown_unit"


def test_tatqa_is_scored_with_its_scale():
    ex = tatqa(12.5, "arithmetic", "percent")
    assert scoring.score(ex, "<answer>12.5</answer><unit>percent</unit>")["primary"] == 1
    assert scoring.score(ex, "<answer>12.50</answer><unit>percent</unit>")["primary"] == 1
    assert scoring.score(ex, "<answer>12.5</answer><unit>none</unit>")["primary"] == 0
    span = tatqa(["Product sales"], "span")
    assert scoring.score(span, '<answer>"product sales"</answer><unit>none</unit>')["primary"] == 1


def test_units_without_a_tatqa_scale():
    ex, pct = tatqa(3.61, "arithmetic"), tatqa(1, "arithmetic", "percent")
    ratio, point = "So\n<answer>3.61</answer><unit>ratio</unit>", "So\n<answer>1</answer><unit>percentage_point</unit>"
    assert scoring.score(ex, ratio)["primary"] == scoring.score(pct, point)["primary"] == 0  # the paper's runs
    assert scoring.score_tatqa(ratio, ex["gold"], map_units=True)["primary"] == 1
    assert scoring.score_tatqa(point, pct["gold"], map_units=True)["primary"] == 1
    assert scoring.score_tatqa(point, ex["gold"], map_units=True)["primary"] == 0
    assert scoring.score_tatqa(ratio.replace("ratio", "basis_point"), ex["gold"], map_units=True)["primary"] == 0


def test_drop_is_scored_by_the_official_evaluator():
    assert scoring.score(DROP_DATE, "So <answer>1941</answer>")["primary"] == 1
    assert scoring.score(DROP_DATE, "<answer>7 December 1941</answer>")["primary"] == 1
    spans = {"dataset": "drop", "gold": {"answer": '["Smith", "Jones"]', "answer_type": "spans",
                                         "answers": [{"number": "", "spans": ["Smith", "Jones"], "date": {}}]}}
    assert scoring.score(spans, '<answer>["Jones", "Smith"]</answer>')["primary"] == 1


def test_strict_scoring_needs_exactly_one_final_tag():
    assert scoring.score(DROP_DATE, "<answer>1941</answer> done")["format_valid"] is False
    assert scoring.score(DROP_DATE, "<answer>1941</answer> done")["primary"] == 0
    two = "The format needs <answer>...</answer>.\n<answer>1941</answer>"
    assert scoring.score(DROP_DATE, two)["primary"] == 0                   # format talk with a second tag fails
    assert scoring.lenient(DROP_DATE, two) == 1                            # the last tag alone is correct
    assert scoring.lenient(DROP_DATE, "no tag at all") == 0
    assert scoring.lenient(DROP_DATE, "<answer>1942</answer>") == 0


def test_gold_lines_pass_the_verifier():
    for ex in (tatqa(12.50, "arithmetic", "million"), tatqa(["a", "b"], "multi-span"), tatqa(["x y"], "span"),
               tatqa(3, "count")):
        assert scoring.score(ex, "reasoning\n" + gold_line(ex))["primary"] == 1
    assert gold_line(tatqa(12.50, "arithmetic", "million")) == "<answer>12.5</answer><unit>million</unit>"


def test_normalization_replaces_the_answer_line_only():
    ex = tatqa(12.5, "arithmetic", "percent")
    text = "Growth is 12.50 %.\n<answer>12.50</answer><unit>percent</unit>"
    assert normalize(ex, text) == "Growth is 12.50 %.\n<answer>12.5</answer><unit>percent</unit>"
    assert scoring.score(ex, normalize(ex, text))["primary"] == 1
    assert normalized_target(ex, text) == normalize(ex, text)


def test_a_gold_without_standard_form_keeps_the_answer_line():
    # nu(g) cannot be written for these TAT-QA golds; the teacher then keeps its own (correct) answer line
    for ex in (tatqa(["Overdue <30 days"], "span"), tatqa(["a", ""], "multi-span"),
               tatqa("31 January 2029", "arithmetic")):
        assert normalized_target(ex, "reasoning\n<answer>x</answer><unit>none</unit>") is None
    ex = tatqa(12.5, "arithmetic", "percent")             # an earlier answer tag: y_1..n || nu(g) fails the verifier
    assert normalized_target(ex, "Write <answer>v</answer>.\n<answer>12.50</answer><unit>percent</unit>") is None


def test_usable_solutions():
    ex = tatqa(3, "count")
    good = "Count the rows.\n<answer>3</answer><unit>none</unit>"
    assert draw_flags(ex, good)["usable"]
    assert not draw_flags(ex, "<answer>3</answer><unit>none</unit>")["usable"]                         # no reasoning
    assert not draw_flags(ex, "The guidance says 3.\n<answer>3</answer><unit>none</unit>")["usable"]   # hint mentioned
    assert not draw_flags(ex, "Count.\n<answer>4</answer><unit>none</unit>")["usable"]                 # wrong
    assert not draw_flags(ex, "<think>Count.</think>\n<answer>3</answer><unit>none</unit>")["usable"]  # control token
    draws = [{"draw": 0, "usable": False}, {"draw": 1, "usable": True}, {"draw": 2, "usable": True}]
    assert first_usable(draws)["draw"] == 1
    assert first_usable(draws[:1]) is None


def test_simple_answer_formats():
    math = {"dataset": "math", "gold": {"answer": "\\frac{1}{2}"}}
    assert scoring.score(math, "So x = 1/2.\n<answer>\\frac12</answer>")["primary"] == 1
    assert scoring.score(math, "<answer>1/2</answer> done")["format_valid"] is False
    gsm = {"dataset": "gsm8k_strict", "gold": {"answer": "1200"}}
    assert scoring.score(gsm, "<answer>1200</answer>")["primary"] == 1
    assert scoring.score(gsm, "<answer>1,200</answer>")["error"] == "not_a_plain_number"
    medcalc = {"dataset": "medcalc", "gold": {"answer": "17.2", "answer_type": "decimal", "lower_limit": "16.34",
                                              "upper_limit": "18.06"}}
    assert scoring.score(medcalc, "<answer>16.5</answer>")["primary"] == 1
    assert scoring.score(medcalc, "<answer>18.1</answer>")["primary"] == 0
    medqa = {"dataset": "medqa", "gold": {"answer": "C"}}
    assert scoring.score(medqa, "<answer>C</answer>")["primary"] == 1
    assert scoring.score(medqa, "<answer>C. Aspirin</answer>")["error"] == "not_an_option_letter"
