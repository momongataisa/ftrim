"""The two prompts of FTrim: with the format instructions and bare."""
import pytest

from ftrim import prompts

TATQA = {"id": "tatqa:1", "dataset": "tatqa", "question": "What is the change in revenue?",
         "table": [["", "2019", "2018"], ["Revenue", "10", "8"]],
         "paragraphs": [{"source_id": "p1", "text": "In thousands."}],
         "gold": {"answer": 2, "answer_type": "arithmetic", "scale": "thousand"}}
DROP = {"id": "drop:1", "dataset": "drop", "question": "How many yards was the longest field goal?", "table": [],
        "paragraphs": [{"source_id": "passage", "text": "The Bears kicked a 44-yard field goal."}],
        "gold": {"answer": "44", "answer_type": "number", "answers": [{"number": "44", "spans": [], "date": {}}]}}


def test_the_bare_prompt_differs_only_in_the_format_instructions():
    full, bare = prompts.build_prompt(DROP), prompts.build_prompt(DROP, bare=True)
    assert full.replace(prompts.DROP_INSTRUCTIONS, prompts.DROP_BARE) == bare
    assert bare == ("Answer the question about the passage. Reason step by step, showing the values you use, then "
                    "answer.\nEnd with <answer>answer</answer>.\n\nPassage:\nThe Bears kicked a 44-yard field goal."
                    "\n\nQuestion: How many yards was the longest field goal?")
    full, bare = prompts.build_prompt(TATQA), prompts.build_prompt(TATQA, bare=True)
    assert full.replace(prompts.FIN_INSTRUCTIONS + "\n" + prompts.TATQA_NOTE, prompts.FIN_BARE) == bare
    assert bare.startswith(prompts.FIN_TASK + "\nEnd with <answer>value</answer><unit>unit</unit>.\n")
    assert "What is the change in revenue?" in bare and '"rows": [["", "2019", "2018"], ["Revenue", "10", "8"]]' in bare


def test_the_format_instructions_describe_the_answer():
    assert "a number in digits only" in prompts.DROP_INSTRUCTIONS and "JSON" in prompts.DROP_INSTRUCTIONS
    assert "two decimals" in prompts.TATQA_NOTE and "Allowed units" in prompts.FIN_INSTRUCTIONS
    for text in (prompts.DROP_BARE, prompts.FIN_BARE):                # one line with the tags, nothing about the value
        assert "\n" not in text and "<answer>" in text and len(text) < 60


def test_the_hint_prompt_adds_the_gold_answer():
    plain, hint = prompts.build_prompt(DROP), prompts.hint_prompt(DROP)
    assert hint.startswith(plain) and "44" in hint[len(plain):]
    plain, hint = prompts.build_prompt(TATQA), prompts.hint_prompt(TATQA)
    assert hint.startswith(plain) and "2 (in thousands)" in hint[len(plain):]
    assert prompts.hint_value({"answer": ["a", "b"], "answer_type": "multi-span"}) == '["a", "b"]'


def test_unseen_benchmarks_have_no_bare_prompt():
    gsm = {"id": "gsm8k:1", "dataset": "gsm8k", "question": "2 + 3?", "gold": {"answer": "5"}}
    assert prompts.build_prompt(gsm).endswith("Problem: 2 + 3?")
    with pytest.raises(ValueError):
        prompts.build_prompt(gsm, bare=True)


def test_simple_formats():
    ex = {"id": "math:1", "dataset": "math", "question": "What is $1+1$?", "paragraphs": [], "gold": {"answer": "2"}}
    assert prompts.build_prompt(ex, bare=True) == ("Solve the math problem. Reason step by step, then answer.\n"
                                                   "End with <answer>answer</answer>.\n\nProblem:\nWhat is $1+1$?")
    assert prompts.build_prompt(ex).endswith("\n\nProblem:\nWhat is $1+1$?")
