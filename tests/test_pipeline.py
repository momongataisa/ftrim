"""The settings of the experiment and the job graph of the pipeline."""
import math

import pytest

from ftrim import cli, config, core, pipeline


def test_the_paper_settings():
    exp = config.experiment()
    assert config.SETTINGS == [("qwen", "tatqa"), ("qwen", "drop"), ("gemma", "tatqa"), ("gemma", "drop")]
    assert exp["seeds"] == [17, 42, 89]
    assert exp["sampling"]["k"] == 4 and exp["sampling"]["temperature"] == 0.7 and exp["sampling"]["top_p"] == 0.9
    assert exp["edits"]["ftrim"] == {"threshold_bits": 1.0}
    assert exp["edits"]["ftrim_half_bit"]["threshold_bits"] == 0.5
    assert exp["edits"]["ftrim_two_bits"]["threshold_bits"] == 2.0
    assert exp["edits"]["random_deletion"]["like"] == "ftrim"
    assert exp["edits"]["stm"]["tau"] == 2.5
    assert exp["training"]["lora"]["r"] == 16 and exp["training"]["lora"]["alpha"] == 32
    assert exp["training"]["lr"] == 2e-5 and exp["training"]["batch"] == 32 and exp["training"]["horizon_factor"] == 3
    assert exp["statistics"] == {"resamples": 20000, "seed": 271828}
    assert math.isclose(core.LN2, math.log(2))


def test_methods_and_their_seeds():
    main, ablation = config.method_seeds("main"), config.method_seeds("ablation")
    assert list(main) == ["sft", "rejection", "rationalization", "random_deletion", "stm", "ftrim"]
    assert all(seeds == [17, 42, 89] for seeds in main.values())
    assert ablation == {"normalized": [17, 42, 89], "ftrim_half_bit": [17], "ftrim_two_bits": [17], "ftrim_zero": [17]}
    assert config.all_methods()[:2] == ["untrained", "thinking"]
    assert set(config.experiment()["edits"]) <= set(config.trained_methods())      # every edited pool is trained
    assert config.eval_benchmarks("drop") == ["drop", "finance_reasoning", "gsm8k", "ifeval", "arc_challenge"]
    assert config.eval_benchmarks(None)[:2] == ["tatqa", "drop"]


def test_the_job_graph():
    jobs = {j.name: j for j in pipeline.jobs(config.SETTINGS, config.method_seeds("main"))}
    assert len(jobs) == 1 + 2 * 2 + 4 * 5 + 4 * 6 * 3 * 2          # data, untrained rows, teacher stages, fit + eval
    assert jobs["gemma:tatqa:rationalize"].deps == ["gemma:tatqa:sample"]
    assert jobs["gemma:tatqa:score"].deps == ["gemma:tatqa:teachers"]
    assert not jobs["gemma:tatqa:teachers"].gpu and not jobs["gemma:tatqa:edit"].gpu
    assert jobs["qwen:drop:rationalization:s42:fit"].deps == ["qwen:drop:teachers"]     # no edit: the pool of step 2
    assert jobs["qwen:drop:ftrim:s42:fit"].deps == ["qwen:drop:edit"]
    assert jobs["qwen:drop:ftrim:s42:eval"].deps == ["qwen:drop:ftrim:s42:fit"]
    assert jobs["qwen:drop:ftrim:s17:fit"].priority < jobs["qwen:drop:ftrim:s42:fit"].priority
    for job in jobs.values():
        assert all(dep in jobs for dep in job.deps)


def test_options_of_the_pipeline():
    one = pipeline.jobs([("gemma", "tatqa")], {"ftrim": [17]}, thinking=False, mechanism=True, mechanism_seeds=[17])
    names = {j.name for j in one}
    assert "gemma:untrained" in names and "gemma:thinking" not in names
    assert {"gemma:tatqa:format-talk", "gemma:tatqa:probe:s17"} <= names
    assert "gemma:tatqa:stm:s17:fit" in names and "gemma:tatqa:stm:s17:eval" not in names   # adapters for the probe
    probe = next(j for j in one if j.name == "gemma:tatqa:probe:s17")
    assert "gemma:tatqa:format-talk" in probe.deps and "gemma:tatqa:ftrim:s17:fit" in probe.deps
    with pytest.raises(ValueError):
        pipeline.jobs([("gemma", "math")], {"ftrim": [17]})
    simple = {j.name for j in pipeline.simple_format_jobs(["qwen", "gemma"])}
    assert "qwen:medqa:score" in simple and "gemma:medcalc:score" in simple and "gemma:medqa:score" not in simple
    assert not any(name.endswith(":fit") or name.endswith(":edit") for name in simple)


def test_method_groups_on_the_command_line():
    assert cli._methods("main", "") == config.method_seeds("main")
    assert cli._methods("all", "") == config.method_seeds()
    assert cli._methods("ftrim,rationalization", "17") == {"ftrim": [17], "rationalization": [17]}
    with pytest.raises(SystemExit):
        cli._methods("unknown", "")
