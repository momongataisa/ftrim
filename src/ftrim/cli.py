"""Command line: `python -m ftrim <stage> ...` (or `ftrim <stage> ...`). Every stage is resumable.

    prepare-data    download the pinned sources and convert every dataset                               [CPU]
    sample          K solutions of the untrained model per training question                            [GPU]
    rationalize     K solutions written with the gold answer as a hint, for the unsolved questions      [GPU]
    teachers        the pools: rejection, rationalization, normalized, sft                              [CPU*]
    score           FTrim's scoring: token log-probabilities under both prompts                         [GPU]
    edit            the edited pools: ftrim (and its thresholds), random_deletion, stm                  [CPU*]
    fit             LoRA fine-tuning of one method and seed                                             [GPU]
    eval            evaluation of the untrained model or of one fine-tuned method and seed              [GPU]
    mechanism       Sect. 5: what the models learn (format-talk, probe, summary)                        [GPU / CPU]
    pipeline        run every remaining stage on the free GPUs
    export-results  compact per-question results of the finished runs -> results/                       [CPU*]
    report          the tables of the paper and of the README from results/ -> results/RESULTS.md       [CPU]

[CPU*] stages need no GPU, but they tokenize the teachers: install the `gpu` extra for the tokenizers. `prepare-data`
and `report` run with the base install.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ftrim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    exp = config.experiment()
    models, trainable = sorted(exp["models"]), config.train_datasets()

    p = sub.add_parser("prepare-data", help="download the pinned sources and convert every dataset")
    p.add_argument("--simple-formats", action="store_true", help="the datasets with simple answer formats (Sect. 5)")

    for name in ("sample", "rationalize", "teachers", "score", "edit"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=models)
        p.add_argument("--dataset", required=True, choices=trainable)

    p = sub.add_parser("fit", help="LoRA fine-tuning on the teachers of one method")
    p.add_argument("--model", required=True, choices=models)
    p.add_argument("--dataset", required=True, choices=exp["datasets"])
    p.add_argument("--method", required=True, choices=config.trained_methods())
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-steps", type=int,
                   help="smoke tests only: stop after this many steps (written to fits/<method>/s<seed>-max<steps>)")

    p = sub.add_parser("eval", help="greedy evaluation on the test set and the unseen benchmarks")
    p.add_argument("--model", required=True, choices=models)
    p.add_argument("--method", required=True, choices=config.all_methods())
    p.add_argument("--dataset", choices=exp["datasets"], help="the target task of a fine-tuned method")
    p.add_argument("--seed", type=int)

    p = sub.add_parser("mechanism", help="Sect. 5: what the models learn")
    p.add_argument("stage", choices=("format-talk", "probe", "summary"))
    p.add_argument("--model", choices=models)
    p.add_argument("--dataset", choices=exp["datasets"])
    p.add_argument("--seed", type=int)

    p = sub.add_parser("pipeline", help="run every remaining stage on the free GPUs")
    p.add_argument("--gpus", default="0", help="comma-separated GPU indices (default: 0)")
    p.add_argument("--models", default=",".join(models), help="comma-separated model keys")
    p.add_argument("--datasets", default=",".join(exp["datasets"]), help="comma-separated target tasks")
    p.add_argument("--settings", default="",
                   help="comma-separated model/dataset pairs, e.g. gemma/tatqa,qwen/drop (default: every pair of "
                        "--models and --datasets)")
    p.add_argument("--methods", default="main",
                   help="comma-separated methods or groups of configs/experiment.yaml (main, ablation, all)")
    p.add_argument("--seeds", default="", help="comma-separated seeds (default: the seeds of each method)")
    p.add_argument("--no-untrained", action="store_true", help="skip the evaluation of the untrained model")
    p.add_argument("--no-thinking", action="store_true", help="skip the evaluation of the thinking mode (long)")
    p.add_argument("--mechanism", action="store_true",
                   help="add the mechanism analysis of Sect. 5 (for --seeds, or for the seeds of the config)")
    p.add_argument("--simple-formats", action="store_true",
                   help="run the settings with simple answer formats (Sect. 5) instead")
    p.add_argument("--dry-run", action="store_true", help="list the jobs and whether they are done")

    p = sub.add_parser("export-results", help="compact per-question results of the finished runs -> results/")
    p.add_argument("--only-finished", action="store_true", help="skip the runs that are missing instead of failing")
    p = sub.add_parser("report", help="the tables from results/ -> results/RESULTS.md")
    p.add_argument("--check", action="store_true", help="fail if the numbers of the paper are not reproduced")
    p.add_argument("--seeds", default="", help="use only the runs with these seeds (default: the seeds of each method)")
    return ap


def _methods(spec: str, seeds: str) -> dict[str, list[int]]:
    """{method: seeds} of a comma-separated list of methods and groups."""
    groups = config.experiment()["methods"]
    every = config.method_seeds()
    out: dict[str, list[int]] = {}
    for name in [x for x in spec.split(",") if x]:
        if name == "all":
            out.update(every)
        elif name in groups:
            out.update(groups[name])
        elif name in every:
            out[name] = every[name]
        else:
            raise SystemExit(f"unknown method or group: {name}")
    if seeds:
        chosen = [int(s) for s in seeds.split(",")]
        out = {name: chosen for name in out}
    return out


def main(argv: list[str] | None = None) -> None:
    a = _parser().parse_args(argv)
    if a.cmd == "prepare-data":
        from . import data
        result = (data.prepare_simple_formats() if a.simple_formats else data.prepare())["datasets"]
    elif a.cmd in ("sample", "rationalize"):
        from .generation import run_draws
        result = run_draws(a.model, a.dataset, a.cmd)
    elif a.cmd == "teachers":
        from .teachers import build
        result = build(a.model, a.dataset)
    elif a.cmd == "score":
        from .edits import score
        result = score(a.model, a.dataset)
    elif a.cmd == "edit":
        from .edits import build
        result = build(a.model, a.dataset)
    elif a.cmd == "fit":
        from .train import fit
        result = fit(a.model, a.dataset, a.method, a.seed, max_steps=a.max_steps)
    elif a.cmd == "eval":
        from .generation import run_eval
        if a.method not in config.UNTRAINED and (a.dataset is None or a.seed is None):
            raise SystemExit("a fine-tuned method needs --dataset and --seed")
        result = run_eval(a.model, a.method, ds=a.dataset, seed=a.seed)
    elif a.cmd == "mechanism":
        from . import mechanism
        if a.stage == "summary":
            result = mechanism.summary()
        elif a.model is None or a.dataset is None:
            raise SystemExit("this stage needs --model and --dataset")
        elif a.stage == "format-talk":
            result = mechanism.format_talk(a.model, a.dataset)
        elif a.seed is None:
            raise SystemExit("the probe needs --seed")
        else:
            result = mechanism.probe(a.model, a.dataset, a.seed)
    elif a.cmd == "pipeline":
        from . import pipeline
        models = [m for m in a.models.split(",") if m]
        if a.simple_formats:
            graph = pipeline.simple_format_jobs(models)
        else:
            datasets = [d for d in a.datasets.split(",") if d]
            pairs = [tuple(x.split("/")) for x in a.settings.split(",") if x]
            pairs = pairs or [(m, d) for m in models for d in datasets]
            graph = pipeline.jobs(pairs, _methods(a.methods, a.seeds), untrained=not a.no_untrained,
                                  thinking=not a.no_thinking, mechanism=a.mechanism,
                                  mechanism_seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None)
        ok = pipeline.run(graph, [int(g) for g in a.gpus.split(",") if g], dry_run=a.dry_run)
        raise SystemExit(0 if ok or a.dry_run else 1)
    elif a.cmd == "export-results":
        from . import results
        result = results.export(only_finished=a.only_finished)
    elif a.cmd == "report":
        from . import report
        result = report.write(check=a.check, seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None)
    else:  # pragma: no cover
        raise SystemExit(a.cmd)
    json.dump(result, sys.stdout, indent=1, ensure_ascii=False, default=str)
    print()


if __name__ == "__main__":
    main()
