"""The experiment settings (configs/experiment.yaml) and the layout of the run directory.

    runs/<model>/eval/{untrained,thinking}/                 one evaluation per model
    runs/<model>/<dataset>/draws/{sample,rationalize}.jsonl K sampled solutions per training question
    runs/<model>/<dataset>/teachers/<method>.jsonl          one teacher pool per fine-tuned method
    runs/<model>/<dataset>/scores/logprobs.npz              token log-probabilities under the two prompts
    runs/<model>/<dataset>/fits/<method>/s<seed>/adapter    LoRA adapter
    runs/<model>/<dataset>/eval/<method>/s<seed>/           outputs and scores on the test set and the unseen benchmarks
    runs/<model>/<dataset>/mechanism/                       Sect. 5: what the models learn

`FTRIM_RUNS`, `FTRIM_DATA` and `FTRIM_RESULTS` move the run, data and results directories.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from .utils import ROOT

RUNS = Path(os.environ.get("FTRIM_RUNS", ROOT / "runs"))
DATA = Path(os.environ.get("FTRIM_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("FTRIM_RESULTS", ROOT / "results"))    # compact results: all that the report needs
UNTRAINED = ("untrained", "thinking")       # evaluated once per model; `thinking` uses the model's thinking mode
SETTINGS = [("qwen", "tatqa"), ("qwen", "drop"), ("gemma", "tatqa"), ("gemma", "drop")]   # the order of every table


@lru_cache(maxsize=1)
def experiment() -> dict:
    path = Path(os.environ.get("FTRIM_CONFIG", ROOT / "configs" / "experiment.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def model(key: str) -> dict:
    return {"key": key, **experiment()["models"][key]}


def snapshot(key: str) -> Path:
    """Local snapshot of the pinned model revision (downloaded on first use)."""
    from huggingface_hub import snapshot_download
    cfg = model(key)
    try:
        return Path(snapshot_download(cfg["hf_id"], revision=cfg["revision"], local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        return Path(snapshot_download(cfg["hf_id"], revision=cfg["revision"]))


# ------------------------------------------------------------------------------------------------------------- methods
def method_seeds(group: str | None = None) -> dict[str, list[int]]:
    """{fine-tuned method: seeds} of one group of configs/experiment.yaml (`main`, `ablation`) or of all groups."""
    groups = experiment()["methods"]
    out: dict[str, list[int]] = {}
    for name in ([group] if group else list(groups)):
        out.update(groups[name])
    return out


def trained_methods() -> list[str]:
    return list(method_seeds())


def all_methods() -> list[str]:
    return [*UNTRAINED, *trained_methods()]


def simple_format_datasets() -> list[str]:
    """Datasets whose answer is a single number or option (teachers and FTrim's scoring only)."""
    cells = experiment().get("simple_formats", {})
    return sorted({ds for datasets in cells.values() for ds in datasets})


def train_datasets() -> list[str]:
    return [*experiment()["datasets"], *simple_format_datasets()]


def eval_benchmarks(dataset: str | None) -> list[str]:
    """Benchmarks of a method fine-tuned on `dataset`: its test set, then the unseen benchmarks. None: the untrained
    model, evaluated on every target task and every unseen benchmark."""
    exp = experiment()
    return ([dataset] if dataset else list(exp["datasets"])) + list(exp["unseen"])


# ------------------------------------------------------------------------------------------------- run directory layout
def model_dir(m: str) -> Path:
    return RUNS / m


def setting_dir(m: str, ds: str) -> Path:
    return RUNS / m / ds


def draw_file(m: str, ds: str, kind: str) -> Path:
    return setting_dir(m, ds) / "draws" / f"{kind}.jsonl"


def teacher_file(m: str, ds: str, method: str) -> Path:
    return setting_dir(m, ds) / "teachers" / f"{method}.jsonl"


def scores_file(m: str, ds: str) -> Path:
    return setting_dir(m, ds) / "scores" / "logprobs.npz"


def fit_dir(m: str, ds: str, method: str, seed: int) -> Path:
    return setting_dir(m, ds) / "fits" / method / f"s{seed}"


def eval_dir(m: str, method: str, *, ds: str | None = None, seed: int | None = None) -> Path:
    """The untrained rows are evaluated once per model; fine-tuned methods per setting and seed."""
    if method in UNTRAINED:
        return model_dir(m) / "eval" / method
    if ds is None or seed is None:
        raise ValueError(f"{method} needs a dataset and a seed")
    return setting_dir(m, ds) / "eval" / method / f"s{seed}"


def mechanism_dir(m: str, ds: str) -> Path:
    return setting_dir(m, ds) / "mechanism"
