# Reproduction

Three levels, from cheap to expensive:

| Level | What it shows | Cost |
|:--|:--|:--|
| [1. `report --check`](#1-the-numbers-of-the-paper) | the numbers of the paper follow from the per-question results in `results/` | CPU, a few minutes |
| [2. The code against the paper's runs](#2-this-code-against-the-papers-runs) | this code rebuilds the prompts, teachers and edits of the paper's runs exactly | done for this release |
| [3. A run from scratch](#3-a-run-of-this-repository-from-scratch) | sampling, teachers, FTrim, fine-tuning and evaluation with this repository | GPUs, hours |

## Installation and commands

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                   # CPU: the method, the data, the scorers, the statistics and the tables
uv sync --extra gpu --extra ifeval        # GPU: generation, scoring, fine-tuning and evaluation
uv sync --all-extras                      # also matplotlib for scripts/make_figures.py
```

`uv sync` removes the extras that it is not given, so name all that you need. The commands below assume that the
environment is active (`source .venv/bin/activate`); otherwise prefix them with `uv run`. With pip, install a clone in
editable mode, for example `pip install -e ".[gpu,ifeval]"`, because the package reads `configs/` and `results/` from
the repository. `pyproject.toml` and `uv.lock` pin the GPU stack (vLLM 0.23.0, PyTorch 2.11.0, transformers 5.12.1,
PEFT 0.18.1), and `configs/experiment.yaml` pins the model revisions. Every GPU job uses one GPU; we used GPUs
with 80 GB.

The pipeline runs every stage on the free GPUs and can be resumed:

```
prepare-data ─► sample ─► rationalize ─► teachers ─► score ─► edit ─► fit ─► eval ─► export-results ─► report
   (CPU)        (GPU)       (GPU)         (CPU)      (GPU)   (CPU)   (GPU)  (GPU)        (CPU)           (CPU)
```

| Result | Command |
|:--|:--|
| Data: download the pinned sources and convert them | `python -m ftrim prepare-data` |
| Table 1 and Sect. 4 | `python -m ftrim pipeline --gpus 0,1,2,3 --methods main` |
| Normalization only, other thresholds (Sect. 3) | `python -m ftrim pipeline --gpus 0,1,2,3 --methods ablation` |
| What the models learn, Fig. 2 (Sect. 5) | `python -m ftrim pipeline --gpus 0,1,2,3 --methods rationalization,random_deletion,stm,ftrim --seeds 17,42 --mechanism` |
| Where FTrim helps less (Sect. 5), teachers only | `python -m ftrim prepare-data --simple-formats`<br>`python -m ftrim pipeline --gpus 0,1,2,3 --simple-formats` |
| Tables of your runs | `export FTRIM_RESULTS=results/mine`<br>`python -m ftrim export-results`<br>`python -m ftrim mechanism summary`<br>`python -m ftrim report` |
| Fig. 2 (`figures` extra) | `python scripts/make_figures.py` |

`--dry-run` lists the jobs and whether they are done. `--settings gemma/tatqa`, `--methods`, `--seeds` and
`--no-thinking` select a part, and `python -m ftrim --help` lists the single stages. The CPU stages between
`prepare-data` and `report` tokenize the teachers, so they also need the `gpu` extra. Runs go to `runs/`
(`FTRIM_RUNS`), data to `data/` (`FTRIM_DATA`) and the compact results to `results/` or to `FTRIM_RESULTS`.
`results/` holds the paper's runs.

## Compute

From the job logs of the paper's runs, the jobs behind Table 1 took about 114 GPU-hours: fine-tuning 75, evaluation
23, the untrained models with and without thinking 6, sampling and rationalization 9, and FTrim's scoring of the
teachers 1.6 for the four settings. FTrim's scoring is therefore 1.4 % of the GPU time of these runs (Sect. 3 of the
paper). The job logs are not part of this repository. The run from scratch of §3 took 19 GPU-hours.

## 1. The numbers of the paper

`python -m ftrim report --check` recomputes every table from `results/` and compares it with the paper
([`src/ftrim/paper.py`](../src/ftrim/paper.py)). It checks every number of the paper that follows from `results/` or
from `configs/experiment.yaml`, and the values behind statements that the paper makes in words, for example "a
threshold of zero removes most of the reasoning". The continuous integration runs it on every commit. Your own runs
match these numbers only up to the noise of fine-tuning and evaluation (§3), so use plain `report` on them.

## 2. This code against the paper's runs

This repository is a cleaned version of the code of the paper's runs. Names, layout and command line changed; the
behaviour did not. Before the release we checked it against the stored files of these runs:

| Stage | Check | Result |
|:--|:--|:--|
| Data | converted files against the checksums of the paper's runs (`configs/data_checksums.json`) | 22 of 22 files identical |
| Prompts | the prompt with the format instructions, the counterfactual prompt and the hint prompt of every question of every dataset, against the original code | 237,017 of 237,017 identical |
| Usable solutions | the flags of every stored solution (verifier, reasoning, hint mention, control tokens), recomputed | 436,456 of 436,456 solutions identical |
| Teachers and edits | the pools of every method, rebuilt from the stored samples and token scores, in the four settings and the five simple-format settings | all identical |
| Token log-probabilities | the function behind FTrim's scoring and the analysis of Sect. 5, recomputed on a GPU for 120 stored sequences of the analysis | identical |
| Fine-tuning | loss of the first training step of a stored run | identical (0.122186) |
| Analysis of Sect. 5 | selected outputs, deleted lines and the summary from the stored log-probabilities | identical (differences below 1e-14) |
| Tables | `report --check` on the exported results | all values equal |

The stored generations and log-probabilities of the paper's runs are not part of this repository, so these checks
cannot be repeated from the release alone.

## 3. A run of this repository from scratch

We ran the whole pipeline again with the released code, from the download of the data to the tables, for one seed of
four methods in two settings, with the untrained models and the analysis of Sect. 5. The 33 jobs took 5 h 38 min on
four GPUs with 80 GB.

```bash
python -m ftrim prepare-data
python -m ftrim pipeline --gpus 0,1,2,3 --settings gemma/tatqa,qwen/drop \
    --methods rationalization,random_deletion,stm,ftrim --seeds 17 --no-thinking --mechanism
FTRIM_RESULTS=results/reproduction python -m ftrim export-results --only-finished
FTRIM_RESULTS=results/reproduction python -m ftrim mechanism summary
python scripts/update_docs.py
```

The results are in [`results/reproduction/`](../results/reproduction/), and `scripts/update_docs.py` generates the
tables below from them. *Paper*: the paper's run with the same seed.

* The sampled solutions, the rationalization and the teachers before the edits are byte-identical to those of the
  paper's runs.
* FTrim's token scores differ for 2 to 3 % of the tokens, by up to 1.7 nats. FTrim then edits the same teachers,
  except 5 of 11,714 in Gemma-TAT and 1 of 9,034 in Qwen-DROP.
* Greedy decoding does not repeat exactly: the engine computes in bfloat16, and an output can depend on the other
  requests in its batch. Qwen's 11,199 outputs on TAT-QA and DROP are byte-identical to the paper's. For Gemma, 95 to
  99 % of the questions keep their score. A second evaluation of the untrained Gemma, six hours later, scored 58.27
  on TAT-QA, against 57.37 in the paper and 56.89 in this run.
* Fine-tuning repeats its first step exactly and then drifts, so the fine-tuned models are new models. Their target
  accuracy differs from the paper's run with the same seed by −0.15 to +0.78 points. FTrim has the highest target
  accuracy and the shortest outputs of the four methods in both settings, as in Table 1 of the paper. In Qwen-DROP
  the margin over random deletion is 0.01 points.
* The generalization accuracy of a single run moves by up to 0.81 points. Differences of this size between single
  runs say little; the paper reports the mean of three seeds in each setting.
* The analysis of Sect. 5 agrees with the paper's within 0.3 nats.

The untrained models, evaluated again on the same prompts. *Same score*: questions with the same score in both runs.

<!-- results:repro-untrained -->

| Untrained model | Benchmark | Questions | Paper | Reproduction | Difference | Same score (%) |
|:--|:--|--:|--:|--:|--:|--:|
| Qwen3.5-4B | TAT-QA | 1,663 | 62.06 | 62.06 | +0.00 | 100.0 |
| Qwen3.5-4B | DROP | 9,536 | 84.38 | 84.38 | +0.00 | 100.0 |
| Qwen3.5-4B | FinanceReasoning | 2,238 | 77.48 | 77.35 | -0.13 | 99.9 |
| Qwen3.5-4B | GSM8K | 1,319 | 93.78 | 93.78 | +0.00 | 100.0 |
| Qwen3.5-4B | IFEval | 541 | 82.99 | 83.18 | +0.18 | 99.8 |
| Qwen3.5-4B | ARC-Challenge | 1,172 | 94.80 | 94.71 | -0.09 | 99.9 |
| Gemma-4-E4B-it | TAT-QA | 1,663 | 57.37 | 56.89 | -0.48 | 96.5 |
| Gemma-4-E4B-it | DROP | 9,536 | 82.30 | 82.28 | -0.02 | 97.0 |
| Gemma-4-E4B-it | FinanceReasoning | 2,238 | 76.94 | 76.41 | -0.54 | 95.4 |
| Gemma-4-E4B-it | GSM8K | 1,319 | 93.86 | 94.09 | +0.23 | 98.9 |
| Gemma-4-E4B-it | IFEval | 541 | 86.14 | 86.51 | +0.37 | 96.7 |
| Gemma-4-E4B-it | ARC-Challenge | 1,172 | 94.11 | 94.20 | +0.09 | 98.9 |

<!-- /results:repro-untrained -->

The fine-tuned runs: target accuracy, generalization accuracy and the mean output length on the target task.
*Paper's three seeds*: lowest and highest of the paper's three runs of the method.

<!-- results:repro-finetuned -->

| Setting | Method (seed 17) | Target accuracy: paper | Reproduction | Paper's three seeds | Generalization accuracy: paper | Reproduction | Paper's three seeds | Tokens: paper | Reproduction |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen-DROP | + Rationalization | 86.39 | 86.38 | 86.39–86.47 | 86.47 | 86.98 | 86.47–87.00 | 323 | 335 |
| Qwen-DROP | Random deletion | 86.50 | 86.82 | 86.29–86.50 | 86.37 | 86.42 | 86.37–87.00 | 296 | 287 |
| Qwen-DROP | STM | 86.65 | 86.50 | 86.42–86.65 | 87.06 | 86.78 | 86.62–87.24 | 380 | 385 |
| Qwen-DROP | FTrim (ours) | 86.72 | 86.83 | 86.44–86.97 | 86.95 | 87.28 | 86.93–87.28 | 284 | 286 |
| Gemma-TAT | + Rationalization | 66.09 | 66.87 | 66.09–67.47 | 87.64 | 87.08 | 87.06–87.64 | 317 | 321 |
| Gemma-TAT | Random deletion | 66.81 | 66.93 | 66.81–67.35 | 87.18 | 86.97 | 87.04–87.24 | 311 | 310 |
| Gemma-TAT | STM | 67.41 | 67.35 | 66.57–67.41 | 87.04 | 87.18 | 87.04–87.53 | 319 | 321 |
| Gemma-TAT | FTrim (ours) | 69.39 | 69.93 | 69.39–70.29 | 87.85 | 87.04 | 87.36–87.85 | 292 | 292 |

<!-- /results:repro-finetuned -->
