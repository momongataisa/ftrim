# Results of the paper's runs

Everything that `python -m ftrim report` needs to rebuild the tables of the paper and of the README, without the
run directory and without a GPU.

| File | Content |
|:--|:--|
| [`RESULTS.md`](RESULTS.md) | the tables: target accuracy, generalization accuracy, gains of FTrim, the threshold, normalization only, what FTrim deletes, what the models learn, simple answer formats, TAT-QA units |
| `tables.json` | the same numbers and more, machine-readable: every seed, each generalization benchmark, statistics of the outputs |
| `per_question/` | one value per question for every evaluated run (2.9 MB) |
| `teachers.json` | the teacher pools of every setting and what each edit deletes |
| `mechanism.json` | the analysis of Sect. 5 and Fig. 2 (`python -m ftrim mechanism summary`) |
| [`reproduction/`](reproduction/) | `per_question/`, `teachers.json` and `mechanism.json` of a run of this repository from scratch ([docs/REPRODUCTION.md](../docs/REPRODUCTION.md)) |

`RESULTS.md` and `tables.json` are written by `python -m ftrim report`; the other files by
`python -m ftrim export-results` and `python -m ftrim mechanism summary` from a run directory.

## Per-question results

```
per_question/questions.npz           <benchmark>/ids, <benchmark>/documents
per_question/<model>_untrained.npz   <method>/<benchmark>/<field>            method: untrained, thinking
per_question/<model>_<dataset>.npz   <method>/s<seed>/<benchmark>/<field>    the model fine-tuned on <dataset>
```

Every field holds one value per question of the benchmark, in the order of its `ids`:

| Field | Meaning |
|:--|:--|
| `correct` | 1 if the output is correct: strict EM on TAT-QA and DROP ([Evaluators](../docs/DATA.md#evaluators)), the rule of the benchmark on the others ([unseen benchmarks](../docs/DATA.md#unseen-benchmarks-evaluation-only)) |
| `lenient` | target task only: last-tag EM, 1 if the output cut at its last answer tag is correct |
| `correct_units`, `lenient_units` | TAT-QA only: `correct` and `lenient` with the units `ratio` and `percentage_point` mapped to TAT-QA scales (the paper's runs count them as wrong) |
| `format_valid` | 1 if the output follows the answer format (TAT-QA, DROP and FinanceReasoning; 0 on the other benchmarks, whose prompts define no strict format) |
| `tokens` | number of generated tokens |
| `answer_tag` | 1 if the output contains an answer tag |
| `no_reasoning` | 1 if the output starts with its answer tag |
| `truncated` | 1 if the generation stopped at the token limit |

`documents` is the index of the table or passage of each question, the unit that the bootstrap resamples. The files
contain no question or answer text.

```python
import numpy as np

runs = np.load("results/per_question/gemma_tatqa.npz")
ftrim = np.stack([runs[f"ftrim/s{seed}/tatqa/correct"] for seed in (17, 42, 89)])   # seeds x questions
print(100 * ftrim.mean())                                                           # 69.91: Gemma-TAT in Table 1

from ftrim.results import Results                      # the same through the package, in percent
print(Results().matrix("gemma", "tatqa", "ftrim", "tatqa").mean())
```

Methods: `untrained`, `thinking`, `sft`, `rejection`, `rationalization`, `normalized` (Normalization only),
`random_deletion`, `stm`, `ftrim`, and the thresholds `ftrim_half_bit`, `ftrim_two_bits`, `ftrim_zero` (seed 17).
Benchmarks: `tatqa`, `drop`, `finance_reasoning`, `gsm8k`, `ifeval`, `arc_challenge`.
